# MarketCanvas-Env — Design Write-Up

## Action and State Space Choices

**State Space — Dual Representation:**
The primary observation is a *semantic state* — a structured JSON dict containing all elements with their properties and computed spatial relations (above, below, overlaps, etc.). This is the natural interface for LLM agents since they already reason over structured text. For future multimodal training, a *visual state* (800×600 RGB array) is also rendered via PIL at each step. The semantic state is lossless and cheap to produce; the visual state enables VLM training but is optional.

**Action Space — High-Level Semantic:**
I implemented high-level semantic actions (`add_element`, `move_element`, `change_color`, etc.) rather than low-level mouse/keyboard primitives. The reasoning:

- **Credit assignment**: High-level actions have clear, immediate effects on the canvas state. A `mouse_drag` requires the agent to learn the entire GUI interaction protocol before it can even begin learning design — conflating two unrelated skills.
- **Sample efficiency**: With semantic actions, each step meaningfully changes the design. PPO with a low-level action space would require orders of magnitude more episodes to learn the same policy.
- **Determinism**: Semantic actions are trivially deterministic. Low-level actions introduce ambiguity (what does clicking at pixel (312, 450) mean? It depends on the UI layout, which we'd need to simulate).

The environment exposes `step_semantic()` for named parameters and `step()` for the standard Gymnasium dict interface, making it usable both programmatically and through RL training loops.

## Reward Function Design

The reward is a weighted sum of four components, scaled to [-1, 1]:

| Component | Weight | What It Measures |
|---|---|---|
| Constraint Satisfaction | 40% | Are prompt-required elements present? (headline, CTA, specific colors) |
| Layout Quality | 25% | Overlap penalties, center alignment, bounds checking, visual hierarchy |
| Accessibility (WCAG) | 20% | Text-background contrast ratios per WCAG 2.0 AA standard |
| Completeness | 15% | Canvas utilization, element count, effort (step count) |

**Formula**: `reward = (0.40*C + 0.25*L + 0.20*A + 0.15*K) * 2 - 1`

The constraint parser extracts requirements from natural-language prompts using keyword matching (e.g., "headline" → requires a text element, "yellow" → requires that color). This is intentionally simple — a production system would use an LLM to parse constraints, but for a deterministic training environment, keyword extraction is sufficient and reproducible.

**Anti-reward-hacking measures:**
- Elements below 20×20 pixels don't count toward constraints (prevents invisible micro-elements)
- Elements must be within canvas bounds to be "visible"
- Clutter penalty above 20 elements (prevents spamming elements)
- Single-element designs score low on completeness
- WCAG contrast check catches white-on-white or same-color text tricks (contrast ratio 1:1 = score 0)

**Known loopholes:**
- The keyword matcher is simple — an agent could satisfy "headline" by adding any text element, regardless of content quality
- Layout scoring doesn't consider typographic hierarchy beyond "largest text on top"
- An agent could game completeness by adding exactly 3 medium-sized elements without meaningful content

## Scaling to 10,000 Parallel PPO Rollouts with a VLM

**Anticipated bottlenecks:**

1. **Rendering**: PIL rendering at 800×600 for 10K environments per step is CPU-bound. At ~2ms per render, that's 20 seconds per batch step — PIL is single-threaded and holds the GIL.
2. **VLM inference / KV-cache memory**: The dominant cost. A VLM policy (e.g., LLaVA-13B) with 10K concurrent rollouts means 10K active KV-caches. At ~1GB per sequence for a 13B model, this exceeds even 8×H100 memory without careful memory management.
3. **State serialization**: Pydantic `model_dump()` on 10K canvas states per step creates GC pressure. Python's GIL serializes this work even across threads.

**Redesign for scale:**

- **Vectorized environments via `gymnasium.vector.AsyncVectorEnv`**: This is the standard Gymnasium API for parallel rollouts. Each sub-environment runs in its own process, sidestepping the GIL. With 10K envs, partition into ~100 workers each managing 100 envs. The canvas engine is pure Python with no shared state, so this scales linearly.

- **GPU-accelerated rendering**: Replace PIL with a GPU rasterizer (e.g., `nvdiffrast` or a custom CUDA kernel). The canvas elements are simple rectangles and text — this is a trivial rasterization problem that a GPU handles at >100K frames/second. Alternatively, for the semantic-only path, skip rendering entirely during training and only render for eval.

- **C++ canvas core**: Rewrite the canvas engine and reward computation in C++ with pybind11 bindings. This eliminates per-element Python object overhead — state becomes a flat struct array. EnvPool (from the SAIL-SG team) provides a framework for exactly this: C++-native vectorized environments with zero-copy observation transfer.

- **VLM inference with PagedAttention**: Use vLLM with PagedAttention to serve the VLM policy. PagedAttention manages KV-cache memory like virtual memory pages — non-contiguous physical blocks mapped to logical sequences. This reduces KV-cache waste from ~60% (naive allocation) to <4%, making 10K concurrent sequences feasible on a single node. Continuous batching in vLLM means environment steps don't block waiting for the slowest sequence in a batch.

- **Dual interface — Gymnasium for training, MCP for eval**: During PPO training, use the Gymnasium `step()` interface with `AsyncVectorEnv` for maximum throughput. The MCP server interface is reserved for interactive evaluation — connecting Claude Desktop or another LLM client to a single environment instance for qualitative assessment. This separation means the MCP overhead (HTTP, JSON-RPC) never touches the training hot path.

- **Async pipeline with prefetch**: Decouple environment stepping (CPU) from model inference (GPU) using double-buffered queues. While the VLM processes batch N, environments execute batch N+1's transitions. This hides environment latency behind inference latency (which dominates at ~100ms per step for a 13B VLM).

The key insight is that this environment is *embarrassingly parallel* — no environment instance shares state with another. The real bottleneck is VLM inference memory and latency, which should be addressed through PagedAttention, continuous batching, and async pipelining rather than environment-side optimizations.
