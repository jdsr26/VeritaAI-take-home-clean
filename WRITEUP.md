# MarketCanvas-Env — Design Write-Up

## Why I Chose These Action and State Representations

**State — semantic first, visual second.**
The core observation is a semantic JSON dict: every element on the canvas with its position, size, colors, content, z-index, plus computed spatial relations (above/below/overlaps). I chose this because LLM-based agents already think in structured text — feeding them a JSON blob is the most natural interface. The visual render (800×600 RGB) is there too, mostly for future VLM experiments, but the semantic state is what actually matters for training. It's lossless, cheap to produce, and deterministic — three things a visual render can only approximate.

Both are returned in the observation: `obs["state_json"]` for the semantic state (serialized so it plays nicely with Gym's space contract), and `obs["visual"]` for the pixel array.

**Actions — I went with high-level semantic actions as the primary mode.**
Things like `add_element`, `move_element`, `change_color`, `delete_element`, etc. The main reason: credit assignment. If the agent wants to "add a yellow CTA button at position (275, 350)", that's one action, one reward signal. With a low-level mouse/keyboard space, the same operation becomes a chain of mouse_move → mouse_drag → mouse_click → keyboard_type steps — and the agent has to figure out GUI interaction *before* it can even start learning design. That's two separate skills conflated, and the sample efficiency hit would be brutal for PPO.

That said, I also implemented a low-level action space (mouse_move, mouse_click, mouse_drag, keyboard_type) because the assignment said "one or both." You can toggle between them with `action_mode="semantic"` or `action_mode="low_level"`. The low-level mode has a virtual cursor with hit-testing, drag-to-create, click-to-select, and type-to-edit. It's a simplified version of what a computer-use agent would see, minus the full GUI rendering overhead.

## Reward Function

Four components, weighted and scaled into [-1, 1]:

| Component | Weight | What it checks |
|---|---|---|
| Constraint satisfaction | 40% | Did the agent add what the prompt asked for? (headline, CTA button, specific colors) |
| Layout quality | 25% | Overlap penalties, alignment, bounds checking, visual hierarchy |
| Accessibility | 20% | WCAG 2.0 AA contrast ratios between text and background colors |
| Completeness | 15% | Canvas utilization, element count, minimum effort |

The formula: `total = (0.40×C + 0.25×L + 0.20×A + 0.15×K) × 2 − 1`

I extract constraints from prompts using keyword matching — "headline" means there should be a text element, "yellow CTA" means a yellow-colored shape with button-like text, etc. It's intentionally simple. A production system would parse prompts with an LLM, but for a deterministic training env, regex-based extraction is reproducible and testable.

The environment defaults to **terminal-only reward** (`terminal_reward_only=True`), which matches how the assignment frames it — the scalar reward represents an end-of-episode quality score. But it's configurable: pass `terminal_reward_only=False` for dense per-step shaping if you want it (which is what the demo does, so you can see reward improve step by step).

**Things I did to make reward-hacking harder:**
- Elements under 20×20 pixels don't count (no invisible micro-elements)
- Elements must be within canvas bounds to be "visible"
- Clutter penalty kicks in above 20 elements
- Single-element designs get penalized on completeness
- WCAG contrast check catches white-on-white tricks (ratio 1:1 = score 0)

**Loopholes I'm aware of:**
- The keyword matcher is shallow — an agent can satisfy "headline" by adding *any* text element, regardless of what it says
- Layout scoring doesn't really capture typographic hierarchy beyond "largest text goes near the top"
- An agent could game completeness by adding exactly 3 medium-sized elements with no meaningful content

I'd address these with a learned reward model in production, but for this take-home the heuristic reward is transparent, deterministic, and easy to debug.

## How I'd Scale to 10,000 Parallel PPO Rollouts with a VLM

This is where it gets interesting. The environment itself is embarrassingly parallel — no shared state between instances — so the bottleneck isn't the sim, it's everything around it.

**What would break first:**

1. **Rendering.** PIL at 800×600 for 10K environments is CPU-bound and single-threaded. At ~2ms per render, that's 20 seconds per batch — unacceptable. I'd replace PIL with a GPU rasterizer (something like `nvdiffrast` or a custom CUDA kernel). The canvas elements are just colored rectangles and text labels — trivial to rasterize on a GPU at >100K fps. Or, for semantic-only training, skip rendering entirely and only render during eval.

2. **VLM inference and KV-cache memory.** This is the real wall. A 13B VLM with 10K concurrent rollouts means 10K active KV-caches. At ~1GB each, that blows past even 8×H100 memory without PagedAttention. The fix: use vLLM with PagedAttention, which manages KV-cache like virtual memory pages — non-contiguous physical blocks mapped to logical sequences. Reduces waste from ~60% to <4%, making 10K concurrent sequences feasible. Continuous batching means env steps don't block waiting for the slowest sequence.

3. **Python overhead.** Pydantic's `model_dump()` on 10K canvases per step creates GC pressure, and the GIL serializes everything. I'd rewrite the canvas engine in C++ with pybind11, turning element state into flat struct arrays. EnvPool provides a nice framework for this — C++-native vectorized envs with zero-copy observation transfer.

**The architecture I'd use:**

- `gymnasium.vector.AsyncVectorEnv` with ~100 workers, each managing 100 sub-environments. The canvas engine is pure Python with no shared state, so this scales linearly until you hit Python overhead.
- For the VLM, vLLM with continuous batching + PagedAttention on the inference side.
- Double-buffered async pipeline: while the VLM processes batch N, environments execute batch N+1's transitions. Hides env latency behind inference latency.
- **Gym for training, MCP for eval.** During PPO training, use the `step()` interface with vectorized envs for throughput. The MCP server (which now supports session isolation) is for interactive evaluation — connect Claude or another LLM client to a live environment for qualitative assessment. This way the MCP overhead (HTTP, JSON-RPC) never touches the training hot path.

The key insight: the environment is trivially parallelizable; the hard problem is VLM inference memory management and latency pipelining.

## Testing & Verification

I verified the implementation through three complementary methods:

- **pytest suite** (`tests/test_env.py`): 13 tests covering observation contract, semantic actions, low-level actions, reward bounds, terminal-only reward mode, episode truncation, and deterministic replay
- **Demo baselines** (`demo.py`): NOP (blank canvas, scores ~-0.8), oracle (scripted reference policy, scores ~+0.95), and random agent (15 random steps, variable score) — shows the reward function has the right shape and the right spread
- **MCP smoke test**: Verified the FastMCP server end-to-end by chaining `reset_environment` → `execute_action` → `get_current_reward` with session isolation, confirming state transitions, reward computation, and session independence all work correctly
- **Claude Desktop MCP run (with screenshots)**: Re-ran the same MCP flow from Claude Desktop using a shared `session_id` and captured screenshots of tool outputs to confirm real client interoperability, not just local function calls