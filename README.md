# MarketCanvas-Env

A minimalist 2D design canvas RL environment with MCP server interface. Simulates a "Mini-Canva" where agents create marketing assets, scored by a 4-component reward function (constraint satisfaction, layout quality, WCAG accessibility, completeness).

## Setup

```bash
pip install -e .
```

## Quick Start

```bash
python demo.py
```

This runs three baselines against the prompt *"Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast"*:

| Baseline | Expected Reward | Description |
|----------|----------------|-------------|
| NOP | ~-0.8 | Zero actions, blank canvas |
| Oracle | ~+0.95 | Scripted optimal design |
| Random | Varies by seed; can be positive | 15 random actions |

## Demo Output

The demo script saves example renders in the project root:

### Oracle Output

![Oracle result](output_oracle.png)

### Random Output

![Random result](output_random.png)

### NOP Output

![NOP result](output_nop.png)

## MCP Server

```bash
python -m marketcanvas.mcp_server
```

Exposes tools: `reset_environment`, `get_canvas_state`, `execute_action`, `get_current_reward`, `render_canvas`.

## Gymnasium Usage

```python
from marketcanvas import MarketCanvasEnv

env = MarketCanvasEnv()
obs, info = env.reset(options={"prompt": "Create a banner with a headline and CTA button"})

obs, reward, terminated, truncated, info = env.step_semantic(
    "add_element",
    type="text", content="Summer Sale",
    x=200, y=50, width=400, height=60,
    color="#1A1A5E", text_color="#FFFFFF",
)
```

## Project Structure

```
src/marketcanvas/
├── elements.py        # Element types and data model
├── canvas.py          # Core canvas engine (CRUD)
├── contrast.py        # WCAG 2.0 contrast (from scratch)
├── spatial.py         # Spatial relationship computation
├── prompt_parser.py   # Target prompt → constraints
├── reward.py          # 4-component reward function
├── renderer.py        # PIL rendering to PNG/array
├── environment.py     # Gymnasium env wrapper
└── mcp_server.py      # FastMCP server
```
