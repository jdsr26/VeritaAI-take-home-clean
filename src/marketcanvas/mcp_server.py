from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from marketcanvas.canvas import Canvas
from marketcanvas.elements import ElementType
from marketcanvas.prompt_parser import parse_prompt, ParsedPrompt
from marketcanvas.renderer import render_to_base64
from marketcanvas.reward import compute_reward

mcp = FastMCP("MarketCanvas")

_canvas = Canvas()
_parsed_prompt = ParsedPrompt(raw="")
_step_count = 0
_max_steps = 50


@mcp.tool
def reset_environment(prompt: str) -> dict[str, Any]:
    """Reset canvas and set a new target prompt."""
    global _canvas, _parsed_prompt, _step_count
    _canvas = Canvas()
    _parsed_prompt = parse_prompt(prompt)
    _step_count = 0
    return {"state": _canvas.to_dict(), "prompt": prompt}


@mcp.tool
def get_canvas_state() -> dict[str, Any]:
    """Returns semantic JSON state of the canvas."""
    return _canvas.to_dict()


@mcp.tool
def execute_action(action_type: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute an action on the canvas. Returns new state, reward, and done flag."""
    global _step_count
    _step_count += 1

    if action_type == "add_element":
        elem_type = params.get("type", "shape")
        if isinstance(elem_type, str):
            elem_type = ElementType(elem_type)
        _canvas.add_element(
            type=elem_type,
            content=params.get("content", ""),
            x=int(params.get("x", 0)),
            y=int(params.get("y", 0)),
            width=int(params.get("width", 100)),
            height=int(params.get("height", 50)),
            color=params.get("color", "#000000"),
            text_color=params.get("text_color", "#FFFFFF"),
        )
    elif action_type == "move_element":
        _canvas.move_element(params["id"], int(params["new_x"]), int(params["new_y"]))
    elif action_type == "resize_element":
        _canvas.resize_element(params["id"], int(params["new_width"]), int(params["new_height"]))
    elif action_type == "change_color":
        _canvas.change_color(params["id"], params["hex_code"])
    elif action_type == "change_text":
        _canvas.change_text(params["id"], params["new_content"])
    elif action_type == "delete_element":
        _canvas.delete_element(params["id"])
    elif action_type == "set_z_index":
        _canvas.set_z_index(params["id"], int(params["new_z"]))

    reward = compute_reward(_canvas, _parsed_prompt, _step_count, _max_steps)
    done = _step_count >= _max_steps

    return {
        "state": _canvas.to_dict(),
        "reward": reward.to_dict(),
        "done": done,
        "step": _step_count,
    }


@mcp.tool
def get_current_reward() -> dict[str, Any]:
    """Calculate and return current reward without stepping."""
    reward = compute_reward(_canvas, _parsed_prompt, _step_count, _max_steps)
    return reward.to_dict()


@mcp.tool
def render_canvas() -> str:
    """Returns base64-encoded PNG of current canvas."""
    return render_to_base64(_canvas)


if __name__ == "__main__":
    mcp.run()
