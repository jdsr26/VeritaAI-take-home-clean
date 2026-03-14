from __future__ import annotations

from typing import Any

from marketcanvas.canvas import Canvas
from marketcanvas.elements import ElementType


def apply_semantic_action(canvas: Canvas, action_type: str, params: dict[str, Any]) -> None:
    """Apply a semantic canvas action in-place."""
    if action_type == "add_element":
        elem_type = params.get("type", "shape")
        if isinstance(elem_type, str):
            elem_type = ElementType(elem_type)
        elif isinstance(elem_type, int):
            elem_type = list(ElementType)[elem_type]

        canvas.add_element(
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
        canvas.move_element(params["id"], int(params["new_x"]), int(params["new_y"]))
    elif action_type == "resize_element":
        canvas.resize_element(params["id"], int(params["new_width"]), int(params["new_height"]))
    elif action_type == "change_color":
        canvas.change_color(params["id"], params["hex_code"])
    elif action_type == "change_text":
        canvas.change_text(params["id"], params["new_content"])
    elif action_type == "delete_element":
        canvas.delete_element(params["id"])
    elif action_type == "set_z_index":
        canvas.set_z_index(params["id"], int(params["new_z"]))
    # noop: do nothing
