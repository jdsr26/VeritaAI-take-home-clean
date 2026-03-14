from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marketcanvas.canvas import Canvas
from marketcanvas.elements import Element
from marketcanvas.elements import ElementType


@dataclass
class CursorState:
    """Virtual cursor state for low-level interaction mode."""

    x: int = 0
    y: int = 0
    selected_element_id: str | None = None
    drag_origin: tuple[int, int] | None = None
    text_buffer: str = ""


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _element_at(canvas: Canvas, x: int, y: int) -> Element | None:
    """Return top-most element at a point, respecting z-index ordering."""
    for el in reversed(canvas.elements):
        if el.x <= x < el.right and el.y <= y < el.bottom:
            return el
    return None


def apply_low_level_action(
    canvas: Canvas,
    cursor_state: CursorState,
    action_type: str,
    params: dict[str, Any],
) -> None:
    """Apply a low-level computer-use action in-place."""
    if action_type == "mouse_move":
        cursor_state.x = _clamp(int(params.get("x", cursor_state.x)), 0, max(0, canvas.width - 1))
        cursor_state.y = _clamp(int(params.get("y", cursor_state.y)), 0, max(0, canvas.height - 1))
        return

    if action_type == "mouse_click":
        if "x" in params:
            cursor_state.x = _clamp(int(params["x"]), 0, max(0, canvas.width - 1))
        if "y" in params:
            cursor_state.y = _clamp(int(params["y"]), 0, max(0, canvas.height - 1))

        el = _element_at(canvas, cursor_state.x, cursor_state.y)
        cursor_state.selected_element_id = el.id if el else None
        return

    if action_type == "mouse_drag":
        x1 = _clamp(int(params.get("x1", cursor_state.x)), 0, max(0, canvas.width - 1))
        y1 = _clamp(int(params.get("y1", cursor_state.y)), 0, max(0, canvas.height - 1))
        x2 = _clamp(int(params.get("x2", x1)), 0, max(0, canvas.width - 1))
        y2 = _clamp(int(params.get("y2", y1)), 0, max(0, canvas.height - 1))

        cursor_state.drag_origin = (x1, y1)
        start_el = _element_at(canvas, x1, y1)

        if start_el is not None:
            # Preserve grab offset so drags feel like pointer-driven interactions.
            offset_x = x1 - start_el.x
            offset_y = y1 - start_el.y
            new_x = _clamp(x2 - offset_x, 0, max(0, canvas.width - start_el.width))
            new_y = _clamp(y2 - offset_y, 0, max(0, canvas.height - start_el.height))
            canvas.move_element(start_el.id, new_x, new_y)
            cursor_state.selected_element_id = start_el.id
        else:
            left = min(x1, x2)
            top = min(y1, y2)
            width = max(1, abs(x2 - x1))
            height = max(1, abs(y2 - y1))
            created = canvas.add_element(
                type=ElementType.SHAPE,
                content="",
                x=left,
                y=top,
                width=width,
                height=height,
                color="#000000",
                text_color="#FFFFFF",
            )
            cursor_state.selected_element_id = created.id

        cursor_state.x = x2
        cursor_state.y = y2
        return

    if action_type == "keyboard_type":
        text = str(params.get("text", ""))
        cursor_state.text_buffer += text
        if cursor_state.selected_element_id is None:
            return

        target = canvas.get_element(cursor_state.selected_element_id)
        if target is None:
            cursor_state.selected_element_id = None
            return

        canvas.change_text(target.id, target.content + text)


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
