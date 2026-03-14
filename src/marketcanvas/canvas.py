from __future__ import annotations

from typing import Any

from marketcanvas.elements import Element, ElementType


class Canvas:
    """Core 800x600 canvas engine with element CRUD operations."""

    def __init__(self, width: int = 800, height: int = 600, background: str = "#FFFFFF") -> None:
        self.width = width
        self.height = height
        self.background = background
        self._elements: list[Element] = []
        self._next_id = 1

    @property
    def elements(self) -> list[Element]:
        return sorted(self._elements, key=lambda e: e.z_index)

    def _generate_id(self) -> str:
        eid = f"el_{self._next_id:03d}"
        self._next_id += 1
        return eid

    def add_element(
        self,
        type: ElementType,
        content: str,
        x: int,
        y: int,
        width: int,
        height: int,
        color: str = "#000000",
        text_color: str = "#FFFFFF",
        z_index: int | None = None,
    ) -> Element:
        if z_index is None:
            z_index = max((e.z_index for e in self._elements), default=0) + 1
        el = Element(
            id=self._generate_id(),
            type=type,
            x=x, y=y,
            width=width, height=height,
            color=color,
            text_color=text_color,
            content=content,
            z_index=z_index,
        )
        self._elements.append(el)
        return el

    def get_element(self, element_id: str) -> Element | None:
        for el in self._elements:
            if el.id == element_id:
                return el
        return None

    def delete_element(self, element_id: str) -> bool:
        for i, el in enumerate(self._elements):
            if el.id == element_id:
                self._elements.pop(i)
                return True
        return False

    def move_element(self, element_id: str, new_x: int, new_y: int) -> bool:
        el = self.get_element(element_id)
        if el is None:
            return False
        el.x = new_x
        el.y = new_y
        return True

    def resize_element(self, element_id: str, new_width: int, new_height: int) -> bool:
        el = self.get_element(element_id)
        if el is None:
            return False
        el.width = max(1, new_width)
        el.height = max(1, new_height)
        return True

    def change_color(self, element_id: str, hex_code: str) -> bool:
        el = self.get_element(element_id)
        if el is None:
            return False
        el.color = hex_code
        return True

    def change_text(self, element_id: str, new_content: str) -> bool:
        el = self.get_element(element_id)
        if el is None:
            return False
        el.content = new_content
        return True

    def set_z_index(self, element_id: str, new_z: int) -> bool:
        el = self.get_element(element_id)
        if el is None:
            return False
        el.z_index = new_z
        return True

    def clear(self) -> None:
        self._elements.clear()
        self._next_id = 1

    def element_count(self) -> int:
        return len(self._elements)

    def to_dict(self) -> dict[str, Any]:
        from marketcanvas.spatial import compute_spatial_relations

        return {
            "canvas": {
                "width": self.width,
                "height": self.height,
                "background": self.background,
            },
            "elements": [e.to_dict() for e in self.elements],
            "spatial_relations": compute_spatial_relations(self.elements),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Canvas":
        canvas_info = data["canvas"]
        c = cls(canvas_info["width"], canvas_info["height"], canvas_info.get("background", "#FFFFFF"))
        for ed in data["elements"]:
            ed = ed.copy()
            ed["type"] = ElementType(ed["type"])
            el = Element(**ed)
            c._elements.append(el)
            num = int(el.id.split("_")[1])
            if num >= c._next_id:
                c._next_id = num + 1
        return c
