from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ElementType(str, Enum):
    TEXT = "text"
    SHAPE = "shape"
    IMAGE = "image"


_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


class Element(BaseModel):
    model_config = {"frozen": False}

    id: str
    type: ElementType
    x: int
    y: int
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    color: str = "#000000"
    text_color: str = "#FFFFFF"
    content: str = ""
    z_index: int = Field(default=0, ge=0)

    @field_validator("color", "text_color")
    @classmethod
    def _validate_hex(cls, v: str) -> str:
        if not _HEX_PATTERN.match(v):
            raise ValueError(f"Invalid hex color: {v}")
        return v.upper()

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_visible(self, canvas_width: int = 800, canvas_height: int = 600) -> bool:
        if self.width < 1 or self.height < 1:
            return False
        return (
            self.x < canvas_width
            and self.y < canvas_height
            and self.right > 0
            and self.bottom > 0
        )

    def meets_minimum_size(self, min_size: int = 20) -> bool:
        return self.width >= min_size and self.height >= min_size

    def is_within_bounds(self, canvas_width: int = 800, canvas_height: int = 600) -> bool:
        return self.x >= 0 and self.y >= 0 and self.right <= canvas_width and self.bottom <= canvas_height

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
