from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from marketcanvas.elements import Element, ElementType


class ConstraintType(str, Enum):
    ELEMENT_EXISTS = "element_exists"
    COLOR_USED = "color_used"
    GOOD_CONTRAST = "good_contrast"
    HAS_ELEMENTS = "has_elements"


class RequiredElement(BaseModel):
    type: ElementType
    keyword: str


class Constraint(BaseModel):
    description: str
    check: ConstraintType
    params: dict[str, Any] = Field(default_factory=dict)


class DesignConstraints(BaseModel):
    """Structured representation of what a target prompt requires."""
    required_elements: list[RequiredElement] = Field(default_factory=list)
    required_colors: dict[str, str] = Field(default_factory=dict)
    require_contrast: bool = False


class ParsedPrompt(BaseModel):
    raw: str
    constraints: list[Constraint] = Field(default_factory=list)
    design: DesignConstraints = Field(default_factory=DesignConstraints)

    def satisfaction_score(self, elements: list[Element]) -> float:
        if not self.constraints:
            return 0.0
        passed = sum(1 for c in self.constraints if _evaluate(c, elements))
        return passed / len(self.constraints)


_KEYWORD_MAP: dict[str, dict[str, Any]] = {
    "headline": {"type": ElementType.TEXT, "keyword": "headline"},
    "title": {"type": ElementType.TEXT, "keyword": "title"},
    "heading": {"type": ElementType.TEXT, "keyword": "heading"},
    "cta": {"type": ElementType.SHAPE, "keyword": "cta"},
    "button": {"type": ElementType.SHAPE, "keyword": "button"},
    "image": {"type": ElementType.IMAGE, "keyword": "image"},
    "logo": {"type": ElementType.IMAGE, "keyword": "logo"},
    "banner": {"type": ElementType.SHAPE, "keyword": "banner"},
    "text": {"type": ElementType.TEXT, "keyword": "text"},
    "subtitle": {"type": ElementType.TEXT, "keyword": "subtitle"},
    "description": {"type": ElementType.TEXT, "keyword": "description"},
}

_COLOR_MAP: dict[str, str] = {
    "red": "#FF0000", "blue": "#0000FF", "green": "#00FF00",
    "yellow": "#FFFF00", "orange": "#FFA500", "purple": "#800080",
    "black": "#000000", "white": "#FFFFFF", "pink": "#FFC0CB",
}


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Parse a natural-language target prompt into typed constraints."""
    lower = prompt.lower()
    design = DesignConstraints()
    constraints: list[Constraint] = []

    for keyword, info in _KEYWORD_MAP.items():
        if keyword in lower:
            req = RequiredElement(type=info["type"], keyword=keyword)
            design.required_elements.append(req)
            constraints.append(Constraint(
                description=f"Has {keyword} element",
                check=ConstraintType.ELEMENT_EXISTS,
                params={"type": info["type"], "keyword": keyword},
            ))

    for color_name, hex_val in _COLOR_MAP.items():
        if color_name in lower:
            design.required_colors[color_name] = hex_val
            constraints.append(Constraint(
                description=f"Uses {color_name} color",
                check=ConstraintType.COLOR_USED,
                params={"color": hex_val, "color_name": color_name},
            ))

    if "contrast" in lower or "accessible" in lower or "wcag" in lower:
        design.require_contrast = True
        constraints.append(Constraint(
            description="Good contrast ratio",
            check=ConstraintType.GOOD_CONTRAST,
            params={},
        ))

    if not constraints:
        constraints.append(Constraint(
            description="Has at least one element",
            check=ConstraintType.HAS_ELEMENTS,
            params={},
        ))

    return ParsedPrompt(raw=prompt, constraints=constraints, design=design)


def _evaluate(constraint: Constraint, elements: list[Element]) -> bool:
    visible = [e for e in elements if e.is_visible() and e.meets_minimum_size()]

    if constraint.check == ConstraintType.ELEMENT_EXISTS:
        req_type = constraint.params["type"]
        keyword = constraint.params["keyword"]
        return any(
            e.type == req_type or keyword in e.content.lower()
            for e in visible
        )

    if constraint.check == ConstraintType.COLOR_USED:
        target = constraint.params["color"].upper()
        return any(e.color.upper() == target for e in visible)

    if constraint.check == ConstraintType.GOOD_CONTRAST:
        from marketcanvas.contrast import contrast_ratio
        text_els = [e for e in visible if e.type == ElementType.TEXT]
        if not text_els:
            return False
        return all(contrast_ratio(e.text_color, e.color) >= 4.5 for e in text_els)

    if constraint.check == ConstraintType.HAS_ELEMENTS:
        return len(visible) > 0

    return False
