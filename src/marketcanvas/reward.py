from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from marketcanvas.canvas import Canvas
from marketcanvas.contrast import wcag_score
from marketcanvas.elements import Element, ElementType
from marketcanvas.prompt_parser import ParsedPrompt
from marketcanvas.spatial import iou, centers_aligned


class RewardBreakdown(BaseModel):
    constraint: float
    layout: float
    accessibility: float
    completeness: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return self.model_dump()


W_CONSTRAINT = 0.40
W_LAYOUT = 0.25
W_ACCESSIBILITY = 0.20
W_COMPLETENESS = 0.15

MAX_ELEMENTS = 20
MIN_ELEMENT_SIZE = 20


def compute_reward(
    canvas: Canvas,
    parsed_prompt: ParsedPrompt,
    step_count: int = 0,
    max_steps: int = 50,
) -> RewardBreakdown:
    """Compute 4-component reward scaled to [-1, 1]."""
    visible = [e for e in canvas.elements if e.is_visible(canvas.width, canvas.height) and e.meets_minimum_size(MIN_ELEMENT_SIZE)]

    c = _constraint_score(visible, parsed_prompt)
    l = _layout_score(visible, canvas)
    a = _accessibility_score(visible)
    comp = _completeness_score(visible, canvas, step_count, max_steps)

    raw = W_CONSTRAINT * c + W_LAYOUT * l + W_ACCESSIBILITY * a + W_COMPLETENESS * comp
    total = raw * 2 - 1  # scale [0,1] -> [-1,1]
    return RewardBreakdown(constraint=c, layout=l, accessibility=a, completeness=comp, total=total)


def _constraint_score(elements: list[Element], prompt: ParsedPrompt) -> float:
    return prompt.satisfaction_score(elements)


def _layout_score(elements: list[Element], canvas: Canvas) -> float:
    if not elements:
        return 0.0

    scores: list[float] = []

    # Overlap penalty
    overlap_penalties: list[float] = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            pair_iou = iou(a, b)
            overlap_penalties.append(max(0.0, pair_iou - 0.2))
    overlap_score = 1.0 - min(1.0, sum(overlap_penalties) * 2) if overlap_penalties else 1.0
    scores.append(overlap_score)

    # Alignment bonus
    alignment = centers_aligned(elements)
    scores.append(alignment)

    # Bounds check
    in_bounds = sum(1 for e in elements if e.is_within_bounds(canvas.width, canvas.height))
    scores.append(in_bounds / len(elements))

    # Visual hierarchy: headline is largest text, CTA in lower half
    hierarchy = _hierarchy_score(elements, canvas)
    scores.append(hierarchy)

    return sum(scores) / len(scores)


def _hierarchy_score(elements: list[Element], canvas: Canvas) -> float:
    text_els = [e for e in elements if e.type == ElementType.TEXT]
    shape_els = [e for e in elements if e.type == ElementType.SHAPE]

    points = 0.0
    checks = 0

    if text_els:
        largest_text = max(text_els, key=lambda e: e.area)
        # Headline should be in upper half
        checks += 1
        if largest_text.y < canvas.height / 2:
            points += 1.0

    if shape_els:
        # CTA/buttons should be in lower half
        checks += 1
        buttons_lower = sum(1 for e in shape_els if e.center_y > canvas.height / 3)
        if shape_els:
            points += buttons_lower / len(shape_els)

    return points / checks if checks > 0 else 0.5


def _accessibility_score(elements: list[Element]) -> float:
    text_els = [e for e in elements if e.type == ElementType.TEXT]
    if not text_els:
        return 0.5  # neutral if no text

    scores: list[float] = []
    for el in text_els:
        # Check text color against the element background.
        # This rewards local readability where the text is actually rendered.
        large = el.height >= 24
        scores.append(wcag_score(el.text_color, el.color, large_text=large))

    return sum(scores) / len(scores)


def _completeness_score(
    elements: list[Element],
    canvas: Canvas,
    step_count: int,
    max_steps: int,
) -> float:
    if not elements:
        return 0.0

    scores: list[float] = []

    # Canvas utilization
    total_area = sum(e.area for e in elements)
    canvas_area = canvas.width * canvas.height
    utilization = min(1.0, total_area / (canvas_area * 0.3))  # 30% coverage = full score
    scores.append(utilization)

    # Clutter penalty
    n = len(elements)
    if n > MAX_ELEMENTS:
        scores.append(max(0.0, 1.0 - (n - MAX_ELEMENTS) / 10))
    elif n == 1:
        scores.append(0.3)  # single element penalty
    else:
        scores.append(1.0)

    # Effort: penalize zero actions
    if step_count == 0:
        scores.append(0.0)
    else:
        scores.append(min(1.0, step_count / 3))  # at least 3 actions for full score

    return sum(scores) / len(scores)
