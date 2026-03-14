from __future__ import annotations

from typing import Any

from marketcanvas.elements import Element


def intersection_area(a: Element, b: Element) -> int:
    """Compute pixel area of intersection between two element bounding boxes."""
    x_overlap = max(0, min(a.right, b.right) - max(a.x, b.x))
    y_overlap = max(0, min(a.bottom, b.bottom) - max(a.y, b.y))
    return x_overlap * y_overlap


def iou(a: Element, b: Element) -> float:
    """Intersection over Union between two elements."""
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(a: Element, b: Element) -> float:
    """Fraction of the smaller element covered by overlap."""
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0


def spatial_relation(source: Element, target: Element) -> str:
    """Determine primary spatial relation from source to target."""
    if overlap_ratio(source, target) > 0.2:
        return "overlaps"
    if source.bottom <= target.y:
        return "above"
    if source.y >= target.bottom:
        return "below"
    if source.right <= target.x:
        return "left_of"
    if source.x >= target.right:
        return "right_of"
    return "overlaps"


def compute_spatial_relations(elements: list[Element]) -> list[dict[str, str]]:
    """Compute all pairwise spatial relations."""
    relations: list[dict[str, str]] = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            relations.append({
                "type": spatial_relation(a, b),
                "source": a.id,
                "target": b.id,
            })
    return relations


def centers_aligned(elements: list[Element], tolerance: int = 10) -> float:
    """Fraction of element pairs with x-centers within tolerance."""
    if len(elements) < 2:
        return 1.0
    aligned = 0
    total = 0
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            total += 1
            if abs(a.center_x - b.center_x) <= tolerance:
                aligned += 1
    return aligned / total
