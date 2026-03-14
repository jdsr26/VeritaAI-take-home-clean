from __future__ import annotations


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _linearize(channel: int) -> float:
    """Convert sRGB channel (0-255) to linear RGB value."""
    s = channel / 255.0
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG 2.0 relative luminance: L = 0.2126*R + 0.7152*G + 0.0722*B."""
    r, g, b = hex_to_rgb(hex_color)
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(color1: str, color2: str) -> float:
    """WCAG 2.0 contrast ratio between two hex colors. Returns value in [1, 21]."""
    l1 = relative_luminance(color1)
    l2 = relative_luminance(color2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def wcag_score(text_color: str, bg_color: str, large_text: bool = False) -> float:
    """Score contrast on a 0-1 scale based on WCAG AA thresholds."""
    ratio = contrast_ratio(text_color, bg_color)
    threshold = 3.0 if large_text else 4.5
    if ratio >= threshold:
        return 1.0
    if ratio < 3.0:
        return 0.0
    return (ratio - 3.0) / (threshold - 3.0)
