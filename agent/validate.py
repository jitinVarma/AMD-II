"""Final validation pass: guarantees every requested style is present,
non-empty, and reasonably sized before a result is ever written out. This is
the last line of defense against a zero score.
"""
from __future__ import annotations

import logging

from .styling import fallback_caption

logger = logging.getLogger("agent.validate")

_MAX_CAPTION_CHARS = 400


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in (". ", "! ", "? "):
        idx = truncated.rfind(sep)
        if idx > max_chars * 0.4:
            return truncated[: idx + 1].strip()
    return truncated.rstrip() + "..."


def validate_and_fix(captions: dict, styles: list[str], description: dict, task_id: str = "") -> dict:
    """Returns a dict with exactly `styles` as keys, each a non-empty,
    length-bounded string. Never raises.
    """
    fixed: dict[str, str] = {}
    for style in styles:
        value = captions.get(style)
        if isinstance(value, str) and value.strip():
            fixed[style] = _truncate_at_sentence(value.strip(), _MAX_CAPTION_CHARS)
        else:
            logger.warning("[%s] style '%s' missing/empty at final validation, synthesizing fallback", task_id, style)
            fixed[style] = fallback_caption(style, description)
    return fixed
