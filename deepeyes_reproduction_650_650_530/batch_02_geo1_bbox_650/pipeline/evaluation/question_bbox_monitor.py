"""Helpers for enforcing question-provided bounding boxes during evaluation."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


QUESTION_BBOX_CATEGORIES = frozenset(
    {
        "Object properties/Object classification",
        "Object properties/Object color",
        "Object properties/Object motion state",
    }
)

_QUESTION_BBOX_RE = re.compile(r"\bBounding\s+box\s*:\s*(\[[^\]]+\])", re.IGNORECASE)


def is_question_bbox_category(category: str) -> bool:
    return category in QUESTION_BBOX_CATEGORIES


def normalize_bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox must contain four coordinates, got {value!r}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"bbox coordinates must be numeric, got {value!r}")
    normalized = [int(item) for item in value]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        raise ValueError(f"bbox must have positive width and height, got {normalized!r}")
    return normalized


def extract_question_bbox(question: str) -> list[int]:
    match = _QUESTION_BBOX_RE.search(question)
    if match is None:
        raise ValueError("question-bbox category has no 'Bounding box: [...]' annotation")
    return normalize_bbox(json.loads(match.group(1)))


def has_complete_tool_call(response: str) -> bool:
    return "<tool_call>" in response and "</tool_call>" in response


def parse_tool_call(response: str) -> dict[str, Any]:
    if not has_complete_tool_call(response):
        raise ValueError("response does not contain a complete tool call")
    payload_text = response.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = ast.literal_eval(payload_text)
    if not isinstance(payload, dict):
        raise ValueError(f"tool call must be an object, got {type(payload).__name__}")

    arguments = payload.get("arguments")
    if arguments is None and isinstance(payload.get("function"), dict):
        arguments = payload["function"].get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool call has no arguments object")
    payload["arguments"] = arguments
    return payload


def synthetic_tool_call(question_bbox: list[int]) -> str:
    payload = {
        "name": "image_zoom_in_tool",
        "arguments": {"bbox_2d": question_bbox, "image_index": 0},
    }
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
