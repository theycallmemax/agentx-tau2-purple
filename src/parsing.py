"""JSON parsing and text extraction utilities."""

from __future__ import annotations

import json
import re
from typing import Any


RESPOND_ACTION_NAME = "respond"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = strip_code_fences(text)
    tag_match = re.search(
        r"<json>\s*(.*?)\s*</json>", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    if tag_match:
        cleaned = tag_match.group(1).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, str):
            return extract_json_object(data)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    balanced = extract_balanced_json_object(cleaned)
    if not balanced:
        return None

    try:
        data = json.loads(balanced)
        if isinstance(data, str):
            return extract_json_object(data)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def normalize_respond_content(content: Any) -> str:
    if not isinstance(content, str):
        return ""

    normalized = content.strip()
    seen: set[str] = set()
    while normalized and normalized not in seen:
        seen.add(normalized)
        parsed = extract_json_object(normalized)
        if not isinstance(parsed, dict):
            break
        if parsed.get("name") != RESPOND_ACTION_NAME:
            break
        arguments = parsed.get("arguments")
        nested = arguments.get("content") if isinstance(arguments, dict) else None
        if not isinstance(nested, str):
            break
        normalized = nested.strip()
    return normalized


def iter_balanced_json_arrays(text: str):
    """Yield every top-level JSON array substring found in text."""
    i = 0
    length = len(text)
    while i < length:
        if text[i] != "[":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escape = False
        j = start
        while j < length:
            c = text[j]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        yield text[start : j + 1]
                        break
            j += 1
        i = j + 1 if j < length else length


def normalize_openai_tool(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce a dict into the OpenAI {"type":"function","function":{...}} shape."""
    if not isinstance(entry, dict):
        return None
    if entry.get("type") == "function" and isinstance(entry.get("function"), dict):
        fn = entry["function"]
        if isinstance(fn.get("name"), str):
            return entry
        return None
    if isinstance(entry.get("name"), str) and isinstance(
        entry.get("parameters"), dict
    ):
        return {
            "type": "function",
            "function": {
                "name": entry["name"],
                "description": entry.get("description", ""),
                "parameters": entry["parameters"],
            },
        }
    return None


def extract_openai_tools(text: str) -> list[dict[str, Any]] | None:
    """Find an OpenAI-style tools array embedded in the first evaluator message."""
    best: list[dict[str, Any]] = []
    for candidate in iter_balanced_json_arrays(text):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, list) or len(data) < 2:
            continue
        normalized: list[dict[str, Any]] = []
        for item in data:
            tool = normalize_openai_tool(item)
            if tool is None:
                normalized = []
                break
            normalized.append(tool)
        if normalized and len(normalized) > len(best):
            best = normalized
    return best or None
