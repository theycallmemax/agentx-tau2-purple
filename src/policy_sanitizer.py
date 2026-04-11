"""Sanitize evaluator policy/tool text before it reaches the model."""

from __future__ import annotations

import re


EXAMPLE_USER_IDS = {"sara_doe_496"}
EXAMPLE_RESERVATION_IDS = {"ZFA04Y", "8JX2WO"}


def sanitize_tool_descriptions(text: str) -> str:
    sanitized = text
    for user_id in EXAMPLE_USER_IDS:
        sanitized = sanitized.replace(f"'{user_id}'", "'<user_id>'")
        sanitized = sanitized.replace(f'"{user_id}"', '"<user_id>"')
    for reservation_id in EXAMPLE_RESERVATION_IDS:
        sanitized = sanitized.replace(f"'{reservation_id}'", "'<reservation_id>'")
        sanitized = sanitized.replace(f'"{reservation_id}"', '"<reservation_id>"')

    sanitized = re.sub(
        r"(such as|for example)\s+'<user_id>'",
        r"\1 '<user_id>'",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized
