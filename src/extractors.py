"""Entity extraction from text using regex patterns."""

from __future__ import annotations

import re
from typing import Any


USER_ID_PATTERN = re.compile(
    r"\b(?!credit_card_)(?!gift_card_)(?!certificate_)[a-z]+_[a-z]+_\d{3,}\b"
)
RESERVATION_ID_PATTERN = re.compile(r"\b(?!HAT\d)(?=[A-Z0-9]{6}\b)(?=.*\d)[A-Z0-9]{6}\b")
FLIGHT_NUMBER_PATTERN = re.compile(r"\bHAT\d{3}\b")
AIRPORT_CODE_PATTERN = re.compile(r"\b[A-Z]{3}\b")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Keep ambiguous metro names like "New York" out of this mapping so the guard can
# still ask the user to disambiguate the airport when needed.
CITY_AIRPORT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\blaguardia\b", re.IGNORECASE), "LGA"),
    (re.compile(r"\bphiladelphia\b", re.IGNORECASE), "PHL"),
    (re.compile(r"\bnewark\b", re.IGNORECASE), "EWR"),
    (re.compile(r"\blos angeles\b", re.IGNORECASE), "LAX"),
    (re.compile(r"\bsan francisco\b", re.IGNORECASE), "SFO"),
    (re.compile(r"\bmiami\b", re.IGNORECASE), "MIA"),
    (re.compile(r"\bdallas\b", re.IGNORECASE), "DFW"),
    (re.compile(r"\batlanta\b", re.IGNORECASE), "ATL"),
    (re.compile(r"\bseattle\b", re.IGNORECASE), "SEA"),
    (re.compile(r"\bboston\b", re.IGNORECASE), "BOS"),
    (re.compile(r"\bdenver\b", re.IGNORECASE), "DEN"),
    (re.compile(r"\bhouston\b", re.IGNORECASE), "IAH"),
    (re.compile(r"\bwashington(?:\s*,?\s*dc)?\b", re.IGNORECASE), "DCA"),
    (re.compile(r"\bphoenix\b", re.IGNORECASE), "PHX"),
    (re.compile(r"\bminneapolis\b", re.IGNORECASE), "MSP"),
    (re.compile(r"\bdetroit\b", re.IGNORECASE), "DTW"),
    (re.compile(r"\borlando\b", re.IGNORECASE), "MCO"),
    (re.compile(r"\bportland\b", re.IGNORECASE), "PDX"),
    (re.compile(r"\blas vegas\b", re.IGNORECASE), "LAS"),
    (re.compile(r"\bsalt lake city\b", re.IGNORECASE), "SLC"),
    (re.compile(r"\btampa\b", re.IGNORECASE), "TPA"),
]


def extract_user_id(text: str) -> str | None:
    match = USER_ID_PATTERN.search(text)
    return match.group(0) if match else None


def extract_reservation_id(text: str) -> str | None:
    match = RESERVATION_ID_PATTERN.search(text)
    return match.group(0) if match else None


def extract_flight_number(text: str) -> str | None:
    match = FLIGHT_NUMBER_PATTERN.search(text)
    return match.group(0) if match else None


def extract_dates(text: str) -> list[str]:
    return DATE_PATTERN.findall(text)


def extract_airport_codes(text: str) -> list[str]:
    matches: list[tuple[int, str]] = []

    for match in AIRPORT_CODE_PATTERN.finditer(text):
        matches.append((match.start(), match.group(0)))

    for pattern, airport_code in CITY_AIRPORT_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), airport_code))

    matches.sort(key=lambda item: item[0])
    ordered_codes = [code for _, code in matches]
    return dedupe_keep_order(ordered_codes)


def extract_requested_cabin(text: str) -> str | None:
    lowered = text.lower()
    if "basic economy" in lowered:
        return "basic_economy"
    if "business" in lowered:
        return "business"
    if "economy" in lowered:
        return "economy"
    return None


def extract_cancel_reason(text: str) -> str | None:
    lowered = text.lower()
    if (
        "change of plan" in lowered
        or "change of plans" in lowered
        or "no longer need" in lowered
        or "don't need to make the trip" in lowered
        or "do not need to make the trip" in lowered
        or "don't need the trip" in lowered
        or "do not need the trip" in lowered
        or "don't need to travel" in lowered
        or "do not need to travel" in lowered
    ):
        return "change_of_plan"
    if "airline cancelled" in lowered or "airline canceled" in lowered:
        return "airline_cancelled"
    if "cancelled flight" in lowered or "canceled flight" in lowered:
        return "airline_cancelled"
    if "sick" in lowered or "ill" in lowered or "medical" in lowered:
        return "health"
    if "weather" in lowered or "storm" in lowered:
        return "weather"
    if "other" in lowered:
        return "other"
    return None


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_entities_from_history(messages: list[dict[str, Any]]) -> dict[str, set[str]]:
    user_ids: set[str] = set()
    reservation_ids: set[str] = set()
    flight_numbers: set[str] = set()

    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith("tool:"):
            continue

        marker = "User message:"
        text = content.rsplit(marker, 1)[1].strip() if marker in content else content

        user_ids.update(USER_ID_PATTERN.findall(text))
        reservation_ids.update(RESERVATION_ID_PATTERN.findall(text))
        flight_numbers.update(FLIGHT_NUMBER_PATTERN.findall(text))

    return {
        "user_ids": user_ids,
        "reservation_ids": reservation_ids,
        "flight_numbers": flight_numbers,
    }
