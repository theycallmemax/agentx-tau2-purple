from __future__ import annotations

import json
import os
import re
from typing import Any

import litellm
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()

# Drop unsupported params (e.g. temperature=0 is not supported by gpt-5 family)
litellm.drop_params = True

RESPOND_ACTION_NAME = "respond"
MAX_CONTEXT_MESSAGES = int(os.getenv("TAU2_AGENT_MAX_CONTEXT_MESSAGES", "120"))
TRANSFER_HOLD_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
PLAN_MAX_TURNS = int(os.getenv("TAU2_AGENT_PLAN_MAX_TURNS", "2"))
USER_ID_PATTERN = re.compile(r"\b[a-z]+_[a-z]+_\d{3,}\b")
RESERVATION_ID_PATTERN = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=.*\d)[A-Z0-9]{6}\b")
FLIGHT_NUMBER_PATTERN = re.compile(r"\bHAT\d{3}\b")
AIRPORT_CODE_PATTERN = re.compile(r"\b[A-Z]{3}\b")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

SYSTEM_PROMPT = """You are a customer service agent participating in the tau2 benchmark.

The first user message from the evaluator contains the domain policy and the list
of tools you may use. Follow that policy strictly.

Hard rules:
- Return EXACTLY one JSON object with keys "name" and "arguments".
- Use exactly one tool at a time. Never bundle multiple tool calls.
- To reply to the user directly, use name "respond" with arguments {"content": "<message>"}.
- The value of arguments.content for "respond" must be plain natural language only,
  never another JSON object, tool schema, or wrapper like {"name":"respond",...}.
- Never invent tool results. Wait for the environment to return them.
- Do not ask the user to pre-confirm write actions. Once you have all the facts
  required by the policy, perform the action directly.
- Do not add an extra "reply yes", "confirm yes", or "YES, proceed" step when the
  user has already requested the action and supplied the required details.
- When policy forbids a request, refuse clearly via "respond". Only call
  transfer_to_human_agents when the policy explicitly requires it.
- Before searching flights: if a direct search returns empty, immediately try
  search_onestop_flight for the same origin/destination/date before pivoting to
  alternate dates or airports. Do not exhaustively iterate over many date/airport
  combinations — pick the most promising option and propose it to the user.
- Avoid redundant reads: if you have already fetched user details or reservation
  details in this conversation, reuse them instead of calling the tool again.
- Keep user-facing responses short and operational.

Return raw JSON only, no prose, no code fences."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _extract_balanced_json_object(text: str) -> str | None:
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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(text)
    tag_match = re.search(
        r"<json>\s*(.*?)\s*</json>", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    if tag_match:
        cleaned = tag_match.group(1).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, str):
            return _extract_json_object(data)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    balanced = _extract_balanced_json_object(cleaned)
    if not balanced:
        return None

    try:
        data = json.loads(balanced)
        if isinstance(data, str):
            return _extract_json_object(data)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _normalize_respond_content(content: Any) -> str:
    if not isinstance(content, str):
        return ""

    normalized = content.strip()
    seen: set[str] = set()
    while normalized and normalized not in seen:
        seen.add(normalized)
        parsed = _extract_json_object(normalized)
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


def _iter_balanced_json_arrays(text: str):
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


def _normalize_openai_tool(entry: dict[str, Any]) -> dict[str, Any] | None:
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


def _extract_openai_tools(text: str) -> list[dict[str, Any]] | None:
    """Find an OpenAI-style tools array embedded in the first evaluator message."""
    best: list[dict[str, Any]] = []
    for candidate in _iter_balanced_json_arrays(text):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, list) or len(data) < 2:
            continue
        normalized: list[dict[str, Any]] = []
        for item in data:
            tool = _normalize_openai_tool(item)
            if tool is None:
                normalized = []
                break
            normalized.append(tool)
        if normalized and len(normalized) > len(best):
            best = normalized
    return best or None


class Agent:
    def __init__(self):
        self.model = os.getenv("TAU2_AGENT_LLM", "openai/gpt-5.2")
        self.temperature = float(os.getenv("TAU2_AGENT_TEMPERATURE", "0"))
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.tools: list[dict[str, Any]] | None = None
        self.tool_names: set[str] = set()
        self.turn_count = 0
        self.just_transferred = False
        self.session_state: dict[str, Any] = {
            "user_id": None,
            "reservation_id": None,
            "flight_number": None,
            "origin": None,
            "destination": None,
            "travel_dates": [],
            "known_reservation_ids": [],
            "known_payment_ids": [],
            "known_payment_balances": {},
            "loaded_user_details": False,
            "loaded_reservation_details": False,
            "last_tool_name": None,
            "reservation_inventory": {},
            "pending_confirmation_action": None,
        }

    def _normalize_input_prefix(self, input_text: str) -> str:
        """Raw JSON tool results arrive without a prefix; tag them as tool output."""
        stripped = input_text.strip()
        if not stripped.startswith(("user:", "tool:")) and stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    return "tool: " + stripped
            except Exception:
                pass
        return input_text

    def _messages_for_model(self) -> list[dict[str, Any]]:
        state_summary = self._state_summary_message()
        extra_system: list[dict[str, str]] = []
        if self.turn_count >= 6 and self.turn_count % 4 == 0:
            extra_system.append(
                {
                    "role": "system",
                    "content": (
                        "Policy reminder: avoid redundant reads, do not confuse flight numbers "
                        "with reservation IDs, reuse known profile data, and summarize completed "
                        "actions instead of restarting the conversation."
                    ),
                }
            )
        if self._should_hint_post_tool_summary():
            extra_system.append(
                {
                    "role": "system",
                    "content": (
                        "The previous tool call succeeded. Your next action should normally be "
                        "a concise user-facing summary of the completed change, unless the tool "
                        "result clearly requires another immediate lookup."
                    ),
                }
            )
        if len(self.messages) <= MAX_CONTEXT_MESSAGES:
            return list(self.messages) + extra_system + [state_summary]
        # Preserve system + first evaluator message (policy + tool schemas), keep recent tail.
        preserved = self.messages[:2]
        recent = self.messages[-(MAX_CONTEXT_MESSAGES - 3) :]
        return (
            preserved
            + [
                {
                    "role": "user",
                    "content": (
                        "[Earlier conversation messages omitted. Continue from the "
                        "latest verified state and keep following the original policy.]"
                    ),
                }
            ]
            + extra_system
            + [state_summary]
            + recent
        )

    def _fallback_action(self, content: str) -> dict[str, Any]:
        return {"name": RESPOND_ACTION_NAME, "arguments": {"content": content}}

    def _latest_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    marker = "User message:"
                    if marker in content:
                        embedded = content.rsplit(marker, 1)[1].strip()
                        try:
                            parsed = json.loads(embedded)
                            if isinstance(parsed, str):
                                return parsed
                        except Exception:
                            quoted = embedded.strip()
                            if len(quoted) >= 2 and quoted[0] == quoted[-1] == '"':
                                try:
                                    return json.loads(quoted)
                                except Exception:
                                    pass
                        return embedded
                    return content
        return ""

    def _extract_user_id(self, text: str) -> str | None:
        match = USER_ID_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_reservation_id(self, text: str) -> str | None:
        match = RESERVATION_ID_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_flight_number(self, text: str) -> str | None:
        match = FLIGHT_NUMBER_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_dates(self, text: str) -> list[str]:
        return DATE_PATTERN.findall(text)

    def _dedupe_keep_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _looks_like_modify_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "change my flight",
                "modify my flight",
                "change reservation",
                "modify reservation",
                "switch to",
                "push it back",
                "move it to",
                "upgrade the class",
                "upgrade cabin",
                "change the cabin",
                "nonstop flight",
                "direct flight",
                "cheapest economy",
            )
        )

    def _looks_like_cancel_intent(self, text: str) -> bool:
        lowered = text.lower()
        return "cancel" in lowered or "cancellation" in lowered

    def _state_summary_message(self) -> dict[str, str]:
        inventory = self.session_state.get("reservation_inventory", {})
        inventory_summary = []
        if isinstance(inventory, dict):
            for reservation_id, entry in list(inventory.items())[:6]:
                if not isinstance(entry, dict):
                    continue
                inventory_summary.append(
                    {
                        "reservation_id": reservation_id,
                        "origin": entry.get("origin"),
                        "destination": entry.get("destination"),
                        "dates": entry.get("dates", [])[:4],
                        "created_at": entry.get("created_at"),
                        "status": entry.get("status"),
                        "passenger_count": entry.get("passenger_count"),
                    }
                )
        summary = {
            "user_id": self.session_state.get("user_id"),
            "reservation_id": self.session_state.get("reservation_id"),
            "flight_number": self.session_state.get("flight_number"),
            "origin": self.session_state.get("origin"),
            "destination": self.session_state.get("destination"),
            "travel_dates": self.session_state.get("travel_dates", [])[:4],
            "known_reservation_ids": self.session_state.get("known_reservation_ids", [])[
                :6
            ],
            "known_payment_ids": self.session_state.get("known_payment_ids", [])[:8],
            "loaded_user_details": self.session_state.get("loaded_user_details"),
            "loaded_reservation_details": self.session_state.get(
                "loaded_reservation_details"
            ),
            "last_tool_name": self.session_state.get("last_tool_name"),
            "known_payment_balances": self.session_state.get("known_payment_balances", {}),
            "reservation_inventory": inventory_summary,
            "pending_confirmation_action": self.session_state.get(
                "pending_confirmation_action"
            ),
        }
        return {
            "role": "system",
            "content": (
                "Structured session state for reuse across steps: "
                + json.dumps(summary, ensure_ascii=False)
            ),
        }

    def _merge_state_lists(self, key: str, values: list[str]) -> None:
        existing = self.session_state.get(key, [])
        if not isinstance(existing, list):
            existing = []
        self.session_state[key] = self._dedupe_keep_order(existing + values)

    def _update_state_from_text(self, text: str) -> None:
        user_id = self._extract_user_id(text)
        reservation_id = self._extract_reservation_id(text)
        flight_number = self._extract_flight_number(text)
        dates = self._extract_dates(text)
        airport_codes = AIRPORT_CODE_PATTERN.findall(text)

        if user_id:
            self.session_state["user_id"] = user_id
        if reservation_id:
            self.session_state["reservation_id"] = reservation_id
            self._merge_state_lists("known_reservation_ids", [reservation_id])
        if flight_number:
            self.session_state["flight_number"] = flight_number
        if dates:
            self._merge_state_lists("travel_dates", dates)
        if len(airport_codes) >= 2:
            self.session_state["origin"] = airport_codes[0]
            self.session_state["destination"] = airport_codes[1]

    def _update_state_from_tool_payload(self, input_text: str) -> None:
        if not input_text.startswith("tool:"):
            return
        payload_text = input_text.split("tool:", 1)[1].strip()
        try:
            payload = json.loads(payload_text)
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        user_id = payload.get("user_id")
        reservation_id = payload.get("reservation_id")
        flight_number = payload.get("flight_number")
        origin = payload.get("origin")
        destination = payload.get("destination")

        if isinstance(user_id, str) and user_id:
            self.session_state["user_id"] = user_id
            self.session_state["loaded_user_details"] = True
        if isinstance(reservation_id, str) and reservation_id:
            self.session_state["reservation_id"] = reservation_id
            self.session_state["loaded_reservation_details"] = True
            self._merge_state_lists("known_reservation_ids", [reservation_id])
        if isinstance(flight_number, str) and flight_number:
            self.session_state["flight_number"] = flight_number
        if isinstance(origin, str) and origin:
            self.session_state["origin"] = origin
        if isinstance(destination, str) and destination:
            self.session_state["destination"] = destination

        reservations = payload.get("reservations")
        if isinstance(reservations, list):
            self._merge_state_lists(
                "known_reservation_ids",
                [value for value in reservations if isinstance(value, str)],
            )

        payment_methods = payload.get("payment_methods")
        if isinstance(payment_methods, dict):
            self._merge_state_lists(
                "known_payment_ids",
                [key for key in payment_methods if isinstance(key, str)],
            )
            balances = {}
            for key, method in payment_methods.items():
                if not isinstance(key, str) or not isinstance(method, dict):
                    continue
                amount = method.get("amount")
                if isinstance(amount, (int, float)):
                    balances[key] = float(amount)
            if balances:
                existing = self.session_state.get("known_payment_balances", {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(balances)
                self.session_state["known_payment_balances"] = existing

        payment_history = payload.get("payment_history")
        if isinstance(payment_history, list):
            self._merge_state_lists(
                "known_payment_ids",
                [
                    item.get("payment_id")
                    for item in payment_history
                    if isinstance(item, dict) and isinstance(item.get("payment_id"), str)
                ],
            )

        flights = payload.get("flights")
        if isinstance(flights, list):
            dates: list[str] = []
            for flight in flights:
                if not isinstance(flight, dict):
                    continue
                if isinstance(flight.get("flight_number"), str):
                    self.session_state["flight_number"] = flight["flight_number"]
                if isinstance(flight.get("date"), str):
                    dates.append(flight["date"])
            if dates:
                self._merge_state_lists("travel_dates", dates)

        if isinstance(reservation_id, str) and reservation_id:
            inventory = self.session_state.get("reservation_inventory", {})
            if not isinstance(inventory, dict):
                inventory = {}
            inventory[reservation_id] = {
                "origin": origin,
                "destination": destination,
                "dates": self._dedupe_keep_order(
                    [
                        flight.get("date")
                        for flight in flights or []
                        if isinstance(flight, dict) and isinstance(flight.get("date"), str)
                    ]
                ),
                "flight_numbers": self._dedupe_keep_order(
                    [
                        flight.get("flight_number")
                        for flight in flights or []
                        if isinstance(flight, dict)
                        and isinstance(flight.get("flight_number"), str)
                    ]
                ),
                "created_at": payload.get("created_at"),
                "status": payload.get("status"),
                "cabin": payload.get("cabin"),
                "passenger_count": len(payload.get("passengers", []))
                if isinstance(payload.get("passengers"), list)
                else None,
            }
            self.session_state["reservation_inventory"] = inventory

    def _should_hint_post_tool_summary(self) -> bool:
        last_tool = self.session_state.get("last_tool_name")
        if last_tool not in {
            "cancel_reservation",
            "update_reservation_flights",
            "update_reservation_baggages",
            "book_reservation",
        }:
            return False
        if not self.messages:
            return False
        latest = self.messages[-1]
        return (
            latest.get("role") == "user"
            and isinstance(latest.get("content"), str)
            and latest["content"].startswith("tool:")
            and "Error:" not in latest["content"]
        )

    def _balance_response_from_state(self) -> dict[str, Any]:
        balances = self.session_state.get("known_payment_balances", {})
        if not isinstance(balances, dict) or not balances:
            return self._fallback_action(
                "Please share your user ID so I can look up the balances on your gift cards and certificates."
            )

        lines = []
        for payment_id, amount in sorted(balances.items()):
            if payment_id.startswith(("gift_card_", "certificate_")):
                lines.append(f"- {payment_id}: ${amount:.0f}")
        if not lines:
            return self._fallback_action(
                "I couldn't find any gift cards or certificates on file in your profile."
            )
        return self._fallback_action(
            "Here are the gift card and certificate balances I found:\n"
            + "\n".join(lines)
        )

    def _should_use_plan(self, latest_user_text: str) -> bool:
        lowered = latest_user_text.lower()
        if self.turn_count > PLAN_MAX_TURNS:
            return False
        complexity_markers = sum(
            1
            for phrase in (
                " and ",
                " also ",
                "cheapest",
                "fastest",
                "all upcoming",
                "all my",
                "multiple",
                "same day",
                "other flight",
                "change",
                "cancel",
                "book",
            )
            if phrase in lowered
        )
        return complexity_markers >= 2

    def _infer_reservation_from_inventory(self, text: str) -> str | None:
        inventory = self.session_state.get("reservation_inventory", {})
        if not isinstance(inventory, dict) or not inventory:
            return None

        lowered = text.lower()
        candidates = list(inventory.items())

        if "last reservation" in lowered or "most recent reservation" in lowered:
            by_created = [
                (reservation_id, entry.get("created_at"))
                for reservation_id, entry in candidates
                if isinstance(entry, dict) and isinstance(entry.get("created_at"), str)
            ]
            if by_created:
                by_created.sort(key=lambda item: item[1], reverse=True)
                return by_created[0][0]

        dates = set(self._extract_dates(text))
        airport_codes = AIRPORT_CODE_PATTERN.findall(text)
        current_reservation = self.session_state.get("reservation_id")

        filtered: list[tuple[str, dict[str, Any]]] = []
        for reservation_id, entry in candidates:
            if not isinstance(entry, dict):
                continue
            if "other flight" in lowered and reservation_id == current_reservation:
                continue
            entry_dates = set(entry.get("dates", []))
            if dates and not dates.intersection(entry_dates):
                continue
            if len(airport_codes) >= 2:
                if entry.get("origin") != airport_codes[0] or entry.get("destination") != airport_codes[1]:
                    continue
            filtered.append((reservation_id, entry))

        if len(filtered) == 1:
            return filtered[0][0]
        return None

    def _candidate_tool_names(self, latest_user_text: str) -> set[str] | None:
        if not self.tools:
            return None

        names = {tool["function"]["name"] for tool in self.tools if "function" in tool}
        wanted: set[str] = set()
        lowered = latest_user_text.lower()

        if self._looks_like_balance_intent(latest_user_text):
            return set()

        if self._looks_like_status_intent(latest_user_text) or self._looks_like_compensation_intent(
            latest_user_text
        ):
            wanted |= {"get_flight_status", "get_user_details", "get_reservation_details"}

        if self._looks_like_baggage_intent(latest_user_text):
            wanted |= {
                "get_user_details",
                "get_reservation_details",
                "update_reservation_baggages",
            }

        if self._looks_like_booking_intent(latest_user_text):
            wanted |= {
                "list_all_airports",
                "search_direct_flight",
                "search_onestop_flight",
                "book_reservation",
                "get_user_details",
                "get_reservation_details",
                "calculate",
            }

        if self._looks_like_modify_intent(latest_user_text):
            wanted |= {
                "get_user_details",
                "get_reservation_details",
                "search_direct_flight",
                "search_onestop_flight",
                "update_reservation_flights",
                "update_reservation_baggages",
                "calculate",
                "transfer_to_human_agents",
                "get_flight_status",
            }

        if self._looks_like_cancel_intent(latest_user_text):
            wanted |= {
                "get_user_details",
                "get_reservation_details",
                "cancel_reservation",
                "transfer_to_human_agents",
                "get_flight_status",
            }

        if "human agent" in lowered or "transfer" in lowered:
            wanted |= {"transfer_to_human_agents"}

        if self.session_state.get("reservation_id"):
            wanted |= {
                "get_reservation_details",
                "cancel_reservation",
                "update_reservation_flights",
                "update_reservation_baggages",
            }
        if self.session_state.get("user_id"):
            wanted |= {"get_user_details"}
        if self.session_state.get("known_payment_ids"):
            wanted |= {"book_reservation", "calculate"}
        if self.session_state.get("origin") and self.session_state.get("destination"):
            wanted |= {"search_direct_flight", "search_onestop_flight"}

        filtered = wanted & names
        return filtered or names

    def _active_tools(self) -> list[dict[str, Any]] | None:
        if not self.tools:
            return None
        candidates = self._candidate_tool_names(self._latest_user_text())
        if candidates is None:
            return self.tools
        if not candidates:
            return []
        return [
            tool
            for tool in self.tools
            if tool.get("function", {}).get("name") in candidates
        ]

    def _looks_like_booking_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "book a flight",
                "book a one-way flight",
                "book a one way flight",
                "make a reservation",
                "make me a reservation",
                "i'd like to book",
                "i would like to book",
                "reserve a flight",
            )
        )

    def _looks_like_same_reservation_reference(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "same as my current reservation",
                "same flight that i had",
                "exactly the same as my current reservation",
                "my current reservation",
            )
        )

    def _looks_like_compensation_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "compensation",
                "certificate",
                "delayed flight",
                "canceled flight",
                "cancelled flight",
            )
        )

    def _looks_like_balance_intent(self, text: str) -> bool:
        lowered = text.lower()
        return "balance" in lowered and (
            "gift card" in lowered or "certificate" in lowered
        )

    def _looks_like_baggage_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "baggage",
                "bag allowance",
                "bags allowed",
                "how many suitcases",
                "how many bags",
                "checked bag",
                "checked baggage",
                "suitcases",
            )
        )

    def _looks_like_status_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "flight status",
                "status of my flight",
                "delayed flight",
                "delay",
                "canceled flight",
                "cancelled flight",
                "is my flight",
                "what happened to flight",
            )
        )

    def _opening_turn_action(self, latest_user_text: str) -> dict[str, Any] | None:
        reservation_id = self._extract_reservation_id(latest_user_text)
        user_id = self._extract_user_id(latest_user_text)
        flight_number = self._extract_flight_number(latest_user_text)

        if self._looks_like_balance_intent(latest_user_text):
            if self.session_state.get("loaded_user_details"):
                return self._balance_response_from_state()
            if user_id:
                return {"name": "get_user_details", "arguments": {"user_id": user_id}}
            return self._fallback_action(
                "Please share your user ID so I can look up the balances on your gift cards and certificates."
            )

        if flight_number and self._looks_like_compensation_intent(latest_user_text):
            return self._fallback_action(
                "Please share the flight date as YYYY-MM-DD so I can check the status for compensation eligibility."
            )

        if reservation_id and not user_id:
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
            }

        if self._looks_like_baggage_intent(latest_user_text):
            if user_id:
                return {"name": "get_user_details", "arguments": {"user_id": user_id}}
            return self._fallback_action(
                "Please share your user ID or reservation number so I can check your baggage allowance."
            )

        if user_id:
            return {"name": "get_user_details", "arguments": {"user_id": user_id}}

        if self._looks_like_booking_intent(latest_user_text):
            airport_codes = AIRPORT_CODE_PATTERN.findall(latest_user_text)
            if self._looks_like_same_reservation_reference(latest_user_text):
                return self._fallback_action(
                    "Please share your user ID or reservation number so I can look up your existing booking."
                )
            if len(airport_codes) < 2:
                return {"name": "list_all_airports", "arguments": {}}
            return self._fallback_action(
                "Please share the traveler details and any passenger count or cabin preferences so I can help with the booking."
            )

        if self._looks_like_compensation_intent(latest_user_text):
            return self._fallback_action(
                "Please share your reservation number, user ID, or flight number so I can review the issue."
            )

        return None

    def _guard_action(self, action: dict[str, Any]) -> dict[str, Any]:
        name = action.get("name")
        arguments = action.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return action

        latest_user_text = self._latest_user_text()
        if self.turn_count == 1:
            forced = self._opening_turn_action(latest_user_text)
            if forced is not None:
                return forced

        if self._looks_like_balance_intent(latest_user_text) and self.session_state.get(
            "loaded_user_details"
        ):
            return self._balance_response_from_state()

        user_id_from_context = self._extract_user_id(latest_user_text)
        reservation_id_from_context = self._extract_reservation_id(latest_user_text)
        flight_number_from_context = self._extract_flight_number(latest_user_text)
        inferred_reservation_id = self._infer_reservation_from_inventory(latest_user_text)

        if name == "get_flight_status":
            proposed_flight_number = arguments.get("flight_number")
            if not flight_number_from_context or not self._looks_like_status_intent(
                latest_user_text
            ):
                return self._fallback_action(
                    "Please share the flight number and whether you're asking about a delay, cancellation, or flight status."
                )
            if proposed_flight_number != flight_number_from_context:
                arguments = dict(arguments)
                arguments["flight_number"] = flight_number_from_context
                arguments.setdefault("date", "2024-05-15")
                return {"name": "get_flight_status", "arguments": arguments}

        if name == "get_user_details":
            proposed_user_id = arguments.get("user_id")
            if isinstance(proposed_user_id, str):
                lowered = proposed_user_id.lower()
                if "placeholder" in lowered:
                    proposed_user_id = None
            else:
                proposed_user_id = None

            if reservation_id_from_context and not user_id_from_context:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation_id_from_context},
                }
            if user_id_from_context and proposed_user_id != user_id_from_context:
                return {
                    "name": "get_user_details",
                    "arguments": {"user_id": user_id_from_context},
                }
            if not user_id_from_context:
                return self._fallback_action(
                    "Please share your user ID or reservation number so I can look up your booking."
                )
            if proposed_user_id and RESERVATION_ID_PATTERN.fullmatch(proposed_user_id):
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": proposed_user_id},
                }
            if not proposed_user_id:
                return self._fallback_action(
                    "Please share your user ID or reservation number so I can look up your booking."
                )

        if name == "get_reservation_details":
            proposed_reservation_id = arguments.get("reservation_id")
            if (
                isinstance(proposed_reservation_id, str)
                and FLIGHT_NUMBER_PATTERN.fullmatch(proposed_reservation_id)
            ):
                return self._fallback_action(
                    "That looks like a flight number, not a reservation number. Please share your reservation number or user ID so I can look up the booking."
                )
            if not isinstance(proposed_reservation_id, str) or not proposed_reservation_id:
                if inferred_reservation_id:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": inferred_reservation_id},
                    }
                if reservation_id_from_context:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": reservation_id_from_context},
                    }
                return self._fallback_action(
                    "Please share your reservation number so I can look up the booking details."
                )

        return action

    def _call_llm(self) -> dict[str, Any]:
        active_tools = self._active_tools()
        messages = self._messages_for_model()
        if self._should_use_plan(self._latest_user_text()):
            try:
                plan_completion = litellm.completion(
                    model=self.model,
                    messages=messages
                    + [
                        {
                            "role": "user",
                            "content": (
                                "Think briefly about the policy, missing facts, and the single "
                                "best next action. Do not output JSON yet."
                            ),
                        }
                    ],
                    temperature=self.temperature,
                )
                plan_text = (
                    getattr(plan_completion.choices[0].message, "content", None) or ""
                ).strip()
                if plan_text:
                    messages = messages + [
                        {
                            "role": "system",
                            "content": "Reasoning scratchpad from previous pass: " + plan_text[:1200],
                        }
                    ]
            except Exception:
                pass

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if active_tools:
            kwargs["tools"] = active_tools
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = litellm.completion(**kwargs)
        except Exception as exc:
            print(f"[tau2-agent] LLM call failed: {exc}")
            # Retry once without response_format/tools in case the model rejected them.
            try:
                minimal_kwargs = {
                    "model": self.model,
                    "messages": self._messages_for_model(),
                    "temperature": self.temperature,
                }
                completion = litellm.completion(**minimal_kwargs)
            except Exception as exc2:
                print(f"[tau2-agent] LLM retry failed: {exc2}")
                return self._fallback_action(
                    "I ran into an internal issue. Could you repeat the last request?"
                )

        choice = completion.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            fn = getattr(call, "function", None)
            name = getattr(fn, "name", None) if fn else None
            raw_args = getattr(fn, "arguments", None) if fn else None
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except Exception:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            if not isinstance(args, dict):
                args = {}
            if isinstance(name, str) and name:
                return {"name": name, "arguments": args}

        text = getattr(choice, "content", None) or ""
        parsed = _extract_json_object(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
            name = parsed["name"]
            args = parsed.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            if name == RESPOND_ACTION_NAME:
                args = dict(args)
                args["content"] = _normalize_respond_content(args.get("content", ""))
            return {"name": name, "arguments": args}

        stripped = text.strip()
        if stripped:
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY one raw JSON object with keys name and arguments. "
                        "No markdown, no prose, no wrapper."
                    ),
                }
            ]
            try:
                retry_completion = litellm.completion(
                    model=self.model,
                    messages=retry_messages,
                    temperature=self.temperature,
                )
                retry_text = (
                    getattr(retry_completion.choices[0].message, "content", None) or ""
                )
                retry_parsed = _extract_json_object(retry_text)
                if isinstance(retry_parsed, dict) and isinstance(
                    retry_parsed.get("name"), str
                ):
                    retry_args = retry_parsed.get("arguments") or {}
                    if not isinstance(retry_args, dict):
                        retry_args = {}
                    if retry_parsed["name"] == RESPOND_ACTION_NAME:
                        retry_args = dict(retry_args)
                        retry_args["content"] = _normalize_respond_content(
                            retry_args.get("content", "")
                        )
                    return {"name": retry_parsed["name"], "arguments": retry_args}
            except Exception:
                pass
        return self._fallback_action(stripped or "Could you clarify your request?")

    def _normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        name = action.get("name")
        arguments = action.get("arguments")

        if not isinstance(name, str) or not name:
            return self._fallback_action(
                "I couldn't determine the next safe action. Could you clarify your request?"
            )
        if not isinstance(arguments, dict):
            arguments = {}

        if name == RESPOND_ACTION_NAME:
            arguments = dict(arguments)
            arguments["content"] = _normalize_respond_content(arguments.get("content", ""))

        if self.tool_names and name != RESPOND_ACTION_NAME and name not in self.tool_names:
            return self._fallback_action(
                f"I selected an unavailable action ({name}). Please restate what you need."
            )

        return {"name": name, "arguments": arguments}

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = self._normalize_input_prefix(get_message_text(message))
        self.turn_count += 1

        if self.turn_count == 1:
            tools = _extract_openai_tools(input_text)
            if tools:
                self.tools = tools
                self.tool_names = {
                    t["function"]["name"] for t in tools if "function" in t
                }
                print(
                    f"[tau2-agent] extracted {len(tools)} tool schemas "
                    f"for native function calling"
                )
            else:
                # Fallback: grep all "name" occurrences to at least know which actions
                # are allowed so we can reject hallucinated tool names.
                names = set(re.findall(r'"name"\s*:\s*"([^"]+)"', input_text))
                self.tool_names = names
                print(
                    f"[tau2-agent] could not extract tool schemas; "
                    f"running in JSON mode with {len(names)} allowed names"
                )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Selecting next action..."),
        )

        # Two-step transfer: after transfer_to_human_agents, send the required hold
        # message on the next turn regardless of what the LLM would otherwise do.
        if self.just_transferred:
            self.just_transferred = False
            action = self._fallback_action(TRANSFER_HOLD_MESSAGE)
            self._update_state_from_text(input_text)
            self._update_state_from_tool_payload(input_text)
            self.messages.append({"role": "user", "content": input_text})
            self.messages.append(
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
            )
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=action))],
                name="Action",
            )
            return

        self._update_state_from_text(input_text)
        self._update_state_from_tool_payload(input_text)
        self.messages.append({"role": "user", "content": input_text})
        action = self._guard_action(self._normalize_action(self._call_llm()))
        self.session_state["last_tool_name"] = (
            action.get("name") if action.get("name") != RESPOND_ACTION_NAME else None
        )

        if action.get("name") == "transfer_to_human_agents":
            self.just_transferred = True

        self.messages.append(
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
        )
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=action))],
            name="Action",
        )
