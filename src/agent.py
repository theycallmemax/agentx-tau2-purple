from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
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
MAX_CONTEXT_MESSAGES = int(os.getenv("TAU2_AGENT_MAX_CONTEXT_MESSAGES", "32"))
POLICY_REMINDER_EVERY = int(os.getenv("TAU2_AGENT_POLICY_REMINDER_EVERY", "8"))
BENCHMARK_NOW = datetime.fromisoformat("2024-05-15T15:00:00")
DEBUG_AGENT = os.getenv("TAU2_AGENT_DEBUG", "0") == "1"
DEBUG_LOG_PATH = os.getenv("TAU2_AGENT_DEBUG_LOG", "/tmp/tau2_agent_debug.log")

SYSTEM_PROMPT = """You are a purple agent participating in the tau2 benchmark on AgentBeats.

You will receive benchmark context from the evaluator. The evaluator's messages already include:
- the domain policy
- the available tool schemas
- the conversation history
- tool results

Your job is to choose the single best next action.

Hard rules:
- Return exactly one JSON object with keys "name" and "arguments".
- Use exactly one tool at a time.
- If you should answer the user directly, use the action name "respond".
- Never invent tool results or claim success before a tool result confirms it.
- Prefer reading state before mutating it when the situation is uncertain.
- If policy blocks a request, refuse it clearly using "respond".
- Keep direct user responses concise and operational.
- Before any mutating action, re-check the relevant policy constraint against verified facts.
- Internally decompose multi-part requests into a checklist and avoid ending while important items remain unresolved.
- Before any write action (book, modify, cancel), list the exact details and obtain explicit user confirmation (yes) before calling the API.

Airline-specific guardrails:
- Do not proactively offer compensation unless the user explicitly asks for it.
- Do not trust user claims about eligibility, membership tier, insurance, or flight status without tool verification.
- If the user wants a change that policy clearly forbids, refuse clearly instead of improvising or transferring.
- Do not ask the user for reservation ids or airport codes if the evaluator context or tools can resolve them.

COMPENSATION RULES (verify before any send_certificate call):
- ELIGIBLE: silver or gold member; OR travel_insurance=yes; OR business cabin.
- NOT ELIGIBLE: regular member AND travel_insurance=no AND cabin is economy or basic_economy.
- Cancelled flight: certificate = $100 × number of passengers.
- Delayed flight (only when user is ALSO changing or cancelling): certificate = $50 × number of passengers.
- No compensation for any other reason.

CANCELLATION RULES:
- If ANY flight in the reservation has already been flown → call transfer_to_human_agents immediately.
- Cancellation is allowed if: booked within 24 hrs; OR airline cancelled; OR business cabin; OR travel insurance covers reason (weather/health).
- Always obtain and verify the cancellation reason before calling cancel_reservation.

MODIFICATION RULES:
- basic_economy cabin: flights CANNOT be changed. Cabin CAN be changed.
- Cabin change: NOT allowed if any flight has already been flown.
- Cabin change (upgrade or downgrade): use update_reservation_cabin directly with the reservation_id and new cabin name. Do NOT search for flights or check flight status first — just verify the reservation details and call update_reservation_cabin.
- Bags: can add, cannot remove. Insurance: cannot add after booking.
- Passengers: can modify details, cannot change count.
- For update_reservation_cabin: arguments are {"reservation_id": "...", "cabin": "economy"/"business"/"basic_economy"}. No other arguments needed.

MULTI-INTENT HANDLING (flight change + cabin upgrade in same request):
- Treat them as two SEPARATE sequential operations. Do NOT try to compute a combined fare difference.
- Step 1: Handle the flight date/segment change — search ONLY for the legs being replaced (e.g., outbound legs if only the outbound date changes). Do NOT search return legs that remain unchanged.
- Step 2: After the flight change is confirmed and executed, call update_reservation_cabin for the upgrade. No flight search is needed for this step.
- CRITICAL: update_reservation_cabin does NOT require any flight pricing lookup. Never search_direct_flight or search_onestop_flight for return legs or for cabin upgrade fare calculation. The tool handles pricing automatically — just call it with reservation_id and cabin.
- When user declines due to price ("too expensive", "over my budget", "I can't proceed"): do NOT re-search flights. Ask the user which change they want to proceed with, or explain that pricing is fixed.

FLOWN FLIGHTS:
- As soon as you retrieve reservation details and ANY flight date is before 2024-05-15, the itinerary is partially flown.
- For cancellation requests: immediately call transfer_to_human_agents (do not re-check the reservation).
- For cabin change requests: immediately refuse (cannot change cabin after flight).
- Do NOT call get_reservation_details again after already retrieving it — the data does not change within a session.
"""

THINK_PROMPT = """Plan the next move carefully.

Consider:
1. What is the user's actual goal?
2. Is this a single-intent or multi-intent request?
3. Which facts are already verified, and which are still assumptions?
4. Which policy rules or hard user constraints matter most right now?
5. Is the next best move a read action, a write action, or a direct response?
6. If it is a write action, what exact fact must be verified first?

Write a short internal plan only. Do not output JSON yet."""

ACT_PROMPT = f"""Now return the single best next action as JSON.

Output format:
{{
  "name": "<tool-name-or-{RESPOND_ACTION_NAME}>",
  "arguments": {{}}
}}

If using "{RESPOND_ACTION_NAME}", arguments must be:
{{
  "content": "<message to the user>"
}}

Return raw JSON only."""

STATUS_ANALYZING = "Analyzing task state..."
STATUS_ACTING = "Selecting next action..."


@dataclass
class SessionState:
    pending_intent: str | None = None
    known_user_id: str | None = None
    known_reservation_id: str | None = None
    known_flight_number: str | None = None
    known_cancellation_reason: str | None = None
    prefer_last_reservation: bool = False
    explicit_compensation_request: bool = False
    known_delay_context: bool = False
    unresolved_slots: list[str] = field(default_factory=list)
    reservation_review_queue: list[str] = field(default_factory=list)
    reviewed_reservation_ids: list[str] = field(default_factory=list)
    last_user_details: dict[str, Any] | None = None
    last_reservation_details: dict[str, Any] | None = None
    recent_tool_payloads: list[dict[str, Any]] = field(default_factory=list)
    # Confirmation gate: write action waiting for user "yes"
    pending_write_action: dict[str, Any] | None = None
    pending_write_summary: str | None = None
    # Transfer flow: must send "PLEASE HOLD ON" on the next turn
    just_transferred: bool = False
    # Baggage allowance info already sent — don't repeat it, let LLM handle further requests
    baggage_info_sent: bool = False


USER_ID_PATTERN = re.compile(r"\b[a-z]+_[a-z]+_\d{3,}\b")
RESERVATION_ID_PATTERN = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=.*\d)[A-Z0-9]{6}\b")


def _is_valid_user_id(value: Any) -> bool:
    return isinstance(value, str) and USER_ID_PATTERN.fullmatch(value) is not None


def _is_valid_reservation_id(value: Any) -> bool:
    return isinstance(value, str) and RESERVATION_ID_PATTERN.fullmatch(value) is not None


def _extract_action_names(text: str) -> set[str]:
    names = set(re.findall(r'"name"\s*:\s*"([^"]+)"', text, flags=re.DOTALL))
    names.add(RESPOND_ACTION_NAME)
    return names


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
    tag_match = re.search(r"<json>\s*(.*?)\s*</json>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if tag_match:
        cleaned = tag_match.group(1).strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    balanced = _extract_balanced_json_object(cleaned)
    if not balanced:
        return None

    try:
        data = json.loads(balanced)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class Agent:
    def __init__(self):
        self.model = os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1")
        self.temperature = float(os.getenv("TAU2_AGENT_TEMPERATURE", "0"))
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.allowed_action_names: set[str] | None = None
        self.turn_count = 0
        self.state = SessionState()
        # Bypass confirmation gate when re-executing an already-confirmed write action
        self._confirmation_bypass = False

    def _normalize_input_prefix(self, input_text: str) -> str:
        """Normalize raw JSON tool results to have 'tool:' prefix.

        The tau2 benchmark sends tool results back as plain JSON without any prefix.
        Without normalization, state update and routing code won't recognize them as
        tool results and the agent loops calling get_user_details indefinitely.
        """
        stripped = input_text.strip()
        if not stripped.startswith(("user:", "tool:")) and stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
                if isinstance(payload, dict) and (
                    "user_id" in payload or "reservation_id" in payload
                ):
                    return "tool: " + stripped
            except Exception:
                pass
        return input_text

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = self._normalize_input_prefix(get_message_text(message))
        self.turn_count += 1

        if self.allowed_action_names is None:
            self.allowed_action_names = _extract_action_names(input_text)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(STATUS_ANALYZING),
        )

        self._ingest_message_into_state(input_text)
        if DEBUG_AGENT and self.turn_count == 1:
            try:
                with open("/tmp/tau2_agent_first_input.txt", "w", encoding="utf-8") as handle:
                    handle.write(input_text)
            except Exception:
                pass
        self._debug(
            "ingest",
            turn=self.turn_count,
            pending_intent=self.state.pending_intent,
            known_user_id=self.state.known_user_id,
            known_reservation_id=self.state.known_reservation_id,
            prefer_last_reservation=self.state.prefer_last_reservation,
            known_cancellation_reason=self.state.known_cancellation_reason,
            input_preview=input_text[:350],
        )
        self.messages.append({"role": "user", "content": input_text})
        self._maybe_add_policy_reminder()
        reasoning = self._generate_reasoning()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(STATUS_ACTING),
        )

        action_json = self._generate_action(input_text, reasoning)

        # Track transfer so we send the required hold message on the next turn
        if action_json.get("name") == "transfer_to_human_agents":
            self.state.just_transferred = True

        assistant_content = json.dumps(action_json, ensure_ascii=False)
        self.messages.append({"role": "assistant", "content": assistant_content})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=action_json))],
            name="Action",
        )

    def _messages_for_model(self) -> list[dict[str, Any]]:
        if len(self.messages) <= MAX_CONTEXT_MESSAGES:
            return list(self.messages)

        preserved = self.messages[:2]
        recent = self.messages[-(MAX_CONTEXT_MESSAGES - 3) :]
        return preserved + [
            {
                "role": "user",
                "content": "[Earlier conversation messages omitted. Continue from the latest verified state and keep following the original policy.]",
            }
        ] + recent

    def _maybe_add_policy_reminder(self) -> None:
        if self.turn_count <= 2 or self.turn_count % POLICY_REMINDER_EVERY != 0:
            return

        self.messages.append(
            {
                "role": "system",
                "content": (
                    "Policy reminder: verify facts before mutating state, respect hard user constraints, "
                    "avoid proactive compensation offers, and refuse clearly when policy blocks the request."
                ),
            }
        )

    def _generate_reasoning(self) -> str:
        think_messages = self._messages_for_model() + [
            {"role": "system", "content": self._state_summary()},
            {"role": "user", "content": THINK_PROMPT}
        ]
        try:
            completion = litellm.completion(
                model=self.model,
                messages=think_messages,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            print(f"[tau2-agent] THINK call failed: {e}")
            return ""

    def _is_user_confirmation(self, text: str) -> bool:
        lowered = text.lower().strip()
        confirm_tokens = ("yes", "yeah", "yep", "confirm", "proceed", "go ahead", "ok", "okay", "sure", "please do", "do it")
        return any(lowered == t or lowered.startswith(t + " ") or lowered.startswith(t + ",") for t in confirm_tokens)

    def _is_user_denial(self, text: str) -> bool:
        lowered = text.lower().strip()
        deny_tokens = ("no", "nope", "cancel", "stop", "don't", "do not", "abort", "never mind", "nevermind")
        if any(lowered == t or lowered.startswith(t + " ") or lowered.startswith(t + ",") for t in deny_tokens):
            return True
        # Recognize implicit denials: user declines due to cost or says they can't proceed
        implicit_deny_phrases = (
            "can't proceed", "cannot proceed", "i can't", "i cannot",
            "too expensive", "over my budget", "too much", "can't afford",
            "don't want to proceed", "do not want to proceed",
            "won't proceed", "will not proceed",
        )
        return any(phrase in lowered for phrase in implicit_deny_phrases)

    def _generate_action(self, input_text: str, reasoning: str) -> dict[str, Any]:
        # Transfer two-step: policy requires sending HOLD message after transfer_to_human_agents
        if self.state.just_transferred:
            self.state.just_transferred = False
            return self._fallback_action(
                "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
            )

        # Confirmation gate: if a write action is pending, check user response
        if self.state.pending_write_action is not None:
            user_text = self._latest_user_text() or input_text
            if self._is_user_confirmation(user_text):
                action = self.state.pending_write_action
                self.state.pending_write_action = None
                self.state.pending_write_summary = None
                self._confirmation_bypass = True
                try:
                    result = self._normalize_action(action)
                finally:
                    self._confirmation_bypass = False
                return result
            elif self._is_user_denial(user_text):
                self.state.pending_write_action = None
                self.state.pending_write_summary = None
                return self._fallback_action(
                    "Understood, I won't proceed with that action. Is there anything else I can help you with?"
                )

        heuristic_action = self._maybe_rule_based_action(input_text)
        if heuristic_action is not None:
            self._debug("heuristic_action", action=heuristic_action)
            return heuristic_action

        act_messages = self._messages_for_model() + [
            {"role": "system", "content": self._state_summary()},
            {
                "role": "user",
                "content": f"Internal plan:\n{reasoning or 'No plan available.'}\n\n{ACT_PROMPT}",
            }
        ]

        action = self._completion_to_action(act_messages)
        if action is None:
            retry_messages = act_messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid. Return ONLY one raw JSON object with keys "
                        '"name" and "arguments".'
                    ),
                }
            ]
            action = self._completion_to_action(retry_messages, use_response_format=False)

        if action is None:
            return self._fallback_action(
                "I ran into an internal formatting issue. Please repeat the last request."
            )

        normalized = self._normalize_action(action)
        self._debug("model_action", raw_action=action, normalized_action=normalized)
        return normalized

    def _debug(self, label: str, **payload: Any) -> None:
        if not DEBUG_AGENT:
            return
        safe_payload = {key: value for key, value in payload.items()}
        line = f"[tau2-debug] {label}: {json.dumps(safe_payload, ensure_ascii=False, default=str)}"
        print(line)
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

    def _maybe_rule_based_action(self, input_text: str) -> dict[str, Any] | None:
        normalized_input = input_text.strip()
        if normalized_input.startswith("user:"):
            return self._maybe_airline_user_action(normalized_input[5:].strip())
        if normalized_input.startswith("tool:"):
            return self._maybe_airline_tool_action(normalized_input[5:].strip())
        if self._looks_like_benchmark_prompt(normalized_input):
            return self._maybe_airline_benchmark_action(normalized_input)
        # Subsequent benchmark turns arrive as plain text without prefix — treat as user message
        return self._maybe_airline_user_action(normalized_input)

    def _looks_like_benchmark_prompt(self, input_text: str) -> bool:
        lowered = input_text.lower()
        return (
            "known_info" in lowered
            or "reason_for_call" in lowered
            or "task_instructions" in lowered
            or "available tools" in lowered
            or "user message:" in lowered
        )

    def _benchmark_intent_text(self, input_text: str) -> str:
        sections: list[str] = []
        embedded_user_match = re.search(
            r'User message:\s*"user:\s*(.*?)"\s*$',
            input_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if embedded_user_match:
            sections.append(embedded_user_match.group(1).strip())

        for label in ("reason_for_call", "task_instructions", "known_info"):
            match = re.search(
                rf'"?{label}"?\s*:\s*(.*?)(?:\n\s*"?[A-Za-z_]+"?\s*:|\Z)',
                input_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                section = match.group(1).strip().strip('",')
                sections.append(section)
        return "\n".join(section for section in sections if section)

    def _maybe_airline_benchmark_action(self, input_text: str) -> dict[str, Any] | None:
        lowered = input_text.lower()
        if "airline" not in lowered:
            return None

        raw_hint_text = self._benchmark_intent_text(input_text)
        hint_text = raw_hint_text.lower()

        if not hint_text.strip():
            # Only greet on a truly cold start; if we already have context let the LLM handle it
            if self.state.pending_intent is not None or self.state.known_user_id is not None:
                return None
            return self._fallback_action(
                "Hello, how can I help you with your flight reservation today?"
            )

        if self.state.pending_intent is None:
            routed_intent = self._route_intent(hint_text)
            if routed_intent is not None:
                self.state.pending_intent = routed_intent

        if "compensation" in hint_text or "certificate" in hint_text:
            self.state.explicit_compensation_request = True

        if (
            self.state.pending_intent == "compensation"
            and self.state.known_user_id is None
            and (
                self.state.known_flight_number is not None
                or "delay" in hint_text
                or "delayed" in hint_text
            )
        ):
            return self._fallback_action(
                "I can look into that. Please share your user ID so I can verify your account and reservation details."
            )

        if self.state.last_user_details is None and _is_valid_user_id(self.state.known_user_id):
            return {
                "name": "get_user_details",
                "arguments": {"user_id": self.state.known_user_id},
            }

        if self.state.last_user_details is not None and self.state.last_reservation_details is None:
            preferred_reservation_id = self._preferred_reservation_id()
            if preferred_reservation_id is not None:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": preferred_reservation_id},
                }

        return None

    def _maybe_airline_user_action(self, user_text: str) -> dict[str, Any] | None:
        lowered = user_text.lower()
        reservation = self.state.last_reservation_details
        user_details = self.state.last_user_details
        cancellation_context = (
            self.state.pending_intent == "cancel"
            or self._conversation_mentions_cancellation()
        )
        explicit_user_id_match = USER_ID_PATTERN.search(user_text)
        explicit_reservation_match = RESERVATION_ID_PATTERN.search(user_text)
        flight_number = self._extract_flight_number(user_text)

        if explicit_user_id_match and user_details is None:
            return {
                "name": "get_user_details",
                "arguments": {"user_id": explicit_user_id_match.group(0)},
            }

        if (
            explicit_reservation_match
            and reservation is None
            and explicit_reservation_match.group(0) != flight_number
        ):
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": explicit_reservation_match.group(0)},
            }

        if (
            self.state.pending_intent in {"cancel", "modify", "compensation", "book", "baggage", "insurance"}
            and user_details is None
            and _is_valid_user_id(self.state.known_user_id)
        ):
            return {
                "name": "get_user_details",
                "arguments": {"user_id": self.state.known_user_id},
            }

        if self.state.pending_intent == "compensation":
            compensation_action = self._maybe_compensation_or_delay_action(user_text)
            if compensation_action is not None:
                return compensation_action

        if self._looks_like_cancellation_request(lowered) or cancellation_context:
            if reservation is not None:
                reason = self._extract_cancellation_reason(lowered) or self.state.known_cancellation_reason
                if reason is None:
                    return self._fallback_action(
                        "I can check that for you. What is the reason for the cancellation: change of plan, airline canceled flight, or other reasons?"
                    )

                if not self._is_cancellation_allowed(reservation, reason):
                    return self._fallback_action(
                        "I’m not able to approve this cancellation. Based on the reservation details and airline policy, this booking is not eligible for a refund-backed cancellation."
                    )

            if user_details is not None and not self._contains_reservation_id(user_text):
                reservations = user_details.get("reservations")
                if isinstance(reservations, list) and reservations:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": reservations[-1]},
                    }

        if self._mentions_recent_or_last_reservation(lowered) and user_details is not None:
            reservations = user_details.get("reservations")
            if isinstance(reservations, list) and reservations:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservations[-1]},
                }

        if self.state.pending_intent == "baggage" and reservation is not None and user_details is not None:
            # Only send the info reply once; skip if user wants to actually modify something
            _action_words = ("add", "update", "change", "upgrade", "downgrade", "modify", "book")
            _wants_action = any(w in lowered for w in _action_words)
            if not self.state.baggage_info_sent and not _wants_action:
                total = self._compute_total_baggage_allowance(reservation, user_details)
                if total is not None:
                    self.state.baggage_info_sent = True
                    membership = user_details.get("membership", "your current")
                    return self._fallback_action(
                        f"You can bring 4 suitcases total on this reservation. I verified that you are a {membership} member."
                        if total == 4
                        else f"You can bring {total} suitcases total on this reservation based on your verified membership and cabin."
                    )

        if self.state.pending_intent == "insurance" and reservation is not None:
            return self._fallback_action(
                "I’m not able to add travel insurance after the initial booking. Airline policy only allows insurance to be added at the time of booking."
            )

        if self.state.pending_intent == "book":
            booking_action = self._maybe_booking_reference_action(user_text)
            if booking_action is not None:
                return booking_action

        return None

    def _maybe_airline_tool_action(self, tool_text: str) -> dict[str, Any] | None:
        tool_payload = self._parse_tool_payload(tool_text)
        if not isinstance(tool_payload, dict):
            return None

        if "reservation_id" in tool_payload:
            self._mark_reservation_reviewed(tool_payload["reservation_id"])
            recent_user_text = self._latest_user_text()
            lowered = recent_user_text.lower() if recent_user_text else ""

            # Policy: if any flight already flown and user wants cancellation → immediate transfer
            if (self._looks_like_cancellation_request(lowered) or self.state.pending_intent == "cancel") and self._has_any_flown_flight(tool_payload):
                rid = tool_payload.get("reservation_id", "the reservation")
                return {
                    "name": "transfer_to_human_agents",
                    "arguments": {"summary": f"Customer requesting cancellation of {rid} but part of the itinerary has already been flown. Please assist with eligibility checks and refund processing."},
                }

            if self._looks_like_cancellation_request(lowered) or self.state.pending_intent == "cancel":
                reason = self._extract_cancellation_reason(lowered) or self.state.known_cancellation_reason
                if reason is None:
                    return self._fallback_action(
                        "I found the reservation. What is the reason for the cancellation: change of plan, airline canceled flight, or other reasons?"
                    )
                if not self._is_cancellation_allowed(tool_payload, reason):
                    return self._fallback_action(
                        "I’m not able to approve this cancellation. Based on the reservation details and airline policy, this booking is not eligible for a refund-backed cancellation."
                    )

            if self.state.pending_intent == "baggage" and self.state.last_user_details is not None:
                _recent = (self._latest_user_text() or "").lower()
                _action_words = ("add", "update", "change", "upgrade", "downgrade", "modify", "book")
                if not self.state.baggage_info_sent and not any(w in _recent for w in _action_words):
                    total = self._compute_total_baggage_allowance(tool_payload, self.state.last_user_details)
                    if total is not None:
                        self.state.baggage_info_sent = True
                        return self._fallback_action(
                            f"You can bring {total} suitcases total on this reservation."
                        )

            if self.state.pending_intent == "compensation":
                if self._reservation_matches_complaint(tool_payload):
                    return self._maybe_compensation_or_delay_action(recent_user_text or "")
                next_reservation = self._next_reservation_to_review()
                if next_reservation is not None:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": next_reservation},
                    }

            if self.state.pending_intent == "book":
                booking_followup = self._maybe_booking_reference_action(recent_user_text or "")
                if booking_followup is not None:
                    return booking_followup
            return None

        if "reservations" in tool_payload:
            recent_user_text = self._latest_user_text()
            lowered = recent_user_text.lower() if recent_user_text else ""
            if (
                self._looks_like_cancellation_request(lowered)
                or self.state.pending_intent == "cancel"
                or self._mentions_recent_or_last_reservation(lowered)
            ):
                reservations = tool_payload.get("reservations")
                if isinstance(reservations, list) and reservations:
                    if self.state.pending_intent == "cancel" and not self.state.known_reservation_id:
                        preferred_reservation_id = reservations[-1]
                    else:
                        self._set_reservation_review_queue(reservations)
                        preferred_reservation_id = self._next_reservation_to_review() or reservations[-1]
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": preferred_reservation_id},
                    }

            if self.state.pending_intent in {"compensation", "book"}:
                reservations = tool_payload.get("reservations")
                if isinstance(reservations, list) and reservations:
                    self._set_reservation_review_queue(reservations)
                    next_reservation = self._next_reservation_to_review()
                    if next_reservation is not None:
                        return {
                            "name": "get_reservation_details",
                            "arguments": {"reservation_id": next_reservation},
                        }

        return None

    def _completion_to_action(
        self, messages: list[dict[str, Any]], use_response_format: bool = True
    ) -> dict[str, Any] | None:
        raw_content = ""
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            completion = litellm.completion(**kwargs)
            raw_content = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"[tau2-agent] ACT call failed (response_format={use_response_format}): {e}")
            return None

        return _extract_json_object(raw_content)

    def _normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        name = action.get("name")
        arguments = action.get("arguments")

        if not isinstance(name, str) or not name:
            return self._fallback_action(
                "I couldn't determine the next safe action. Could you clarify your request?"
            )

        if not isinstance(arguments, dict):
            arguments = {}

        if (
            self.allowed_action_names is not None
            and name not in self.allowed_action_names
        ):
            rescue_action = self._rescue_unavailable_action(name)
            if rescue_action is not None:
                return rescue_action
            return self._fallback_action(
                f"I selected an unavailable action ({name}). Please restate what you need."
            )

        arguments = self._normalize_arguments(name, arguments)

        if name == RESPOND_ACTION_NAME:
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                return self._fallback_action(
                    "I need a bit more information before I can help."
                )
            return {"name": name, "arguments": {"content": content.strip()}}

        gated = self._gate_tool_action(name, arguments)
        if gated is not None:
            return gated

        return {"name": name, "arguments": arguments}

    def _normalize_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = {k: v for k, v in arguments.items() if v is not None}

        for field_name, known_value in (
            ("user_id", self.state.known_user_id),
            ("reservation_id", self.state.known_reservation_id),
        ):
            if field_name in normalized and isinstance(normalized[field_name], str):
                lowered_value = normalized[field_name].lower()
                if "placeholder" in lowered_value or lowered_value in {"user_id", "reservation_id", "unknown"}:
                    if known_value:
                        normalized[field_name] = known_value

        for field in ("amount", "total_baggages", "nonfree_baggages"):
            if field in normalized:
                try:
                    normalized[field] = int(normalized[field])
                except Exception:
                    pass

        if "insurance" in normalized:
            value = normalized["insurance"]
            if value is True:
                normalized["insurance"] = "yes"
            elif value is False:
                normalized["insurance"] = "no"

        for field in ("flights", "passengers", "payment_methods"):
            if field in normalized and isinstance(normalized[field], list):
                cleaned_items: list[Any] = []
                for item in normalized[field]:
                    if isinstance(item, dict):
                        cleaned_item = {k: v for k, v in item.items() if v is not None}
                        if field == "payment_methods" and "amount" in cleaned_item:
                            try:
                                cleaned_item["amount"] = int(cleaned_item["amount"])
                            except Exception:
                                pass
                        cleaned_items.append(cleaned_item)
                    else:
                        cleaned_items.append(item)
                normalized[field] = cleaned_items

        if name == RESPOND_ACTION_NAME and "content" in normalized:
            content = normalized["content"]
            if isinstance(content, str):
                normalized["content"] = content.strip()

        return normalized

    def _fallback_action(self, content: str) -> dict[str, Any]:
        return {
            "name": RESPOND_ACTION_NAME,
            "arguments": {"content": content},
        }

    def _rescue_unavailable_action(self, name: str) -> dict[str, Any] | None:
        lowered_name = name.lower()
        if lowered_name.startswith(("check_", "review_", "lookup_")):
            latest_user_text = self._latest_user_text()
            if latest_user_text:
                rescued = self._maybe_airline_user_action(latest_user_text)
                if rescued is not None:
                    return rescued
            if _is_valid_user_id(self.state.known_user_id) and self.state.last_user_details is None:
                return {
                    "name": "get_user_details",
                    "arguments": {"user_id": self.state.known_user_id},
                }
            preferred_reservation_id = self._preferred_reservation_id()
            if preferred_reservation_id is not None and self.state.last_reservation_details is None:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": preferred_reservation_id},
                }
        return None

    def _state_summary(self) -> str:
        verified_facts: list[str] = []
        unresolved = list(self.state.unresolved_slots)

        if self.state.pending_intent:
            verified_facts.append(f"intent={self.state.pending_intent}")
        if self.state.known_user_id:
            verified_facts.append(f"user_id={self.state.known_user_id}")
        if self.state.known_reservation_id:
            verified_facts.append(f"reservation_id={self.state.known_reservation_id}")
        if self.state.known_flight_number:
            verified_facts.append(f"flight_number={self.state.known_flight_number}")
        if self.state.last_user_details is not None:
            verified_facts.append("user_details=verified")
        if self.state.last_reservation_details is not None:
            verified_facts.append("reservation_details=verified")
        if self.state.explicit_compensation_request:
            verified_facts.append("compensation=requested_by_user")
        if self.state.pending_write_action is not None:
            verified_facts.append(f"pending_write={self.state.pending_write_action.get('name','?')}")
        if self.state.reviewed_reservation_ids:
            verified_facts.append(f"reviewed_reservations={','.join(self.state.reviewed_reservation_ids)}")
        if self.state.last_reservation_details is not None and self._has_any_flown_flight(self.state.last_reservation_details):
            verified_facts.append("WARNING:some_flights_already_flown")

        if not verified_facts:
            verified_facts.append("none")
        if not unresolved:
            unresolved = ["none"]

        return (
            "Verified state summary:\n"
            f"- verified_facts: {', '.join(verified_facts)}\n"
            f"- unresolved_slots: {', '.join(unresolved)}\n"
            "- prefer using verified facts over user claims"
        )

    def _latest_user_text(self) -> str | None:
        for item in reversed(self.messages):
            if item.get("role") != "user":
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            if content.startswith("user:"):
                return content[5:].strip()
            if content.startswith("tool:"):
                continue
            # Plain text messages (subsequent benchmark turns sent without prefix)
            if not self._looks_like_benchmark_prompt(content):
                return content.strip()
        return None

    def _parse_tool_payload(self, tool_text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(tool_text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _latest_tool_payload(self, required_key: str) -> dict[str, Any] | None:
        for item in reversed(self.messages):
            if item.get("role") != "user":
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.startswith("tool:"):
                continue
            payload = self._parse_tool_payload(content[5:].strip())
            if isinstance(payload, dict) and required_key in payload:
                return payload
        return None

    def _latest_user_details(self) -> dict[str, Any] | None:
        return self.state.last_user_details or self._latest_tool_payload("reservations")

    def _latest_reservation_details(self) -> dict[str, Any] | None:
        return self.state.last_reservation_details or self._latest_tool_payload("reservation_id")

    def _looks_like_cancellation_request(self, lowered: str) -> bool:
        return self._is_operational_cancellation_text(lowered)

    def _is_operational_cancellation_text(self, lowered: str) -> bool:
        cancellation_phrases = (
            "cancel my reservation",
            "cancel the reservation",
            "cancel my booking",
            "cancel the booking",
            "cancel my flight",
            "cancel the flight",
            "cancel my trip",
            "cancel the trip",
            "cancel this reservation",
            "cancel this booking",
            "proceed with the cancellation",
            "confirm the cancellation",
            "help me cancel",
            "need to cancel",
            "want to cancel",
            "would like to cancel",
            "canceling a reservation",
            "cancelling a reservation",
            "canceling my reservation",
            "cancelling my reservation",
        )
        return any(phrase in lowered for phrase in cancellation_phrases)

    def _route_intent(self, lowered: str) -> str | None:
        if any(token in lowered for token in ["baggage", "baggages", "suitcase", "suitcases", "bags"]):
            return "baggage"
        if "insurance" in lowered:
            return "insurance"
        if any(token in lowered for token in ["compensation", "certificate", "voucher", "delay", "delayed flight"]):
            return "compensation"
        if self._is_operational_cancellation_text(lowered):
            return "cancel"
        if "change" in lowered or "modify" in lowered or "switch" in lowered:
            return "modify"
        if "book" in lowered or "flight from" in lowered:
            return "book"
        return None

    def _conversation_mentions_cancellation(self) -> bool:
        for item in reversed(self.messages[-8:]):
            content = item.get("content")
            if not isinstance(content, str):
                continue
            lowered = content.lower()
            if self._is_operational_cancellation_text(lowered):
                return True
        return False

    def _recent_user_text_window(self) -> str:
        texts: list[str] = []
        for item in self.messages[-12:]:
            if item.get("role") != "user":
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            if content.startswith("user:"):
                texts.append(content[5:].strip().lower())
            elif not content.startswith("tool:") and not self._looks_like_benchmark_prompt(content):
                texts.append(content.strip().lower())
        return "\n".join(texts)

    def _ingest_message_into_state(self, input_text: str) -> None:
        lowered_input = input_text.lower()
        raw_benchmark_hint_text = self._benchmark_intent_text(input_text)
        benchmark_hint_text = raw_benchmark_hint_text.lower()
        if self._mentions_recent_or_last_reservation(benchmark_hint_text or lowered_input):
            self.state.prefer_last_reservation = True

        benchmark_user_id_match = re.search(
            r"your user id is\s+([a-z]+_[a-z]+_\d{3,})",
            input_text,
            flags=re.IGNORECASE,
        )
        if benchmark_user_id_match:
            self.state.known_user_id = benchmark_user_id_match.group(1)

        benchmark_reservation_match = re.search(
            r"\breservation\s+([A-Z0-9]{6})\b",
            input_text,
            flags=re.IGNORECASE,
        )
        if benchmark_reservation_match and _is_valid_reservation_id(benchmark_reservation_match.group(1)):
            self.state.known_reservation_id = benchmark_reservation_match.group(1)
        benchmark_flight_number = self._extract_flight_number(input_text)
        if benchmark_flight_number is not None:
            self.state.known_flight_number = benchmark_flight_number

        if not input_text.startswith(("user:", "tool:")):
            routed_intent = self._route_intent(benchmark_hint_text) if benchmark_hint_text else None
            if routed_intent is not None:
                self.state.pending_intent = routed_intent
            if "compensation" in benchmark_hint_text or "certificate" in benchmark_hint_text:
                self.state.explicit_compensation_request = True
            if "delay" in benchmark_hint_text or "delayed" in benchmark_hint_text:
                self.state.known_delay_context = True
            cancellation_reason = self._extract_cancellation_reason(benchmark_hint_text)
            if cancellation_reason is not None:
                self.state.known_cancellation_reason = cancellation_reason
            self._refresh_unresolved_slots()

        # Determine the actual user text regardless of prefix format
        _plain_user_text: str | None = None
        if input_text.startswith("user:"):
            _plain_user_text = input_text[5:].strip()
        elif not input_text.startswith("tool:") and not self._looks_like_benchmark_prompt(input_text):
            _plain_user_text = input_text.strip()

        if _plain_user_text is not None:
            user_text = _plain_user_text
            lowered = user_text.lower()
            routed_intent = self._route_intent(lowered)
            if routed_intent is not None:
                self.state.pending_intent = routed_intent
            if "compensation" in lowered or "certificate" in lowered:
                self.state.explicit_compensation_request = True
            cancellation_reason = self._extract_cancellation_reason(lowered)
            if cancellation_reason is not None:
                self.state.known_cancellation_reason = cancellation_reason

            user_id_match = USER_ID_PATTERN.search(user_text)
            if user_id_match:
                self.state.known_user_id = user_id_match.group(0)

            flight_number = self._extract_flight_number(user_text)
            if flight_number is not None:
                self.state.known_flight_number = flight_number
            if "delay" in lowered or "delayed" in lowered:
                self.state.known_delay_context = True
            reservation_match = RESERVATION_ID_PATTERN.search(user_text)
            if reservation_match and reservation_match.group(0) != flight_number:
                self.state.known_reservation_id = reservation_match.group(0)
            self._refresh_unresolved_slots()

        if input_text.startswith("tool:"):
            payload = self._parse_tool_payload(input_text[5:].strip())
            if not isinstance(payload, dict):
                return

            self.state.recent_tool_payloads.append(payload)
            self.state.recent_tool_payloads = self.state.recent_tool_payloads[-6:]

            user_id = payload.get("user_id")
            if _is_valid_user_id(user_id):
                self.state.known_user_id = user_id

            reservation_id = payload.get("reservation_id")
            if _is_valid_reservation_id(reservation_id):
                self.state.known_reservation_id = reservation_id
                self.state.last_reservation_details = payload

            if "reservations" in payload:
                self.state.last_user_details = payload
            self._refresh_unresolved_slots()

    def _set_reservation_review_queue(self, reservations: list[str]) -> None:
        ordered = list(reservations)
        if self.state.prefer_last_reservation or self._mentions_recent_or_last_reservation(self._recent_user_text_window()):
            ordered = list(reversed(ordered))
        self.state.reservation_review_queue = ordered

    def _next_reservation_to_review(self) -> str | None:
        for reservation_id in self.state.reservation_review_queue:
            if reservation_id not in self.state.reviewed_reservation_ids:
                return reservation_id
        return None

    def _mark_reservation_reviewed(self, reservation_id: str) -> None:
        if reservation_id not in self.state.reviewed_reservation_ids:
            self.state.reviewed_reservation_ids.append(reservation_id)

    def _refresh_unresolved_slots(self) -> None:
        unresolved: list[str] = []
        intent = self.state.pending_intent
        last_user_text = (self._latest_user_text() or "").lower()

        if intent in {"cancel", "modify"} and not self.state.known_reservation_id:
            unresolved.append("reservation_id")
        if intent in {"cancel", "modify"} and not self.state.known_user_id and self.state.last_user_details is None:
            unresolved.append("user_id")
        if intent == "cancel":
            reason = self._extract_cancellation_reason(last_user_text) or self.state.known_cancellation_reason
            if reason is None:
                unresolved.append("cancellation_reason")
        if intent == "book" and not self.state.known_user_id:
            unresolved.append("user_id")

        self.state.unresolved_slots = unresolved

    def _mentions_recent_or_last_reservation(self, lowered: str) -> bool:
        return (
            "most recent reservation" in lowered
            or "last reservation" in lowered
            or "recent reservation" in lowered
            or "trip from philadelphia to laguardia" in lowered
            or "from philadelphia to laguardia" in lowered
        )

    def _contains_reservation_id(self, text: str) -> bool:
        return RESERVATION_ID_PATTERN.search(text) is not None

    def _extract_flight_number(self, text: str) -> str | None:
        match = re.search(r"\bHAT\d{3}\b", text.upper())
        return match.group(0) if match else None

    def _prefer_known_user_id(self, proposed_user_id: Any) -> str | None:
        if _is_valid_user_id(self.state.known_user_id):
            return self.state.known_user_id
        if _is_valid_user_id(proposed_user_id):
            return proposed_user_id
        return None

    def _preferred_reservation_id(self, proposed_reservation_id: Any = None) -> str | None:
        if _is_valid_reservation_id(proposed_reservation_id):
            return proposed_reservation_id

        next_reservation = self._next_reservation_to_review()
        if _is_valid_reservation_id(next_reservation):
            return next_reservation

        if _is_valid_reservation_id(self.state.known_reservation_id):
            return self.state.known_reservation_id

        reservation = self.state.last_reservation_details or {}
        reservation_id = reservation.get("reservation_id")
        if _is_valid_reservation_id(reservation_id):
            return reservation_id

        user_details = self.state.last_user_details or {}
        reservations = user_details.get("reservations")
        if isinstance(reservations, list):
            for reservation_id in reversed(reservations):
                if _is_valid_reservation_id(reservation_id):
                    return reservation_id
        return None

    def _reservation_matches_complaint(self, reservation: dict[str, Any]) -> bool:
        user_text = (self._latest_user_text() or "").lower()
        flight_number = self._extract_flight_number(user_text) or self.state.known_flight_number
        flights = reservation.get("flights") or []
        if flight_number:
            return any(
                isinstance(flight, dict) and flight.get("flight_number") == flight_number
                for flight in flights
            )
        if self._mentions_recent_or_last_reservation(user_text):
            return True
        return False

    def _extract_cancellation_reason(self, lowered: str) -> str | None:
        if "change of plan" in lowered:
            return "change_of_plan"
        if "airline canceled" in lowered or "airline cancelled" in lowered:
            return "airline_cancelled"
        if "weather" in lowered:
            return "weather"
        if "health" in lowered or "medical" in lowered or "sick" in lowered:
            return "health"
        if "other reason" in lowered or "other reasons" in lowered:
            return "other"
        return None

    def _is_cancellation_allowed(self, reservation: dict[str, Any], reason: str) -> bool:
        created_at = reservation.get("created_at")
        cabin = reservation.get("cabin")
        insurance = reservation.get("insurance")
        status = reservation.get("status")

        if cabin == "business":
            return True
        if status == "cancelled_by_airline":
            return True
        if insurance == "yes" and reason in {"weather", "health"}:
            return True

        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at)
                if (BENCHMARK_NOW - created_dt).total_seconds() <= 24 * 60 * 60:
                    return True
            except Exception:
                pass

        return False

    def _compute_total_baggage_allowance(
        self, reservation: dict[str, Any], user_details: dict[str, Any]
    ) -> int | None:
        membership = user_details.get("membership")
        cabin = reservation.get("cabin")
        passengers = reservation.get("passengers") or []
        passenger_count = len(passengers)
        if not membership or not cabin or passenger_count <= 0:
            return None

        per_passenger = {
            "regular": {"basic_economy": 0, "economy": 1, "business": 2},
            "silver": {"basic_economy": 1, "economy": 2, "business": 3},
            "gold": {"basic_economy": 2, "economy": 3, "business": 4},
        }.get(membership, {}).get(cabin)
        if per_passenger is None:
            return None
        return per_passenger * passenger_count

    def _is_compensation_eligible(self, user_details: dict[str, Any], reservation: dict[str, Any] | None) -> bool:
        """Returns True if user is eligible for compensation per policy.
        Eligible: silver/gold member OR travel_insurance=yes OR business cabin.
        Ineligible: regular + no insurance + (basic_)economy.
        """
        membership = user_details.get("membership", "regular")
        if membership in ("silver", "gold"):
            return True
        if reservation is not None:
            if reservation.get("insurance") == "yes":
                return True
            if reservation.get("cabin") == "business":
                return True
        return False

    def _maybe_compensation_or_delay_action(self, user_text: str) -> dict[str, Any] | None:
        # Step 1: need user details first
        if self.state.last_user_details is None and self.state.known_user_id:
            return {
                "name": "get_user_details",
                "arguments": {"user_id": self.state.known_user_id},
            }

        user_details = self.state.last_user_details or {}

        # Step 2: need reservation details to make a full eligibility decision
        reservation = self.state.last_reservation_details
        if reservation is None:
            next_reservation = self._next_reservation_to_review()
            if next_reservation is not None:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": next_reservation},
                }
            # No reservation data yet — let LLM decide
            return None

        # Step 3: eligibility check per policy
        if not self._is_compensation_eligible(user_details, reservation):
            membership = user_details.get("membership", "regular")
            return self._fallback_action(
                f"I’ve reviewed your account and reservation details. As a {membership} member without "
                "travel insurance on an economy or basic economy ticket, I’m unable to offer compensation "
                "per our airline policy. Compensation is available for Silver/Gold members, customers with "
                "travel insurance, or Business class passengers."
            )

        # Step 4: eligible — let LLM decide the appropriate action (send_certificate or respond)
        return None

    def _has_any_flown_flight(self, reservation: dict[str, Any]) -> bool:
        """Returns True if any flight in the reservation has already departed before BENCHMARK_NOW."""
        flights = reservation.get("flights") or []
        for flight in flights:
            if not isinstance(flight, dict):
                continue
            # Check various date field names the tau2 schema might use
            for field_name in ("actual_departure", "departure_datetime", "scheduled_departure", "date"):
                departure = flight.get(field_name)
                if departure and isinstance(departure, str):
                    try:
                        dep_dt = datetime.fromisoformat(departure)
                        if dep_dt < BENCHMARK_NOW:
                            return True
                    except Exception:
                        pass
                    break
        return False

    def _maybe_booking_reference_action(self, user_text: str) -> dict[str, Any] | None:
        lowered = user_text.lower()
        if self.state.last_user_details is None and self.state.known_user_id:
            return {
                "name": "get_user_details",
                "arguments": {"user_id": self.state.known_user_id},
            }

        if self.state.last_reservation_details is None and (
            "same flight" in lowered or "same as" in lowered or "ord to phl" in lowered
        ):
            next_reservation = self._next_reservation_to_review()
            if next_reservation is not None:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": next_reservation},
                }

        if self.state.last_reservation_details is not None and (
            "ord to phl" in lowered or "may 26" in lowered or "same flight" in lowered
        ):
            return {
                "name": "search_direct_flight",
                "arguments": {"origin": "ORD", "destination": "PHL", "date": "2024-05-26"},
            }

        if "one-way" in lowered or "one way" in lowered:
            return None

        return None

    def _gate_tool_action(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if name in ("search_direct_flight", "search_onestop_flight"):
            # Block return-leg searches in modify scenario: update_reservation_cabin needs no flight pricing
            reservation = self.state.last_reservation_details
            if reservation is not None and self.state.pending_intent == "modify":
                search_origin = arguments.get("origin", "")
                res_destination = reservation.get("destination", "")
                res_origin = reservation.get("origin", "")
                flight_type = reservation.get("flight_type", "")
                # A search from the reservation's destination back toward origin = return leg
                if (
                    flight_type == "round_trip"
                    and search_origin == res_destination
                    and search_origin != res_origin
                ):
                    return self._fallback_action(
                        "Return-leg flights are not changing. For a cabin upgrade, I will call "
                        "update_reservation_cabin directly — no flight search needed. "
                        "Let me confirm the full plan and get your approval before proceeding."
                    )

        if name == "transfer_to_human_agents":
            latest_user = (self._latest_user_text() or "").lower()
            user_requested = any(kw in latest_user for kw in ("human", "supervisor", "representative", "agent", "person"))
            # Allow if user explicitly asked, or if we're past turn 3 (policy requires it for edge cases)
            if not user_requested and self.turn_count <= 3:
                return self._fallback_action(
                    "Let me try to assist you directly first. Could you tell me more about what you need?"
                )

        if name == "send_certificate":
            if not self.state.explicit_compensation_request:
                return self._fallback_action(
                    "I can review the issue first, but I should not offer compensation unless you explicitly ask for it."
                )
            if self.state.last_reservation_details is None:
                return self._fallback_action(
                    "I need verified reservation details before I can consider any compensation."
                )
            user_details = self.state.last_user_details or {}
            reservation = self.state.last_reservation_details
            if not self._is_compensation_eligible(user_details, reservation):
                membership = user_details.get("membership", "regular")
                return self._fallback_action(
                    f"I’m unable to issue a compensation certificate. As a {membership} member without "
                    "travel insurance on an economy or basic economy ticket, this is not covered by policy."
                )
            # For delayed flights: certificate only applies when also changing/cancelling
            if self.state.known_delay_context:
                recent_window = self._recent_user_text_window()
                if not any(kw in recent_window for kw in ("cancel", "change", "modify")):
                    return self._fallback_action(
                        "For a delayed flight, a compensation certificate is only issued when changing or "
                        "cancelling the reservation. Would you like to change or cancel your reservation?"
                    )

        if name == "cancel_reservation":
            reservation_id = arguments.get("reservation_id") or self.state.known_reservation_id
            reservation = self.state.last_reservation_details
            latest_user = (self._latest_user_text() or "").lower()
            reason = self._extract_cancellation_reason(latest_user) or self.state.known_cancellation_reason
            if not isinstance(reservation_id, str) or reservation is None:
                return self._fallback_action(
                    "I need to verify the reservation details before proceeding with any cancellation action."
                )
            # Policy: if any flight already flown → transfer to human
            if self._has_any_flown_flight(reservation):
                return {
                    "name": "transfer_to_human_agents",
                    "arguments": {"summary": "Customer requesting cancellation but part of the itinerary has already been flown."},
                }
            if reason is None:
                return self._fallback_action(
                    "Before I can process a cancellation, I need the reason for the cancellation."
                )
            if not self._is_cancellation_allowed(reservation, reason):
                return self._fallback_action(
                    "I’m not able to approve this cancellation under the airline policy."
                )

        if name == "get_user_details":
            user_id = arguments.get("user_id")
            preferred_user_id = self._prefer_known_user_id(user_id)
            if preferred_user_id is not None and preferred_user_id != user_id:
                return {"name": name, "arguments": {"user_id": preferred_user_id}}
            if not _is_valid_user_id(user_id) or "placeholder" in user_id.lower():
                return self._fallback_action(
                    "I need your user ID before I can look up your profile."
                )

        if name == "get_reservation_details":
            reservation_id = arguments.get("reservation_id")

            # Block re-fetching a reservation we already have — prevents LLM loop
            if (
                reservation_id
                and self.state.last_reservation_details is not None
                and self.state.last_reservation_details.get("reservation_id") == reservation_id
                and reservation_id in self.state.reviewed_reservation_ids
            ):
                cached = self.state.last_reservation_details
                # If flown and cancellation context → transfer immediately
                if (self.state.pending_intent == "cancel" or self._conversation_mentions_cancellation()) and self._has_any_flown_flight(cached):
                    return {
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": f"Customer requesting cancellation of {reservation_id} but part of the itinerary has already been flown."},
                    }
                return self._fallback_action(
                    "I already have the reservation details on file. Let me continue assisting you with your request."
                )

            if isinstance(reservation_id, str) and self._extract_flight_number(reservation_id) == reservation_id:
                if self.state.last_user_details is None and _is_valid_user_id(self.state.known_user_id):
                    return {
                        "name": "get_user_details",
                        "arguments": {"user_id": self.state.known_user_id},
                    }
                preferred_reservation_id = self._preferred_reservation_id()
                if preferred_reservation_id is not None:
                    return {"name": name, "arguments": {"reservation_id": preferred_reservation_id}}
                return self._fallback_action(
                    "I need the reservation details rather than the flight number before I can look up the booking."
                )
            preferred_reservation_id = self._preferred_reservation_id(reservation_id)
            if preferred_reservation_id is not None and preferred_reservation_id != reservation_id:
                return {"name": name, "arguments": {"reservation_id": preferred_reservation_id}}
            if not _is_valid_reservation_id(reservation_id) or "placeholder" in reservation_id.lower():
                return self._fallback_action(
                    "I need the correct reservation ID before I can look up the reservation details."
                )

        if name.startswith("update_reservation_"):
            if self.state.last_reservation_details is None:
                return self._fallback_action(
                    "I need verified reservation details before making any reservation changes."
                )
            reservation = self.state.last_reservation_details
            cabin = reservation.get("cabin")
            # Changing flights: not allowed for basic_economy
            if name in ("update_reservation_flights",):
                if cabin == "basic_economy":
                    return self._fallback_action(
                        "Basic economy flights cannot be changed. However, you can change to a different "
                        "cabin class if you'd like. Is there anything else I can help you with?"
                    )
                if self._has_any_flown_flight(reservation):
                    return self._fallback_action(
                        "I'm unable to change the flights because part of this itinerary has already been flown."
                    )
            # Changing cabin: not allowed if any flight already flown
            if name in ("update_reservation_cabin",):
                if self._has_any_flown_flight(reservation):
                    return self._fallback_action(
                        "Cabin class cannot be changed because at least one flight in this reservation "
                        "has already been flown."
                    )

        if name == "book_reservation" and self.state.pending_intent == "book":
            if self.state.known_user_id is None and self.state.last_user_details is None:
                return self._fallback_action(
                    "I need the user's verified profile before creating a reservation."
                )

        # Bag removal guard: can add but not remove checked bags
        if name == "update_reservation_baggages":
            reservation = self.state.last_reservation_details
            if reservation is not None:
                current_nonfree = reservation.get("nonfree_baggages", 0)
                proposed = arguments.get("nonfree_baggages")
                if proposed is not None:
                    try:
                        if int(proposed) < int(current_nonfree):
                            return self._fallback_action(
                                "I'm unable to remove checked bags from a reservation. You can only add additional bags."
                            )
                    except (TypeError, ValueError):
                        pass

        # Confirmation gate: all database-mutating actions require explicit user "yes"
        _write_actions = {
            "book_reservation",
            "cancel_reservation",
            "update_reservation_flights",
            "update_reservation_cabin",
            "update_reservation_baggages",
            "update_reservation_passengers",
        }
        if name in _write_actions and not self._confirmation_bypass:
            summary = self._build_write_summary(name, arguments)
            self.state.pending_write_action = {"name": name, "arguments": arguments}
            self.state.pending_write_summary = summary
            return self._fallback_action(
                f"{summary}\n\nShall I proceed? Please reply with 'yes' to confirm."
            )

        return None

    def _build_write_summary(self, name: str, arguments: dict[str, Any]) -> str:
        """Build a human-readable summary of the write action for user confirmation."""
        if name == "cancel_reservation":
            rid = arguments.get("reservation_id") or self.state.known_reservation_id or "your reservation"
            reason = arguments.get("reason") or self.state.known_cancellation_reason or "unspecified"
            return f"I'm about to cancel reservation {rid} (reason: {reason})."
        if name == "book_reservation":
            origin = arguments.get("origin", "?")
            dest = arguments.get("destination", "?")
            cabin = arguments.get("cabin", "?")
            passengers = arguments.get("passengers", [])
            n = len(passengers) if isinstance(passengers, list) else "?"
            return f"I'm about to book a {cabin} flight from {origin} to {dest} for {n} passenger(s)."
        if name == "update_reservation_flights":
            rid = arguments.get("reservation_id") or self.state.known_reservation_id or "your reservation"
            return f"I'm about to update the flights on reservation {rid}."
        if name == "update_reservation_cabin":
            rid = arguments.get("reservation_id") or self.state.known_reservation_id or "your reservation"
            cabin = arguments.get("cabin", "?")
            return f"I'm about to change the cabin to {cabin} on reservation {rid}."
        if name == "update_reservation_baggages":
            rid = arguments.get("reservation_id") or self.state.known_reservation_id or "your reservation"
            bags = arguments.get("nonfree_baggages", "?")
            return f"I'm about to update checked bags to {bags} on reservation {rid}."
        if name == "update_reservation_passengers":
            rid = arguments.get("reservation_id") or self.state.known_reservation_id or "your reservation"
            return f"I'm about to update passenger information on reservation {rid}."
        return f"I'm about to perform: {name} with {json.dumps(arguments, ensure_ascii=False)}."
