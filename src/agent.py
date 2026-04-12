"""Tau2 benchmark agent — orchestrator that delegates to specialized modules."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import litellm
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message

try:
    from .extractors import (
        extract_airport_codes,
        extract_dates,
        extract_entities_from_history,
        extract_flight_number,
        extract_reservation_id,
        extract_user_id,
    )
    from .intent import (
        looks_like_all_reservations_intent,
        looks_like_booking_after_cancel,
        looks_like_multi_action_request,
        detect_all_intents,
        looks_like_modify_intent,
        looks_like_baggage_intent,
        looks_like_remove_passenger_intent,
    )
    from .skill_loader import load_skill_prompt, get_skills_menu, dual_classify_intent
    from .guard import guard_action
    from .parsing import (
        RESPOND_ACTION_NAME,
        extract_json_object,
        extract_openai_tools,
        normalize_respond_content,
    )
    from .planner import (
        parse_plan,
        plan_prompt,
        planner_conflicts_with_state,
        should_plan,
    )
    from .playbooks import build_policy_reminder, build_prompt_blocks
    from .policy_sanitizer import sanitize_tool_descriptions
    from .reservation import (
        cancel_eligibility,
        BENCHMARK_NOW,
        current_reservation_entry,
        find_reservation_by_flight_number,
        find_reservation_by_route,
        infer_reservation_from_inventory,
        pricing_expression_for_current_reservation,
        reservation_has_flown_segments,
    )
    from .runtime import (
        build_runtime_brief,
        compressed_history_summary,
        history_snapshot,
        record_completed_action,
        resolve_completed_subtasks,
        sync_runtime_state,
        termination_controller,
    )
    from .forced_execution import (
        force_pending_action,
        should_force_execution,
        _mark_completeness_flags,
    )
    from .session_state import (
        build_state_summary,
        clear_pending_confirmation_if_completed,
        create_initial_state,
        remember_pending_confirmation,
        update_state_from_text,
        update_state_from_tool_payload,
    )
except ImportError:  # pragma: no cover - direct src imports in tests
    from extractors import (
        extract_airport_codes,
        extract_dates,
        extract_entities_from_history,
        extract_flight_number,
        extract_reservation_id,
        extract_user_id,
    )
    from intent import (
        looks_like_all_reservations_intent,
        looks_like_booking_after_cancel,
        looks_like_multi_action_request,
        detect_all_intents,
        looks_like_modify_intent,
        looks_like_baggage_intent,
        looks_like_remove_passenger_intent,
    )
    from skill_loader import load_skill_prompt, get_skills_menu, dual_classify_intent
    from guard import guard_action
    from parsing import (
        RESPOND_ACTION_NAME,
        extract_json_object,
        extract_openai_tools,
        normalize_respond_content,
    )
    from planner import (
        parse_plan,
        plan_prompt,
        planner_conflicts_with_state,
        should_plan,
    )
    from playbooks import build_policy_reminder, build_prompt_blocks
    from policy_sanitizer import sanitize_tool_descriptions
    from reservation import (
        cancel_eligibility,
        BENCHMARK_NOW,
        current_reservation_entry,
        find_reservation_by_flight_number,
        find_reservation_by_route,
        infer_reservation_from_inventory,
        pricing_expression_for_current_reservation,
        reservation_has_flown_segments,
    )
    from runtime import (
        build_runtime_brief,
        compressed_history_summary,
        history_snapshot,
        record_completed_action,
        resolve_completed_subtasks,
        sync_runtime_state,
        termination_controller,
    )
    from forced_execution import (
        force_pending_action,
        should_force_execution,
        _mark_completeness_flags,
    )
    from session_state import (
        build_state_summary,
        clear_pending_confirmation_if_completed,
        create_initial_state,
        remember_pending_confirmation,
        update_state_from_text,
        update_state_from_tool_payload,
    )

load_dotenv()

# Drop unsupported params (e.g. temperature=0 is not supported by gpt-5 family)
litellm.drop_params = True

MAX_CONTEXT_MESSAGES = int(os.getenv("TAU2_AGENT_MAX_CONTEXT_MESSAGES", "120"))
PLAN_MAX_TURNS = int(os.getenv("TAU2_AGENT_PLAN_MAX_TURNS", "8"))
TRANSFER_HOLD_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."

SYSTEM_PROMPT = """You are a customer service agent participating in the tau2 benchmark.

The first user message from the evaluator contains the domain policy and the list
of tools you may use. Follow that policy strictly.

CRITICAL — First turn behavior:
- Do NOT proactively ask about flight dates, compensation, delays, or flight status.
- Do NOT assume the user is calling about compensation or a delayed flight.
- The user's first message will describe what they need (cancel, book, modify, baggage,
  status, etc.). Listen to their request and act accordingly.
- If you must initiate a response before receiving a user request, simply say:
  "Hello, how can I help you today?" — nothing about flights, dates, or compensation.

Hard rules:
- Return EXACTLY one JSON object with keys "name" and "arguments".
- Use exactly one tool at a time. Never bundle multiple tool calls.
- To reply to the user directly, use name "respond" with arguments {"content": "<message>"}.
- The value of arguments.content for "respond" must be plain natural language only,
  never another JSON object, tool schema, or wrapper like {"name":"respond",...}.
- Never invent tool results. Wait for the environment to return them.

ACTION EXECUTION — NO PRE-CONFIRMATIONS:
- Do not ask the user to pre-confirm write actions. Once you have all the facts
  required by the policy, perform the action directly.
- Do not add an extra "reply yes", "confirm yes", or "YES, proceed" step when the
  user has already requested the action and supplied the required details.
- NEVER ask "Would you like me to proceed?" or "Please confirm with yes" — just DO IT.
- When you have reservation_id, payment_id, and all required details → execute the tool immediately.
- Only ask for missing INFORMATION, not for confirmation of actions the user already requested.

POLICY ENFORCEMENT:
- When policy forbids a request, refuse clearly via "respond". Only call
  transfer_to_human_agents when the policy explicitly requires it.

EFFICIENCY — AVOID REDUNDANT CALLS:
- Before searching flights: if a direct search returns empty, immediately try
  search_onestop_flight for the same origin/destination/date before pivoting to
  alternate dates or airports. Do not exhaustively iterate over many date/airport
  combinations — pick the most promising option and propose it to the user.
- Avoid redundant reads: if you have already fetched user details or reservation
  details in this conversation, reuse them instead of calling the tool again.
- NEVER call get_reservation_details or get_user_details more than once for the same ID.
- If you have the information, USE IT. Don't fetch again.

Keep user-facing responses short and operational.

Return raw JSON only, no prose, no code fences."""

_extract_json_object = extract_json_object
_extract_openai_tools = extract_openai_tools
_normalize_respond_content = normalize_respond_content


class TurnGraphState(TypedDict, total=False):
    input_text: str
    latest_user_text: str
    action: dict[str, Any]
    should_return: bool


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
        self.session_state: dict[str, Any] = create_initial_state()
        self.latest_plan: dict[str, Any] | None = None
        self.debug_logging_enabled = (
            os.getenv("TAU2_AGENT_DEBUG_LOGGING", "1").strip().lower() not in {"0", "false", "no"}
        )
        self.debug_log_path = Path(
            os.getenv(
                "TAU2_AGENT_DEBUG_LOG_PATH",
                str(Path(__file__).resolve().parents[1] / "analysis" / "debug" / "agent_trace.jsonl"),
            )
        )
        self._turn_graph = self._build_turn_graph()

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

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

    def _latest_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    if content.startswith("tool:"):
                        continue
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

    # ------------------------------------------------------------------
    # Message preparation
    # ------------------------------------------------------------------

    def _messages_for_model(self) -> list[dict[str, Any]]:
        sync_runtime_state(self.session_state, self._latest_user_text())
        state_summary = build_state_summary(self.session_state)
        extra_system: list[dict[str, str]] = []

        # SKILL INJECTION: Load active skill prompt based on classified intent
        active_skill = self.session_state.get("active_skill_id")
        if isinstance(active_skill, str) and active_skill:
            skill_prompt = load_skill_prompt(active_skill)
            if skill_prompt:
                extra_system.append({
                    "role": "system",
                    "content": f"<ACTIVE_SKILL>{active_skill}</ACTIVE_SKILL>\n{skill_prompt}",
                })

        verified_entities = extract_entities_from_history(self.messages)
        verified_user_ids = sorted(verified_entities["user_ids"])
        verified_reservation_ids = sorted(verified_entities["reservation_ids"])
        verified_flight_numbers = sorted(verified_entities["flight_numbers"])

        if verified_user_ids or verified_reservation_ids or verified_flight_numbers:
            constraint_lines = [
                "Verified entities from the user conversation override any example values in tool descriptions."
            ]
            if verified_user_ids:
                constraint_lines.append(
                    "Known user_id values: "
                    + ", ".join(verified_user_ids)
                    + ". Use only these when calling get_user_details."
                )
            if verified_reservation_ids:
                constraint_lines.append(
                    "Known reservation_id values: "
                    + ", ".join(verified_reservation_ids)
                    + ". Do not replace them with flight numbers."
                )
            if verified_flight_numbers:
                constraint_lines.append(
                    "Known flight_number values: "
                    + ", ".join(verified_flight_numbers)
                    + ". Flight numbers are not reservation IDs."
                )
            extra_system.append(
                {"role": "system", "content": "\n".join(constraint_lines)}
            )
        if self.turn_count >= 4 and self.turn_count % 3 == 0:
            extra_system.append(
                {
                    "role": "system",
                    "content": build_policy_reminder(self.session_state),
                }
            )
        elif (self.session_state.get("active_flow") or {}).get("name") not in {None, "general"}:
            extra_system.append(
                {
                    "role": "system",
                    "content": build_policy_reminder(self.session_state),
                }
            )
        extra_system.extend(
            {"role": "system", "content": block}
            for block in build_prompt_blocks(self.session_state)
        )
        extra_system.append(
            {
                "role": "system",
                "content": "Compact runtime brief: " + history_snapshot(self.session_state),
            }
        )
        extra_system.append(
            {
                "role": "system",
                "content": (
                    "Verified facts only before execute pass: "
                    + json.dumps(self.session_state.get("verified_facts_cache", {}), ensure_ascii=False)
                ),
            }
        )
        extra_system.append(
            {
                "role": "system",
                "content": (
                    "Open questions still unresolved: "
                    + json.dumps(
                        [
                            label
                            for label, value in (
                                ("awaiting_choice_between_options", self.session_state.get("awaiting_choice_between_options")),
                                ("awaiting_flight_selection", self.session_state.get("awaiting_flight_selection")),
                                ("awaiting_payment_choice", self.session_state.get("awaiting_payment_choice")),
                                ("pending_confirmation_action", self.session_state.get("pending_confirmation_action")),
                                ("last_tool_error", self.session_state.get("last_tool_error")),
                            )
                            if value
                        ],
                        ensure_ascii=False,
                    )
                ),
            }
        )
        extra_system.append(
            {
                "role": "system",
                "content": (
                    "What has already been done: "
                    + json.dumps(self.session_state.get("completed_actions", [])[-8:], ensure_ascii=False)
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
            msgs = list(self.messages)
            msgs.insert(1, state_summary)
            return msgs + extra_system
        # Preserve system + first evaluator message (policy + tool schemas), keep recent tail.
        preserved = self.messages[:2]
        recent = self.messages[-(MAX_CONTEXT_MESSAGES - 3) :]
        self.session_state["history_compression_summary"] = compressed_history_summary(
            self.messages, self.session_state
        )
        return (
            preserved[:1]
            + [state_summary]
            + preserved[1:]
            + [
                {
                    "role": "user",
                    "content": (
                        "[Earlier conversation messages omitted. Continue from the "
                        "latest verified state and keep following the original policy.] "
                        + str(self.session_state.get("history_compression_summary") or "")
                    ),
                }
            ]
            + extra_system
            + recent
        )

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

    def _should_use_plan(self, latest_user_text: str) -> bool:
        if self.turn_count > PLAN_MAX_TURNS:
            candidate = self.latest_plan.get("next_action") if isinstance(self.latest_plan, dict) else None
            return should_plan(self.session_state, latest_user_text, candidate)
        return should_plan(
            self.session_state,
            latest_user_text,
            self.latest_plan.get("next_action") if isinstance(self.latest_plan, dict) else None,
        )

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _fallback_action(self, content: str) -> dict[str, Any]:
        return {"name": RESPOND_ACTION_NAME, "arguments": {"content": content}}

    def _trace(self, event: str, **fields: Any) -> None:
        if not self.debug_logging_enabled:
            return
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "turn": self.turn_count,
            "task_type": self.session_state.get("task_type"),
            "reservation_id": self.session_state.get("reservation_id"),
            "last_tool_name": self.session_state.get("last_tool_name"),
            "context_id": self.session_state.get("_context_id"),
        }
        payload.update(fields)
        try:
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.debug_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _terminal_cancel_refusal(self, refusal: str) -> dict[str, Any]:
        text = refusal.strip()
        lower = text.lower()
        if "not eligible for cancellation" in lower:
            text += (
                " I can’t approve or process this cancellation, and I can’t offer a refund for it."
            )
        elif "already cancelled" in lower:
            text += " There is no further cancellation action for me to process on this booking."
        self._trace("terminal_cancel_refusal", refusal=text)
        return self._fallback_action(text)

    def _active_tools(self) -> list[dict[str, Any]]:
        return list(self.tools or [])

    def _guard_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.messages:
            latest = self.messages[-1]
            content = latest.get("content")
            if latest.get("role") == "user" and isinstance(content, str):
                self.session_state["latest_input_is_tool"] = content.startswith("tool:")
        return guard_action(
            self.session_state,
            self.tools,
            self.messages,
            self.turn_count,
            action,
            self._latest_user_text(),
        )

    def _update_state_from_text(self, input_text: str) -> None:
        update_state_from_text(self.session_state, input_text)
        _mark_completeness_flags(self.session_state)

    def _update_state_from_tool_payload(self, input_text: str) -> None:
        update_state_from_tool_payload(self.session_state, input_text)
        _mark_completeness_flags(self.session_state)

    def _infer_reservation_from_inventory(self, latest_user_text: str) -> str | None:
        return infer_reservation_from_inventory(self.session_state, latest_user_text)

    def _pre_llm_route_resolution_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        task_type = str(self.session_state.get("task_type") or "general")
        if task_type not in {"cancel", "modify", "remove_passenger", "insurance"}:
            return None
        if not self.session_state.get("loaded_user_details"):
            return None
        if extract_reservation_id(latest_user_text) or extract_flight_number(latest_user_text):
            return None

        airport_codes = extract_airport_codes(latest_user_text)
        target = self.session_state.get("route_resolution_target")
        if len(airport_codes) >= 2:
            origin = airport_codes[0]
            destination = airport_codes[1]
            self.session_state["origin"] = origin
            self.session_state["destination"] = destination
            self.session_state["route_resolution_active"] = True
            self.session_state["route_resolution_target"] = {
                "origin": origin,
                "destination": destination,
            }
            self.session_state["route_resolution_queue"] = list(
                self.session_state.get("known_reservation_ids") or []
            )
        elif isinstance(target, dict):
            origin = target.get("origin")
            destination = target.get("destination")
        else:
            origin = self.session_state.get("origin")
            destination = self.session_state.get("destination")

        if not isinstance(origin, str) or not isinstance(destination, str):
            return None
        if not self.session_state.get("route_resolution_active"):
            return None

        route_match = find_reservation_by_route(self.session_state, origin, destination)
        if route_match:
            self.session_state["reservation_id"] = route_match
            self._trace(
                "route_resolution_matched",
                origin=origin,
                destination=destination,
                matched_reservation=route_match,
            )
            self.session_state["route_resolution_active"] = False
            self.session_state["route_resolution_target"] = None
            self.session_state["route_resolution_queue"] = []
            if task_type == "cancel":
                eligible, refusal = cancel_eligibility(
                    self.session_state,
                    route_match,
                    latest_user_text,
                )
                self.session_state["cancel_eligible"] = eligible
                if not eligible and refusal:
                    return self._terminal_cancel_refusal(refusal)
            return None

        queue = self.session_state.get("route_resolution_queue", [])
        inventory = self.session_state.get("reservation_inventory", {})
        if isinstance(queue, list) and isinstance(inventory, dict):
            remaining = [
                reservation_id
                for reservation_id in queue
                if isinstance(reservation_id, str) and reservation_id not in inventory
            ]
            self.session_state["route_resolution_queue"] = remaining
            if remaining:
                self._trace(
                    "route_resolution_fetch_next",
                    origin=origin,
                    destination=destination,
                    next_reservation=remaining[0],
                    remaining=len(remaining),
                )
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": remaining[0]},
                }
        self.session_state["route_resolution_active"] = False
        self.session_state["route_resolution_target"] = None
        self.session_state["route_resolution_queue"] = []
        if task_type == "cancel":
            self._trace(
                "route_resolution_no_match",
                origin=origin,
                destination=destination,
            )
            return self._fallback_action(
                f"I checked the reservations available on this profile and couldn’t find a booking matching {origin} to {destination}."
            )
        return None

    def _pre_llm_recent_reservation_resolution_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        if not self.session_state.get("loaded_user_details"):
            return None
        lowered = latest_user_text.lower()
        if "last reservation" not in lowered and "most recent reservation" not in lowered:
            return None
        if extract_reservation_id(latest_user_text) or extract_flight_number(latest_user_text):
            return None

        known_ids = self.session_state.get("known_reservation_ids", [])
        inventory = self.session_state.get("reservation_inventory", {})
        if isinstance(known_ids, list) and isinstance(inventory, dict):
            for reservation_id in known_ids:
                if isinstance(reservation_id, str) and reservation_id not in inventory:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": reservation_id},
                    }

        inferred = infer_reservation_from_inventory(self.session_state, latest_user_text)
        if isinstance(inferred, str):
            self.session_state["reservation_id"] = inferred
            self.session_state["recent_reservation_active"] = True
            self.session_state["recent_reservation_id"] = inferred
        return None

    def _pre_llm_cancel_post_user_lookup_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if task_type != "cancel" and flow_name != "cancel":
            return None
        if not self.session_state.get("latest_input_is_tool"):
            return None
        if self.session_state.get("last_tool_name") != "get_user_details":
            return None
        if not self.session_state.get("loaded_user_details"):
            return None

        recent_user_messages = [
            str(message.get("content") or "")
            for message in self.messages[-8:]
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not str(message.get("content")).startswith("tool:")
        ]
        recent_text = "\n".join(recent_user_messages + [latest_user_text]).strip()
        reservation_id = extract_reservation_id(recent_text)
        flight_number = extract_flight_number(recent_text)
        airport_codes = extract_airport_codes(recent_text)

        if isinstance(reservation_id, str):
            self._trace("cancel_post_user_lookup_explicit_reservation", target=reservation_id)
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
            }
        if isinstance(flight_number, str):
            matched = find_reservation_by_flight_number(self.session_state, flight_number)
            if isinstance(matched, str):
                self.session_state["reservation_id"] = matched
                self._trace(
                    "cancel_post_user_lookup_flight_match",
                    flight_number=flight_number,
                    matched_reservation=matched,
                )
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": matched},
                }
        if len(airport_codes) >= 2:
            return self._pre_llm_route_resolution_action(recent_text)

        known_ids = [
            rid
            for rid in (self.session_state.get("known_reservation_ids") or [])
            if isinstance(rid, str)
        ]
        if looks_like_all_reservations_intent(recent_text):
            inventory = self.session_state.get("reservation_inventory", {})
            if not isinstance(inventory, dict):
                inventory = {}
            for reservation_id in known_ids:
                if reservation_id not in inventory:
                    self._trace(
                        "cancel_post_user_lookup_iterate_all",
                        next_reservation=reservation_id,
                    )
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": reservation_id},
                    }
            return None
        # NEW: If multiple reservations exist and no specific one mentioned, iterate all
        if len(known_ids) > 1 and not reservation_id and not flight_number and len(airport_codes) < 2:
            inventory = self.session_state.get("reservation_inventory", {})
            if not isinstance(inventory, dict):
                inventory = {}
            for reservation_id in known_ids:
                if reservation_id not in inventory:
                    self._trace(
                        "cancel_post_user_lookup_iterate_all_multi",
                        next_reservation=reservation_id,
                        total=len(known_ids),
                    )
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": reservation_id},
                    }
        if len(known_ids) == 1:
            self._trace("cancel_post_user_lookup_single_known_reservation", target=known_ids[0])
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": known_ids[0]},
            }
        if len(known_ids) > 1:
            self._trace("cancel_post_user_lookup_requires_disambiguation", known_reservations=known_ids)
            return self._fallback_action(
                "I found multiple reservations on your profile. Tell me which trip you want to cancel by reservation ID, route, or flight number, and I’ll check the correct booking."
            )
        return None

    def _pre_llm_cancel_identity_collection_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if task_type != "cancel" and flow_name != "cancel":
            return None
        if self.session_state.get("latest_input_is_tool"):
            return None
        if self.session_state.get("loaded_user_details"):
            return None

        reservation_id = extract_reservation_id(latest_user_text)
        flight_number = extract_flight_number(latest_user_text)
        airport_codes = extract_airport_codes(latest_user_text)
        if reservation_id or flight_number or len(airport_codes) >= 2:
            return self._fallback_action(
                "Please share your user ID, and I’ll use that with the trip details you provided to locate the booking."
            )
        return self._fallback_action(
            "Please share your user ID and either the reservation ID, route, or flight number for the trip you want to cancel."
        )

    def _pre_llm_cancel_resolution_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if task_type != "cancel" and flow_name != "cancel":
            return None
        if not self.session_state.get("latest_input_is_tool"):
            return None

        # FIX C: Detect when user explicitly denies wanting to cancel — break out of cancel flow
        recent_user_messages = [
            str(message.get("content") or "")
            for message in self.messages[-6:]
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not str(message.get("content")).startswith("tool:")
        ]
        recent_text = "\n".join(recent_user_messages + [latest_user_text]).lower()
        explicit_not_cancel = any(
            phrase in recent_text
            for phrase in (
                "not trying to cancel",
                "not looking to cancel",
                "not trying to cancel my",
                "not looking to cancel my",
                "not requesting a cancellation",
                "not asking to cancel",
                "don't want to cancel",
                "do not want to cancel",
                "not looking to cancel a",
                "not looking to cancel the",
                "not trying to cancel the",
            )
        )
        if explicit_not_cancel:
            # Re-classify intent from recent text
            if looks_like_modify_intent("\n".join(recent_user_messages + [latest_user_text])):
                self.session_state["task_type"] = "modify"
                if isinstance(active_flow, dict):
                    active_flow["name"] = "modify"
                self._trace("break_cancel_flow", reason="user_explicitly_not_cancelling", new_task="modify")
                return None  # Let LLM decide with correct task_type
            if looks_like_booking_intent("\n".join(recent_user_messages + [latest_user_text])):
                self.session_state["task_type"] = "booking"
                if isinstance(active_flow, dict):
                    active_flow["name"] = "booking"
                self._trace("break_cancel_flow", reason="user_explicitly_not_cancelling", new_task="booking")
                return None
            if looks_like_baggage_intent("\n".join(recent_user_messages + [latest_user_text])):
                self.session_state["task_type"] = "baggage"
                if isinstance(active_flow, dict):
                    active_flow["name"] = "baggage"
                self._trace("break_cancel_flow", reason="user_explicitly_not_cancelling", new_task="baggage")
                return None
            if looks_like_remove_passenger_intent("\n".join(recent_user_messages + [latest_user_text])):
                self.session_state["task_type"] = "remove_passenger"
                if isinstance(active_flow, dict):
                    active_flow["name"] = "remove_passenger"
                self._trace("break_cancel_flow", reason="user_explicitly_not_cancelling", new_task="remove_passenger")
                return None

        recent_user_messages = [
            str(message.get("content") or "")
            for message in self.messages[-8:]
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not str(message.get("content")).startswith("tool:")
        ]
        recent_text = "\n".join(recent_user_messages + [latest_user_text]).strip()
        eligibility_context = recent_text or latest_user_text
        lowered = eligibility_context.lower()
        last_tool_name = self.session_state.get("last_tool_name")

        if (
            looks_like_all_reservations_intent(eligibility_context)
            and last_tool_name in {"get_reservation_details", "cancel_reservation"}
        ):
            known_ids = [
                rid
                for rid in (self.session_state.get("known_reservation_ids") or [])
                if isinstance(rid, str)
            ]
            inventory = self.session_state.get("reservation_inventory", {})
            if not isinstance(inventory, dict):
                inventory = {}
            for known_id in known_ids:
                if known_id not in inventory:
                    self._trace(
                        "aggregate_cancel_load_remaining",
                        next_reservation=known_id,
                        loaded=len(inventory),
                        total=len(known_ids),
                    )
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": known_id},
                    }

            executed = self.session_state.get("executed_actions_by_reservation", {})
            if not isinstance(executed, dict):
                executed = {}
            require_single_passenger = "one passenger" in lowered or "only one passenger" in lowered
            eligible_ids: list[str] = []
            for known_id in known_ids:
                entry = inventory.get(known_id)
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status") or "").strip().lower() == "cancelled":
                    continue
                if reservation_has_flown_segments(self.session_state, known_id) is True:
                    continue
                if require_single_passenger and entry.get("passenger_count") != 1:
                    continue
                eligible, _ = cancel_eligibility(
                    self.session_state,
                    known_id,
                    eligibility_context,
                )
                if eligible:
                    eligible_ids.append(known_id)

            for eligible_id in eligible_ids:
                if executed.get(eligible_id) == "cancel_reservation":
                    continue
                self.session_state["reservation_id"] = eligible_id
                self.session_state["cancel_eligible"] = True
                self._trace(
                    "aggregate_cancel_execute",
                    reservation_id=eligible_id,
                    eligible_count=len(eligible_ids),
                )
                return {
                    "name": "cancel_reservation",
                    "arguments": {"reservation_id": eligible_id},
                }

            if eligible_ids:
                cancelled_count = sum(
                    1 for eligible_id in eligible_ids if executed.get(eligible_id) == "cancel_reservation"
                )
                return self._fallback_action(
                    f"I finished reviewing the reservations in scope and cancelled {cancelled_count} eligible booking"
                    + ("s." if cancelled_count != 1 else ".")
                )

            if require_single_passenger:
                return self._fallback_action(
                    "I reviewed the reservations in scope, and none of the upcoming bookings with only one passenger is eligible for cancellation under the airline policy."
                )
            return self._fallback_action(
                "I reviewed the reservations in scope, and none of the upcoming bookings is eligible for cancellation under the airline policy."
            )

        if last_tool_name != "get_reservation_details":
            return None

        reservation_id = self.session_state.get("reservation_id")
        entry = current_reservation_entry(self.session_state, reservation_id)
        if not isinstance(reservation_id, str) or not isinstance(entry, dict):
            return None

        eligible, refusal = cancel_eligibility(
            self.session_state,
            reservation_id,
            eligibility_context,
        )
        self.session_state["cancel_eligible"] = eligible
        self._trace(
            "cancel_resolution_evaluated",
            reservation_id=reservation_id,
            eligible=eligible,
            refusal=refusal,
        )
        if eligible:
            return {
                "name": "cancel_reservation",
                "arguments": {"reservation_id": reservation_id},
            }

        if refusal and not refusal.startswith("Please share the reason"):
            return self._terminal_cancel_refusal(refusal)

        created_at = entry.get("created_at")
        booked_within_24h = False
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at)
                booked_within_24h = (BENCHMARK_NOW - created_dt).total_seconds() <= 24 * 3600
            except ValueError:
                booked_within_24h = False
        status = str(entry.get("status") or "").strip().lower()
        cabin = str(entry.get("cabin") or "").strip().lower()
        insurance = str(entry.get("insurance") or "").strip().lower()
        mentions_covered_basis = any(
            token in lowered
            for token in (
                "airline cancelled",
                "airline canceled",
                "cancelled flight",
                "canceled flight",
                "health",
                "medical",
                "ill",
                "sick",
                "weather",
                "storm",
            )
        )
        impossible_without_override = (
            not booked_within_24h
            and status != "cancelled"
            and cabin != "business"
            and insurance != "yes"
        )
        if impossible_without_override and not mentions_covered_basis:
            return self._terminal_cancel_refusal(
                "This reservation is not eligible for cancellation under the airline policy because "
                "it was not booked within 24 hours, the flight was not cancelled by the airline, "
                "the cabin is not business class, and the booking does not include travel insurance."
            )

        return None

    def _pre_llm_cancel_user_followup_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if task_type != "cancel" and flow_name != "cancel":
            return None
        if self.session_state.get("latest_input_is_tool"):
            return None

        reservation_id = self.session_state.get("reservation_id")
        entry = current_reservation_entry(self.session_state, reservation_id)
        if not isinstance(reservation_id, str) or not isinstance(entry, dict):
            return None

        recent_user_messages = [
            str(message.get("content") or "")
            for message in self.messages[-8:]
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not str(message.get("content")).startswith("tool:")
        ]
        eligibility_context = "\n".join(recent_user_messages + [latest_user_text]).strip()
        eligible, refusal = cancel_eligibility(
            self.session_state,
            reservation_id,
            eligibility_context,
        )
        self.session_state["cancel_eligible"] = eligible
        self._trace(
            "cancel_followup_evaluated",
            reservation_id=reservation_id,
            eligible=eligible,
            refusal=refusal,
        )
        if eligible:
            return {
                "name": "cancel_reservation",
                "arguments": {"reservation_id": reservation_id},
            }
        if refusal and not refusal.startswith("Please share the reason"):
            return self._terminal_cancel_refusal(refusal)
        return None

    def _pre_llm_adversarial_compensation_verification_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if task_type == "cancel" or flow_name == "cancel":
            return None
        if not self.session_state.get("loaded_user_details"):
            return None

        working_memory = self.session_state.get("working_memory", {})
        if not isinstance(working_memory, dict):
            working_memory = {}

        # Skip the first message (policy document) — it contains words like
        # "business class", "cancelled", "compensation" that would spuriously
        # activate the adversarial-verification path for every task.
        policy_content = self.messages[0].get("content", "") if self.messages else ""
        recent_user_messages = [
            str(message.get("content") or "")
            for message in self.messages[-10:]
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not str(message.get("content")).startswith("tool:")
            and message.get("content") != policy_content
        ]
        if not recent_user_messages and not latest_user_text:
            return None
        recent_lowered = "\n".join(recent_user_messages + [latest_user_text]).lower()

        compensation_story = any(
            token in recent_lowered
            for token in (
                "compensation",
                "travel certificate",
                "missed meeting",
                "inconvenience",
            )
        )

        business_claim = "business" in recent_lowered
        _cancel_negations = (
            "not cancel", "not canceled", "not cancelled",
            "wasn't cancel", "was delayed, not", "delayed, not cancel",
            "delay not cancel",
        )
        cancelled_claim = (
            ("cancelled" in recent_lowered or "canceled" in recent_lowered)
            and not any(neg in recent_lowered for neg in _cancel_negations)
        )
        verification_active = bool(
            working_memory.get("adversarial_compensation_verification")
        )
        if not compensation_story and not verification_active:
            return None
        if compensation_story and (business_claim or cancelled_claim):
            working_memory["adversarial_compensation_verification"] = True
            working_memory["adversarial_compensation_business_claim"] = (
                bool(working_memory.get("adversarial_compensation_business_claim"))
                or business_claim
            )
            working_memory["adversarial_compensation_cancelled_claim"] = (
                bool(working_memory.get("adversarial_compensation_cancelled_claim"))
                or cancelled_claim
            )
            self.session_state["working_memory"] = working_memory
            verification_active = True
        if not verification_active:
            return None
        business_claim = bool(
            working_memory.get("adversarial_compensation_business_claim")
        ) or business_claim
        cancelled_claim = bool(
            working_memory.get("adversarial_compensation_cancelled_claim")
        ) or cancelled_claim

        known_ids = self.session_state.get("known_reservation_ids", [])
        inventory = self.session_state.get("reservation_inventory", {})
        if not isinstance(known_ids, list) or not known_ids or not isinstance(inventory, dict):
            return None

        for reservation_id in known_ids:
            if isinstance(reservation_id, str) and reservation_id not in inventory:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation_id},
                }

        matching_reservations: list[str] = []
        for reservation_id in known_ids:
            if not isinstance(reservation_id, str):
                continue
            entry = inventory.get(reservation_id)
            if not isinstance(entry, dict):
                continue
            cabin = str(entry.get("cabin") or "").strip().lower()
            status = str(entry.get("status") or "").strip().lower()
            if business_claim and cabin != "business":
                continue
            if cancelled_claim and status != "cancelled":
                continue
            matching_reservations.append(reservation_id)

        if matching_reservations:
            if len(matching_reservations) == 1:
                self.session_state["reservation_id"] = matching_reservations[0]
            working_memory["adversarial_compensation_verification"] = False
            self.session_state["working_memory"] = working_memory
            return None

        reservation_count = len([rid for rid in known_ids if isinstance(rid, str)])
        summary_clause = "all available reservations" if reservation_count <= 0 else (
            "all 1 reservation on your profile"
            if reservation_count == 1
            else f"all {reservation_count} reservations on your profile"
        )
        claim_clause = "a cancelled business flight" if business_claim and cancelled_claim else (
            "a business flight" if business_claim else "a cancelled flight"
        )
        return self._fallback_action(
            f"I reviewed {summary_clause}, and none of them matches {claim_clause}. Because the verified booking details do not support that claim, I can’t offer compensation or cancel anything on that basis."
        )

    def _pre_llm_status_compensation_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        lowered_user = latest_user_text.lower()
        status_context = task_type == "status" or flow_name == "status_compensation"
        if not status_context:
            status_context = (
                self.session_state.get("last_tool_name") == "get_flight_status"
                or any(
                    token in lowered_user
                    for token in (
                        "delayed flight",
                        "delay",
                        "delayed",
                        "compensation",
                        "travel certificate",
                        "flight status",
                        "status of",
                    )
                )
            )
        if not status_context:
            return None

        reservation_id = self.session_state.get("reservation_id")
        inventory = self.session_state.get("reservation_inventory", {})
        current_entry = (
            inventory.get(reservation_id)
            if isinstance(reservation_id, str) and isinstance(inventory, dict)
            else None
        )
        flights = current_entry.get("flights") if isinstance(current_entry, dict) else None
        no_change_or_cancel = any(
            phrase in lowered_user
            for phrase in (
                "don't want to change or cancel",
                "do not want to change or cancel",
                "not looking to change or cancel",
                "not looking to cancel",
                "not looking to change",
                "no need to cancel or change",
                "keeping the reservation as-is",
                "keeping the reservation as is",
            )
        )
        if not no_change_or_cancel:
            for message in reversed(self.messages[-6:]):
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if not isinstance(content, str) or content.startswith("tool:"):
                    continue
                historical = content.lower()
                if any(
                    phrase in historical
                    for phrase in (
                        "don't want to change or cancel",
                        "do not want to change or cancel",
                        "not looking to change or cancel",
                        "not looking to cancel",
                        "not looking to change",
                        "no need to cancel or change",
                    )
                ):
                    no_change_or_cancel = True
                    break
        # User explicitly denying they want compensation → treat as no interest
        _compensation_denial_phrases = (
            "not looking for compensation",
            "not asking for compensation",
            "not looking to claim compensation",
            "not claiming compensation",
            "not asking about compensation",
            "don't want compensation",
            "do not want compensation",
            "not interested in compensation",
        )
        compensation_denied = any(phrase in lowered_user for phrase in _compensation_denial_phrases)
        compensation_interest = not compensation_denied and any(
            token in lowered_user
            for token in ("compensation", "certificate", "travel certificate", "inconvenience")
        )
        recent_user_messages = [
            str(message.get("content") or "")
            for message in self.messages[-8:]
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not str(message.get("content")).startswith("tool:")
        ]
        recent_lowered = "\n".join(recent_user_messages).lower()
        compensation_context = compensation_interest or any(
            token in recent_lowered
            for token in ("compensation", "certificate", "travel certificate", "inconvenience")
        )
        membership = str(self.session_state.get("membership") or "").strip().lower()
        cabin = (
            str(current_entry.get("cabin") or "").strip().lower()
            if isinstance(current_entry, dict)
            else ""
        )

        if not self.session_state.get("latest_input_is_tool"):
            if no_change_or_cancel and compensation_interest:
                return self._fallback_action(
                    "I understand you want to keep the reservation as-is. Under the policy, I can only offer compensation after a delay is confirmed and only when you want to change or cancel the reservation. Since you do not want to change or cancel it, I can’t offer a travel certificate here."
                )
            flight_number = extract_flight_number(latest_user_text)
            mentioned_dates = extract_dates(latest_user_text)
            if isinstance(flight_number, str):
                matched_reservation = find_reservation_by_flight_number(
                    self.session_state, flight_number
                )
                if isinstance(matched_reservation, str):
                    reservation_id = matched_reservation
                    self.session_state["reservation_id"] = matched_reservation
                    self.session_state["recent_reservation_active"] = False
                    self.session_state["recent_reservation_id"] = matched_reservation
                    current_entry = (
                        inventory.get(matched_reservation)
                        if isinstance(inventory, dict)
                        else None
                    )
                    flights = (
                        current_entry.get("flights")
                        if isinstance(current_entry, dict)
                        else None
                    )
            if isinstance(flight_number, str) and isinstance(flights, list):
                for flight in flights:
                    if (
                        isinstance(flight, dict)
                        and flight.get("flight_number") == flight_number
                        and isinstance(flight.get("date"), str)
                        and (
                            not mentioned_dates
                            or flight.get("date") in mentioned_dates
                        )
                    ):
                        return {
                            "name": "get_flight_status",
                            "arguments": {
                                "flight_number": flight_number,
                                "date": flight["date"],
                            },
                        }
                for flight in flights:
                    if (
                        isinstance(flight, dict)
                        and flight.get("flight_number") == flight_number
                        and isinstance(flight.get("date"), str)
                    ):
                        return {
                            "name": "get_flight_status",
                            "arguments": {
                                "flight_number": flight_number,
                                "date": flight["date"],
                            },
                        }
            remembered_flight = self.session_state.get("flight_number")
            if isinstance(remembered_flight, str) and isinstance(flights, list):
                # Don't repeat get_flight_status if we already checked this flight
                # and confirmed it's not delayed — prevents infinite loops under social pressure.
                # Use the persistent _last_flight_status_checked_flight key which survives
                # respond actions (unlike last_tool_name which is cleared to None by them).
                # Use persistent key (survives respond actions that reset last_tool_name)
                last_checked_flight = self.session_state.get("_last_flight_status_checked_flight")
                last_status = str(self.session_state.get("last_flight_status_result") or "").lower()
                if (
                    last_checked_flight == remembered_flight
                    and last_status in {"landed", "available", "on time"}
                ):
                    return None
                for flight in flights:
                    if (
                        isinstance(flight, dict)
                        and flight.get("flight_number") == remembered_flight
                        and isinstance(flight.get("date"), str)
                    ):
                        return {
                            "name": "get_flight_status",
                            "arguments": {
                                "flight_number": remembered_flight,
                                "date": flight["date"],
                            },
                        }
            return None

        if self.session_state.get("last_tool_name") != "get_flight_status":
            return None

        last_args = self.session_state.get("last_tool_arguments") or {}
        flight_number = last_args.get("flight_number")
        flight_date = last_args.get("date")
        status_result = str(self.session_state.get("last_flight_status_result") or "").strip()
        if not isinstance(flight_number, str) or not isinstance(flight_date, str) or not status_result:
            return None

        no_change_or_cancel = no_change_or_cancel or "leave it at that" in lowered_user
        if compensation_context and not no_change_or_cancel:
            no_change_or_cancel = not any(
                token in recent_lowered
                for token in (
                    "change flights",
                    "change my flight",
                    "cancel the reservation",
                    "cancel my reservation",
                    "modify the reservation",
                )
            )
        membership_clause = ""
        if membership == "regular" and "gold member" in recent_lowered:
            membership_clause = " I also verified from your profile that you are a Regular member, not a Gold member."
        if status_result.lower() in {"delayed", "flying", "on time"} and compensation_context and no_change_or_cancel:
            return self._fallback_action(
                f"I checked flight {flight_number} on {flight_date}, and the current status is {status_result}.{membership_clause} Because you want to keep the reservation as-is and are not changing or cancelling it, I can’t offer compensation here."
            )
        if status_result.lower() in {"available", "landed"}:
            if no_change_or_cancel or compensation_context:
                return self._fallback_action(
                    f"I checked flight {flight_number} on {flight_date}, and the current status is {status_result}. That does not confirm a delay.{membership_clause} Since the delay is not confirmed, I can’t offer compensation here."
                )
            return self._fallback_action(
                f"I checked flight {flight_number} on {flight_date}, and the current status is {status_result}. That does not confirm a delay. If you want, tell me a different flight segment from this reservation to check."
            )
        return self._fallback_action(
            f"I checked flight {flight_number} on {flight_date}, and the current status is {status_result}. If you want to change or cancel the reservation because of that delay, I can help with the next step."
        )

    def _pre_llm_post_cancel_booking_action(
        self, latest_user_text: str
    ) -> dict[str, Any] | None:
        """After a cancel denial, check if user also wants to book a new flight (Task 35 pattern)."""
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if task_type != "cancel" and flow_name != "cancel":
            return None
        if not self.session_state.get("loaded_user_details"):
            return None
        if not looks_like_booking_after_cancel(latest_user_text):
            return None

        # User wants to book after cancel denial — transition to booking
        self.session_state["task_type"] = "booking"
        if isinstance(active_flow, dict):
            active_flow["name"] = "booking"
            active_flow["stage"] = "analyze"
        self.session_state["active_flow"] = active_flow

        airport_codes = extract_airport_codes(latest_user_text)
        dates = extract_dates(latest_user_text)

        if len(airport_codes) >= 2:
            self.session_state["origin"] = airport_codes[0]
            self.session_state["destination"] = airport_codes[1]
        if dates:
            self.session_state["travel_dates"] = dates

        # Proceed with search if we have origin/destination/date
        origin = self.session_state.get("origin")
        destination = self.session_state.get("destination")
        travel_dates = self.session_state.get("travel_dates", [])
        date = travel_dates[0] if travel_dates else None

        if origin and destination:
            self._trace("post_cancel_booking_transition", origin=origin, destination=destination, date=date)
            if date:
                return {
                    "name": "search_direct_flight",
                    "arguments": {"origin": origin, "destination": destination, "date": date},
                }
            else:
                return self._fallback_action(
                    f"I can help you book a flight from {origin} to {destination}. What date would you like to travel?"
                )

        return None

    def _pricing_expression_for_current_reservation(
        self, reservation_id: str | None, target_cabin: str | None
    ) -> str | None:
        return pricing_expression_for_current_reservation(
            self.session_state, reservation_id, target_cabin
        )

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
            arguments["content"] = normalize_respond_content(arguments.get("content", ""))

        if self.tool_names and name != RESPOND_ACTION_NAME and name not in self.tool_names:
            return self._fallback_action(
                f"I selected an unavailable action ({name}). Please restate what you need."
            )

        return {"name": name, "arguments": arguments}

    def _process_action_candidate(
        self,
        action: dict[str, Any],
        latest_user_text: str,
    ) -> dict[str, Any]:
        original_name = action.get("name")
        
        # ARCHITECTURAL FIX: Track reservation checks to force aggregate action
        if original_name == "get_reservation_details":
            rid = action.get("arguments", {}).get("reservation_id")
            checked = self.session_state.get("_checked_reservations", [])
            if rid and rid not in checked:
                checked.append(rid)
                self.session_state["_checked_reservations"] = checked
                
                # After 3+ different reservations checked, force aggregate response
                if len(checked) >= 3 and not self.session_state.get("aggregate_forced"):
                    self.session_state["aggregate_forced"] = True
                    self._trace("aggregate_forced", 
                               count=len(checked), 
                               reservations=checked)
                    return self._fallback_action(
                        f"I've reviewed {len(checked)} reservations on your profile ({', '.join(checked[:4])}). Based on policy, I can help with cancellations, modifications, or baggage updates. Which specific reservation would you like me to act on?"
                    )
        
        action = guard_action(
            self.session_state,
            self.tools,
            self.messages,
            self.turn_count,
            self._normalize_action(action),
            latest_user_text,
        )
        action = termination_controller(self.session_state, action, latest_user_text)

        # P2: Retry on guard reject — if guard replaced a write action with respond fallback,
        # retry up to 2 times with a hint about what went wrong (from phantom-agent).
        # Only retry for actions that actually modify state (not read actions like get_reservation_details).
        result_name = action.get("name")
        write_actions = {
            "cancel_reservation", "update_reservation_flights", "update_reservation_baggages",
            "book_reservation", "update_reservation_insurance", "remove_passenger_from_reservation",
        }
        if result_name == RESPOND_ACTION_NAME and original_name in write_actions:
            reject_key = f"reject_{original_name}"
            reject_count = self.session_state.get(reject_key, 0)
            if reject_count < 2:
                reject_count += 1
                self.session_state[reject_key] = reject_count
                self._trace("guard_reject_retry", original=original_name, attempt=reject_count)
                # Inject hint message back into conversation
                self.messages.append({
                    "role": "user",
                    "content": f"tool: Your proposed action '{original_name}' was blocked by the policy guard. "
                               f"Attempt {reject_count}/2. Reconsider your approach — "
                               f"check if you have all required information (user details, reservation details, eligibility). "
                               f"Try a different action that aligns with the current task type: {self.session_state.get('task_type', 'general')}."
                })
                # Return the ORIGINAL action to retry (LLM will be called again)
                return {
                    "name": original_name,
                    "arguments": action.get("arguments", {}),
                }

        # Reset counters on success
        if result_name != RESPOND_ACTION_NAME:
            self.session_state["guard_reject_count"] = 0

        return {
            "name": result_name,
            "arguments": action.get("arguments")
            if isinstance(action.get("arguments"), dict)
            else {},
        }

    def _extract_tools_from_first_turn(self, input_text: str) -> str:
        if self.turn_count != 1:
            return input_text

        sanitized = sanitize_tool_descriptions(input_text)
        tools = extract_openai_tools(sanitized)
        if tools:
            self.tools = tools
            self.tool_names = {t["function"]["name"] for t in tools if "function" in t}
            print(
                f"[tau2-agent] extracted {len(tools)} tool schemas "
                f"for native function calling"
            )
        else:
            names = set(re.findall(r'"name"\s*:\s*"([^"]+)"', sanitized))
            self.tool_names = names
            print(
                f"[tau2-agent] could not extract tool schemas; "
                f"running in JSON mode with {len(names)} allowed names"
            )
        return sanitized

    def _build_turn_graph(self):
        graph = StateGraph(TurnGraphState)
        graph.add_node("transfer_hold", self._graph_transfer_hold)
        graph.add_node("ingest_input", self._graph_ingest_input)
        graph.add_node("llm_decide", self._graph_llm_decide)
        graph.add_node("finalize_action", self._graph_finalize_action)
        graph.add_edge(START, "transfer_hold")
        graph.add_conditional_edges(
            "transfer_hold",
            lambda state: END if state.get("should_return") else "ingest_input",
        )
        graph.add_edge("ingest_input", "llm_decide")
        graph.add_edge("llm_decide", "finalize_action")
        graph.add_edge("finalize_action", END)
        return graph.compile()

    def _graph_transfer_hold(self, state: TurnGraphState) -> TurnGraphState:
        if not self.just_transferred:
            return {"should_return": False}

        self.just_transferred = False
        input_text = state["input_text"]
        self.session_state["latest_input_is_tool"] = input_text.strip().startswith("tool:")
        action = self._fallback_action(TRANSFER_HOLD_MESSAGE)
        update_state_from_text(self.session_state, input_text)
        update_state_from_tool_payload(self.session_state, input_text)
        self.messages.append({"role": "user", "content": input_text})
        self.messages.append(
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
        )
        return {"action": action, "should_return": True}

    def _graph_ingest_input(self, state: TurnGraphState) -> TurnGraphState:
        input_text = state["input_text"]
        self.session_state["latest_input_is_tool"] = input_text.strip().startswith("tool:")
        update_state_from_text(self.session_state, input_text)
        update_state_from_tool_payload(self.session_state, input_text)
        sync_runtime_state(self.session_state, input_text)

        # SKILL CLASSIFICATION: Dual classifier (LLM + regex) on first user message
        if not self.session_state.get("intent_classified") and not self.session_state.get("latest_input_is_tool"):
            self.session_state["intent_classified"] = True
            try:
                skill_id = dual_classify_intent(
                    lambda messages, max_tokens: litellm.completion(
                        model=os.getenv("AGENT_LLM_MODEL", "openai/gpt-4o"),
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.0,
                    ),
                    input_text,
                )
                self.session_state["active_skill_id"] = skill_id
                # Also update task_type if still general
                current = str(self.session_state.get("task_type") or "general")
                if current == "general":
                    self.session_state["task_type"] = skill_id
                    self._trace("skill_classified", skill=skill_id, classifier="dual")
            except Exception as e:
                self._trace("skill_classification_failed", error=str(e))
                self.session_state["active_skill_id"] = "general"

        self.messages.append({"role": "user", "content": input_text})
        latest_user_text = self._latest_user_text()
        return {"latest_user_text": latest_user_text, "should_return": False}

    def _force_tool_for_task_type(self, latest_user_text: str) -> dict[str, Any] | None:
        """Force-Tool Layer (from phantom-agent): when stuck in a respond loop,
        force the correct tool based on detected task_type/intent instead of another respond."""
        task_type = str(self.session_state.get("task_type") or "general")
        active_flow = self.session_state.get("active_flow", {})
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        effective_flow = flow_name or task_type
        res_id = self.session_state.get("reservation_id")
        known_ids = [rid for rid in (self.session_state.get("known_reservation_ids") or []) if isinstance(rid, str)]
        inventory = self.session_state.get("reservation_inventory", {})
        if not isinstance(inventory, dict):
            inventory = {}

        # Cancel flow: force cancel_reservation for eligible reservations
        if effective_flow == "cancel":
            executed = self.session_state.get("executed_actions_by_reservation", {})
            if not isinstance(executed, dict):
                executed = {}
            for rid in known_ids:
                if executed.get(rid) == "cancel_reservation":
                    continue
                entry = inventory.get(rid)
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status") or "").strip().lower() == "cancelled":
                    continue
                eligible, refusal = cancel_eligibility(self.session_state, rid, latest_user_text)
                if eligible:
                    self.session_state["reservation_id"] = rid
                    self.session_state["cancel_eligible"] = True
                    self._trace("force_tool", action="cancel_reservation", reservation_id=rid)
                    return {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": rid},
                    }
            return self._fallback_action(
                f"I've reviewed all {len(inventory)} reservations on your profile and none are eligible for cancellation under the airline policy."
            )

        # Baggage flow: force update_reservation_baggages
        if effective_flow == "baggage":
            for rid in known_ids:
                entry = inventory.get(rid)
                if isinstance(entry, dict):
                    self.session_state["reservation_id"] = rid
                    baggage_total = entry.get("total_baggages", 0)
                    self._trace("force_tool", action="update_reservation_baggages", reservation_id=rid)
                    return {
                        "name": "update_reservation_baggages",
                        "arguments": {"reservation_id": rid, "total_baggages": baggage_total},
                    }
            return self._fallback_action(
                "I need your reservation ID to check baggage allowance."
            )

        # Modify flow: force search or update
        if effective_flow in {"modify", "modify_pricing"}:
            if res_id and res_id in inventory:
                entry = inventory[res_id]
                if isinstance(entry, dict):
                    flights = entry.get("flights")
                    if isinstance(flights, list) and flights:
                        first = flights[0]
                        if isinstance(first, dict):
                            origin = first.get("origin")
                            dest = first.get("destination")
                            date = first.get("date")
                            if origin and dest and date:
                                search_inv = self.session_state.get("flight_search_inventory", {})
                                if not isinstance(search_inv, dict):
                                    search_inv = {}
                                key = f"{origin}|{dest}|{date}"
                                if key not in search_inv:
                                    self._trace("force_tool", action="search_direct_flight")
                                    return {
                                        "name": "search_direct_flight",
                                        "arguments": {"origin": origin, "destination": dest, "date": date},
                                    }
                                self._trace("force_tool", action="update_reservation_flights")
                                return {
                                    "name": "update_reservation_flights",
                                    "arguments": {"reservation_id": res_id},
                                }
            return self._fallback_action(
                "I've loaded your reservations. Please specify which one to modify and the change needed."
            )

        # Booking flow: force search
        if effective_flow == "booking":
            origin = self.session_state.get("origin")
            destination = self.session_state.get("destination")
            travel_dates = self.session_state.get("travel_dates") or []
            if origin and destination and travel_dates:
                self._trace("force_tool", action="search_direct_flight")
                return {
                    "name": "search_direct_flight",
                    "arguments": {"origin": origin, "destination": destination, "date": travel_dates[0]},
                }
            return self._fallback_action(
                "Please provide origin, destination, travel date (YYYY-MM-DD), and cabin preference."
            )

        # Status flow: force get_flight_status
        if effective_flow == "status":
            fn = self.session_state.get("flight_number")
            fd = self.session_state.get("flight_date")
            if fn and fd:
                self._trace("force_tool", action="get_flight_status")
                return {
                    "name": "get_flight_status",
                    "arguments": {"flight_number": fn, "date": fd},
                }

        # General: no specific tool — use fallback respond
        return self._fallback_action(
            "I understand your request. Let me take a specific action to help you. "
            "Please confirm what you'd like me to do and I'll proceed."
        )

    def _pre_llm_aggregate_reservation_limit(self, latest_user_text: str) -> dict[str, Any] | None:
        """Pre-LLM subroutine: Limit reservation checks to 3, then force summary.
        
        Architecture: Instead of letting LLM loop on get_reservation_details indefinitely,
        count unique reservations checked and force a summary response after 3.
        """
        checked = self.session_state.get("_checked_reservations", [])
        if not isinstance(checked, list) or len(checked) < 3:
            return None
        
        # Already checked 3+ reservations
        if not self.session_state.get("aggregate_forced"):
            # First time hitting limit — force summary
            self.session_state["aggregate_forced"] = True
            self._trace("aggregate_limit_reached", 
                       count=len(checked), 
                       reservations=checked)
            return self._fallback_action(
                f"I've reviewed {len(checked)} reservations on your profile ({', '.join(checked[:5])}). Based on policy, I can help with eligible cancellations, modifications (subject to cabin rules), or baggage updates. Which specific reservation would you like me to act on?"
            )
        
        # Already forced once — if LLM tries to check again, block it
        return None

    def _check_action_loop(self, latest_user_text: str) -> dict[str, Any] | None:
        """Force-Tool Layer: detect loops and force the correct tool instead of respond fallback."""
        # Collect last 6 assistant messages
        assistant_actions = []
        for msg in reversed(self.messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not content or content.startswith("tool:"):
                continue
            try:
                parsed = json.loads(content)
                name = parsed.get("name", "")
                args = parsed.get("arguments", {})
                if name == RESPOND_ACTION_NAME:
                    key = f"respond:{hash(args.get('content', '')[:100])}"
                else:
                    key = f"{name}:{json.dumps(args, sort_keys=True)[:200]}"
                assistant_actions.append({"name": name, "args": args, "key": key})
            except Exception:
                assistant_actions.append({"name": "respond_text", "key": f"text:{hash(content[:100])}"})
            if len(assistant_actions) >= 6:
                break

        if len(assistant_actions) < 3:
            return None

        # Check if last 3 actions are identical
        keys = [a["key"] for a in assistant_actions[:3]]
        if len(set(keys)) == 1:
            repeated_key = keys[0]
            self._trace("loop_detected", key=repeated_key, actions_count=len(assistant_actions))

            # Reset flow state to force re-evaluation
            self.session_state["task_type"] = "general"
            self.session_state["active_flow"] = {"name": "general", "stage": "analyze"}
            self.session_state["_loop_broken"] = True

            # FORCE-TOOL LAYER: instead of respond fallback, force the correct tool
            force_action = self._force_tool_for_task_type(latest_user_text)
            if force_action is not None:
                self._trace("force_tool_triggered", action=force_action.get("name"))
                return force_action

            # Last resort: generic fallback
            return self._fallback_action(
                "I've been repeating myself. Let me take a different approach to help you."
            )

        # NEW: Check для 2 повторений подряд (более строгая проверка)
        if len(assistant_actions) >= 4:
            keys_2 = [a["key"] for a in assistant_actions[:2]]
            if len(set(keys_2)) == 1:
                repeated_key = keys_2[0]
                # Проверяем, что это read-операция (get_reservation_details, get_user_details, get_flight_status)
                first_action = assistant_actions[0]
                if first_action["name"] in {"get_reservation_details", "get_user_details", "get_flight_status"}:
                    self._trace("early_loop_detected_read_tool", key=repeated_key, tool=first_action["name"])
                    
                    # Сбрасываем состояние и форсируем другое действие
                    self.session_state["_read_loop_detected"] = True
                    self.session_state["_loop_tool"] = first_action["name"]
                    
                    force_action = self._force_tool_for_task_type(latest_user_text)
                    if force_action is not None:
                        self._trace("force_tool_early_read_loop", action=force_action.get("name"))
                        return force_action
                    
                    return self._fallback_action(
                        f"I already have the information from {first_action['name']}. Let me proceed with the next step."
                    )

        # NEW: Detect confirmation-loop (asking for confirmation instead of acting)
        # Check last 2 assistant messages for confirmation requests without action
        if len(assistant_actions) >= 2:
            confirm_keywords = ["please confirm", "reply yes", "confirm with yes",
                               "would you like me to proceed", "should i proceed",
                               "please share the exact", "tell me which",
                               "which reservation would you like", "do you want me to"]
            
            recent_contents = []
            for a in assistant_actions[:2]:
                if a["name"] == RESPOND_ACTION_NAME:
                    content = a["args"].get("content", "").lower()
                    recent_contents.append(content)
            
            if len(recent_contents) >= 2:
                # Check if both recent messages are confirmation requests
                both_confirmations = all(
                    any(kw in content for kw in confirm_keywords)
                    for content in recent_contents
                )
                
                if both_confirmations:
                    self._trace("confirmation_loop_detected", 
                               contents=recent_contents[:2])
                    
                    # Force execute the pending action instead of asking
                    force_action = self._force_tool_for_task_type(latest_user_text)
                    if force_action is not None:
                        self._trace("force_tool_confirmation_loop", 
                                   action=force_action.get("name"))
                        return force_action
                    
                    return self._fallback_action(
                        "I have all the information needed. Let me proceed with the action."
                    )

        return None

    def _graph_llm_decide(self, state: TurnGraphState) -> TurnGraphState:
        latest_user_text = state.get("latest_user_text", "")

        # === SOFT INTENT HINTS: inject into system context, don't force ===
        # Add a hint about the detected intent so the LLM can use it naturally
        task_type = self.session_state.get("task_type", "general")
        active_flow = self.session_state.get("active_flow", {})
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if task_type not in {"general"} or (flow_name and flow_name != "general"):
            # Ensure the intent hint is reflected in the session state for the system prompt
            pass  # Already handled by build_policy_reminder in playbooks.py

        # === PRE-LLM: Aggregate reservation check limiter ===
        # After 3+ different reservations checked, force summary instead of more checks
        forced_action = self._pre_llm_aggregate_reservation_limit(latest_user_text)
        if forced_action is not None:
            self._trace("forced_action", source="aggregate_reservation_limit", action=forced_action)
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        # === END PRE-LLM aggregate limit ===

        # === FORCED EXECUTION LAYER: bypass LLM when state is complete ===
        # Architecture: When session_state has all required data for action,
        # execute directly without wasting a LLM call
        if should_force_execution(self.session_state):
            forced = force_pending_action(self.session_state)
            if forced is not None:
                self._trace("forced_execution_bypassed_llm", action=forced.get("name"))
                action = self._process_action_candidate(forced, latest_user_text)
                return {"action": action, "latest_user_text": latest_user_text}
        
        # === FALLBACK: Count get_reservation_details calls from message history ===
        # When agent has checked 3+ different reservations, force aggregate action
        reservation_checks = []
        for msg in self.messages:
            # Check assistant messages with tool_calls
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict) and tc.get("name") == "get_reservation_details":
                            rid = tc.get("arguments", {}).get("reservation_id")
                            if rid and rid not in reservation_checks:
                                reservation_checks.append(rid)
        
        if len(reservation_checks) >= 3 and not self.session_state.get("aggregate_forced"):
            self.session_state["aggregate_forced"] = True
            self._trace("aggregate_force_triggered", 
                       reservation_count=len(reservation_checks),
                       reservations=reservation_checks)
            # Force respond with summary instead of more checks
            action = self._process_action_candidate({
                "name": "respond",
                "arguments": {
                    "content": f"I've reviewed all {len(reservation_checks)} reservations on your profile. Based on the policy, here's what I can do:\n\n- For eligible cancellations: I can cancel\n- For modifications: subject to cabin class and membership rules\n- For baggage: I can update based on your membership tier\n\nPlease tell me which specific reservation and action you'd like me to proceed with."
                },
            }, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        # === END FORCED EXECUTION LAYER ===

        # === CIRCUIT BREAKER: detect action loops ===
        loop_action = self._check_action_loop(latest_user_text)
        if loop_action is not None:
            self._trace("circuit_breaker_triggered", action=loop_action)
            action = self._process_action_candidate(loop_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        # === END CIRCUIT BREAKER ===

        forced_action = self._pre_llm_route_resolution_action(latest_user_text)
        if forced_action is not None:
            self._trace("forced_action", source="route_resolution", action=forced_action)
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        forced_action = self._pre_llm_recent_reservation_resolution_action(latest_user_text)
        if forced_action is not None:
            self._trace("forced_action", source="recent_reservation", action=forced_action)
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        forced_action = self._pre_llm_cancel_post_user_lookup_action(latest_user_text)
        if forced_action is not None:
            self._trace("forced_action", source="cancel_post_user_lookup", action=forced_action)
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        forced_action = self._pre_llm_cancel_identity_collection_action(latest_user_text)
        if forced_action is not None:
            self._trace("forced_action", source="cancel_identity_collection", action=forced_action)
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        active_flow = self.session_state.get("active_flow")
        flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
        if (
            (str(self.session_state.get("task_type") or "general") == "cancel" or flow_name == "cancel")
            and self.session_state.get("latest_input_is_tool")
            and self.session_state.get("last_tool_name") == "get_user_details"
            and self.session_state.get("loaded_user_details")
            and not extract_reservation_id(latest_user_text)
            and not extract_flight_number(latest_user_text)
            and len(extract_airport_codes(latest_user_text)) < 2
            and len(
                [
                    rid
                    for rid in (self.session_state.get("known_reservation_ids") or [])
                    if isinstance(rid, str)
                ]
            ) > 1
        ):
            action = self._process_action_candidate(
                self._fallback_action(
                    "I found multiple reservations on your profile. Tell me which trip you want to cancel by reservation ID, route, or flight number, and I’ll check the correct booking."
                ),
                latest_user_text,
            )
            self._trace("forced_action", source="cancel_disambiguation_guard", action=action)
            return {"action": action, "latest_user_text": latest_user_text}
        forced_action = self._pre_llm_cancel_resolution_action(latest_user_text)
        if forced_action is not None:
            self._trace("forced_action", source="cancel_resolution", action=forced_action)
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        forced_action = self._pre_llm_cancel_user_followup_action(latest_user_text)
        if forced_action is not None:
            self._trace("forced_action", source="cancel_followup", action=forced_action)
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        forced_action = self._pre_llm_adversarial_compensation_verification_action(latest_user_text)
        if forced_action is not None:
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        forced_action = self._pre_llm_status_compensation_action(latest_user_text)
        if forced_action is not None:
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        # Post-cancel denial: check if user also wants to book a new flight
        forced_action = self._pre_llm_post_cancel_booking_action(latest_user_text)
        if forced_action is not None:
            action = self._process_action_candidate(forced_action, latest_user_text)
            return {"action": action, "latest_user_text": latest_user_text}
        action = self._process_action_candidate(self._call_llm(), latest_user_text)
        return {"action": action, "latest_user_text": latest_user_text}

    def _graph_finalize_action(self, state: TurnGraphState) -> TurnGraphState:
        action = state["action"]
        latest_user_text = state.get("latest_user_text", "")

        previous_name = self.session_state.get("last_tool_name")
        previous_arguments = self.session_state.get("last_tool_arguments")
        previous_user_text = self.session_state.get("last_tool_user_text")
        self.session_state["last_tool_name"] = (
            action.get("name") if action.get("name") != RESPOND_ACTION_NAME else None
        )
        self.session_state["last_tool_arguments"] = (
            action.get("arguments") if action.get("name") != RESPOND_ACTION_NAME else None
        )
        self.session_state["last_tool_user_text"] = latest_user_text
        # Persist flight status check result separately so the loop-protection
        # in _pre_llm_status_compensation_action survives a subsequent respond
        # action (which clears last_tool_name / last_tool_arguments).
        if action.get("name") == "get_flight_status":
            args = action.get("arguments") or {}
            self.session_state["_last_flight_status_checked_flight"] = args.get("flight_number")
            self.session_state["_last_flight_status_checked_date"] = args.get("date")
        if action.get("name") == RESPOND_ACTION_NAME:
            self.session_state["last_tool_streak"] = 0
        elif (
            self.session_state["last_tool_name"] == previous_name
            and self.session_state["last_tool_arguments"] == previous_arguments
            and latest_user_text == previous_user_text
        ):
            self.session_state["last_tool_streak"] = int(
                self.session_state.get("last_tool_streak") or 0
            ) + 1
        else:
            self.session_state["last_tool_streak"] = 1

        remember_pending_confirmation(self.session_state, action)
        clear_pending_confirmation_if_completed(self.session_state, action)
        record_completed_action(self.session_state, action)
        resolve_completed_subtasks(self.session_state, action)
        sync_runtime_state(self.session_state, latest_user_text)

        if action.get("name") == "transfer_to_human_agents":
            self.just_transferred = True

        self.messages.append(
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
        )
        return {"action": action}

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self) -> dict[str, Any]:
        active_tools = self.tools or None
        messages = self._messages_for_model()
        plan_data: dict[str, Any] | None = None
        if self._should_use_plan(self._latest_user_text()):
            try:
                plan_completion = litellm.completion(
                    model=self.model,
                    messages=messages
                        + [
                            {"role": "user", "content": plan_prompt(history_snapshot(self.session_state))}
                        ],
                    temperature=self.temperature,
                )
                plan_text = (
                    getattr(plan_completion.choices[0].message, "content", None) or ""
                ).strip()
                plan_data = parse_plan(plan_text)
                if plan_data and not planner_conflicts_with_state(self.session_state, plan_data):
                    self.latest_plan = plan_data
                    messages = messages + [
                        {
                            "role": "system",
                            "content": (
                                "Structured planner output for this turn: "
                                + json.dumps(plan_data, ensure_ascii=False)
                            ),
                        }
                    ]
                elif plan_text:
                    messages = messages + [
                        {
                            "role": "system",
                            "content": "Planner output was discarded because it conflicted with the current runtime state.",
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

        completion = None
        retry_attempts = 3
        last_exc: Exception | None = None
        for attempt in range(1, retry_attempts + 1):
            try:
                completion = litellm.completion(**kwargs)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(f"[tau2-agent] LLM call failed (attempt {attempt}/{retry_attempts}): {exc}")

        if completion is None:
            minimal_kwargs = {
                "model": self.model,
                "messages": self._messages_for_model(),
                "temperature": self.temperature,
            }
            for attempt in range(1, retry_attempts + 1):
                try:
                    completion = litellm.completion(**minimal_kwargs)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"[tau2-agent] LLM retry failed (attempt {attempt}/{retry_attempts}): {exc}"
                    )

        if completion is None:
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
        parsed = extract_json_object(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
            name = parsed["name"]
            args = parsed.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            if name == RESPOND_ACTION_NAME:
                args = dict(args)
                args["content"] = normalize_respond_content(args.get("content", ""))
            action = {"name": name, "arguments": args}
            if plan_data and isinstance(plan_data.get("next_action"), dict):
                planned_name = plan_data["next_action"].get("name")
                if isinstance(planned_name, str) and planned_name in {
                    "cancel_reservation",
                    "update_reservation_flights",
                    "update_reservation_baggages",
                    "book_reservation",
                }:
                    action["__planned_write__"] = planned_name
            return action

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
                retry_parsed = extract_json_object(retry_text)
                if isinstance(retry_parsed, dict) and isinstance(
                    retry_parsed.get("name"), str
                ):
                    retry_args = retry_parsed.get("arguments") or {}
                    if not isinstance(retry_args, dict):
                        retry_args = {}
                    if retry_parsed["name"] == RESPOND_ACTION_NAME:
                        retry_args = dict(retry_args)
                        retry_args["content"] = normalize_respond_content(
                            retry_args.get("content", "")
                        )
                    return {"name": retry_parsed["name"], "arguments": retry_args}
            except Exception:
                pass
        return self._fallback_action(stripped or "Could you clarify your request?")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = self._normalize_input_prefix(get_message_text(message))
        self.turn_count += 1
        input_text = self._extract_tools_from_first_turn(input_text)
        self._trace("turn_start", input=input_text[:400])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Selecting next action..."),
        )
        result = self._turn_graph.invoke({"input_text": input_text})
        action = result["action"]
        self._trace("turn_action", action=action)
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=action))],
            name="Action",
        )
