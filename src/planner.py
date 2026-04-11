"""Structured planning pass for risky tau2 turns."""

from __future__ import annotations

import json
from typing import Any

RISKY_WRITE_ACTIONS = {
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "book_reservation",
}


def should_plan(state: dict[str, Any], latest_user_text: str, candidate_action: dict[str, Any] | None = None) -> bool:
    lowered = latest_user_text.lower()
    if int(state.get("error_streak") or 0) > 0:
        return True
    if state.get("terminal_user_intent") == "stop_without_changes":
        return True
    active_flow = state.get("active_flow") or {}
    flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
    if flow_name in {"reservation_triage", "status_compensation", "aggregate"}:
        return True
    if latest_user_text.strip().startswith("tool:"):
        if state.get("last_tool_name") in {"get_user_details", "get_reservation_details", "search_direct_flight", "search_onestop_flight", "calculate"}:
            return True
    if any(token in lowered for token in (" and ", " also ", "all upcoming", "include all", "all passengers", "other reservation", "keep it as-is")):
        return True
    if candidate_action and candidate_action.get("name") in RISKY_WRITE_ACTIONS:
        return True
    if state.get("pending_confirmation_action"):
        return True
    queue = state.get("subtask_queue")
    return isinstance(queue, list) and len(queue) > 1


def plan_prompt(runtime_brief: str) -> str:
    schema = {
        "intent": "short label",
        "flow_name": "active flow family",
        "flow_stage": "analyze|lookup|policy_check|write|finalize",
        "known_facts": ["..."],
        "missing_facts": ["..."],
        "policy_checks": ["..."],
        "subtasks": ["..."],
        "retry_risk": "none|low|high",
        "termination_mode": "continue|ask_user|summarize|finalize",
        "next_action": {"name": "tool_or_respond", "arguments": {}},
        "safe_fallback_action": {"name": "tool_or_respond", "arguments": {}},
    }
    return (
        "Produce a compact planning JSON for the next step.\n"
        "Use this runtime brief:\n"
        f"{runtime_brief}\n\n"
        "Return only a JSON object with this shape:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def parse_plan(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    next_action = payload.get("next_action")
    if not isinstance(next_action, dict):
        payload["next_action"] = {"name": "respond", "arguments": {}}
    return payload


def planner_conflicts_with_state(state: dict[str, Any], plan: dict[str, Any]) -> bool:
    next_action = plan.get("next_action")
    if not isinstance(next_action, dict):
        return True
    name = next_action.get("name")
    if not isinstance(name, str) or not name:
        return True
    active_flow = state.get("active_flow") or {}
    flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
    if int(state.get("error_streak") or 0) >= 1 and next_action == state.get("last_tool_arguments"):
        return True
    if name == "cancel_reservation" and flow_name not in {"cancel", "general"}:
        return True
    if name == "update_reservation_flights" and flow_name not in {"modify", "modify_pricing", "general"}:
        return True
    if name in {"search_direct_flight", "search_onestop_flight"} and state.get("reservation_locked"):
        return True
    if flow_name == "reservation_triage" and name in {"search_direct_flight", "search_onestop_flight", "calculate", "update_reservation_flights"}:
        return True
    if flow_name == "status_compensation" and name in {"search_direct_flight", "search_onestop_flight", "calculate"}:
        return True
    return False
