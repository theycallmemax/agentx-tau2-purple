"""Runtime state-machine helpers, subtask handling, and termination control."""

from __future__ import annotations

import json
from typing import Any

SUCCESSFUL_WRITE_TOOLS = {
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "book_reservation",
}


def classify_flow_name(state: dict[str, Any], latest_user_text: str) -> str:
    lowered = latest_user_text.lower()
    if any(phrase in lowered for phrase in ("all upcoming", "include all", "total cost of all", "all reservations")):
        return "aggregate"
    if state.get("task_type") == "status":
        if any(phrase in lowered for phrase in ("which reservation", "check reservation", "other reservation", "find my")):
            return "reservation_triage"
        return "status_compensation"
    if state.get("task_type") == "cancel" and isinstance(state.get("reservation_id"), str):
        return "cancel"
    if state.get("task_type") == "modify" and isinstance(state.get("reservation_id"), str):
        return "modify_pricing"
    if state.get("task_type") == "booking":
        return "booking"
    task_type = state.get("task_type")
    if isinstance(task_type, str) and task_type and task_type != "general":
        return task_type
    return "general"


def determine_flow_name(state: dict[str, Any]) -> str:
    task_type = state.get("task_type")
    if isinstance(task_type, str) and task_type and task_type != "general":
        return task_type
    return "general"


def decompose_subtasks(latest_user_text: str, flow_name: str) -> list[dict[str, str]]:
    lowered = latest_user_text.lower()
    subtasks: list[dict[str, str]] = []
    if any(phrase in lowered for phrase in ("all upcoming", "all my upcoming", "include all")):
        subtasks.append({"kind": "aggregate", "status": "pending"})
    if "cancel" in lowered:
        subtasks.append({"kind": "cancel", "status": "pending"})
    if any(phrase in lowered for phrase in ("keep the other", "keep it as-is", "keep it as is", "leave it as-is")):
        subtasks.append({"kind": "modify", "status": "pending"})
    if any(phrase in lowered for phrase in ("change", "modify", "upgrade", "downgrade", "nonstop", "direct")):
        subtasks.append({"kind": "modify", "status": "pending"})
    if "bag" in lowered:
        subtasks.append({"kind": "baggage", "status": "pending"})
    if not subtasks:
        subtasks.append({"kind": flow_name, "status": "pending"})
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for subtask in subtasks:
        kind = subtask["kind"]
        if kind in seen:
            continue
        seen.add(kind)
        deduped.append(subtask)
    return deduped


def sync_runtime_state(state: dict[str, Any], latest_user_text: str) -> None:
    flow_name = classify_flow_name(state, latest_user_text)
    active_flow = state.get("active_flow")
    if not isinstance(active_flow, dict):
        active_flow = {}
    active_flow.setdefault("name", flow_name)
    active_flow["name"] = flow_name
    active_flow.setdefault("stage", "analyze")
    if latest_user_text and not str(latest_user_text).strip().startswith("tool:"):
        state["subtask_queue"] = decompose_subtasks(latest_user_text, flow_name)
    if not isinstance(state.get("completed_subtasks"), list):
        state["completed_subtasks"] = []
    if not isinstance(state.get("completed_actions"), list):
        state["completed_actions"] = []
    state["active_flow"] = active_flow
    raw_facts = state.get("raw_facts")
    if not isinstance(raw_facts, dict):
        raw_facts = {}
    raw_facts.update(
        {
            "user_id": state.get("user_id"),
            "reservation_id": state.get("reservation_id"),
            "origin": state.get("origin"),
            "destination": state.get("destination"),
            "travel_dates": state.get("travel_dates", [])[:4],
        }
    )
    working = state.get("working_memory")
    if not isinstance(working, dict):
        working = {}
    working.update(
        {
            "active_flow": active_flow,
            "flow_version": state.get("flow_version"),
            "pending_confirmation_action": state.get("pending_confirmation_action"),
            "pending_confirmation_by_reservation": state.get("pending_confirmation_by_reservation", {}),
            "requested_cabin": state.get("requested_cabin"),
            "awaiting_choice_between_options": state.get("awaiting_choice_between_options"),
            "awaiting_flight_selection": state.get("awaiting_flight_selection"),
            "awaiting_payment_choice": state.get("awaiting_payment_choice"),
            "terminal_user_intent": state.get("terminal_user_intent"),
        }
    )
    derived = state.get("derived_facts")
    if not isinstance(derived, dict):
        derived = {}
    derived.update(
        {
            "cancel_eligible": state.get("cancel_eligible"),
            "cancel_reason": state.get("cancel_reason"),
            "compensation_ineligible": state.get("compensation_ineligible"),
            "nonstop_unavailable": state.get("nonstop_unavailable"),
            "reservation_locked": state.get("reservation_locked"),
            "already_summarized_write_result": state.get("already_summarized_write_result"),
            "last_tool_error": state.get("last_tool_error"),
            "error_streak": state.get("error_streak"),
        }
    )
    state["raw_facts"] = raw_facts
    state["working_memory"] = working
    state["derived_facts"] = derived


def record_completed_action(state: dict[str, Any], action: dict[str, Any]) -> None:
    name = action.get("name")
    if not isinstance(name, str) or not name:
        return
    completed_actions = state.get("completed_actions")
    if not isinstance(completed_actions, list):
        completed_actions = []
    reservation_id = None
    arguments = action.get("arguments")
    if isinstance(arguments, dict):
        candidate = arguments.get("reservation_id")
        if isinstance(candidate, str):
            reservation_id = candidate
    entity_aware_name = f"{name}:{reservation_id}" if reservation_id else name
    if entity_aware_name not in completed_actions:
        completed_actions.append(entity_aware_name)
    state["completed_actions"] = completed_actions
    active_flow = state.get("active_flow")
    if not isinstance(active_flow, dict):
        active_flow = {"name": determine_flow_name(state), "stage": "analyze"}
    if name in SUCCESSFUL_WRITE_TOOLS:
        active_flow["stage"] = "finalize"
        state["already_summarized_write_result"] = False
        state["reservation_locked"] = True
        state["pricing_context_version"] = state.get("flow_version")
    elif name in {"get_user_details", "get_reservation_details", "get_flight_status"}:
        active_flow["stage"] = "policy_check"
    elif name in {"search_direct_flight", "search_onestop_flight", "calculate"}:
        active_flow["stage"] = "lookup"
    elif name == "respond":
        if state.get("pending_confirmation_action"):
            active_flow["stage"] = "policy_check"
    state["active_flow"] = active_flow


def resolve_completed_subtasks(state: dict[str, Any], action: dict[str, Any]) -> None:
    queue = state.get("subtask_queue")
    if not isinstance(queue, list) or not queue:
        return
    name = action.get("name")
    if name in SUCCESSFUL_WRITE_TOOLS:
        queue[0]["status"] = "done"
    elif name == "respond":
        content = str(action.get("arguments", {}).get("content", "")).lower()
        if "not eligible" in content or "cannot" in content or "can't" in content:
            queue[0]["status"] = "done"
        if "here are" in content or "total" in content or "completed" in content:
            queue[0]["status"] = "done"
    completed = state.get("completed_subtasks")
    if not isinstance(completed, list):
        completed = []
    completed_ids = state.get("completed_subtask_ids")
    if not isinstance(completed_ids, list):
        completed_ids = []
    while queue and queue[0].get("status") == "done":
        item = queue.pop(0)
        completed.append(item)
        completed_ids.append(item.get("kind", "unknown"))
    state["subtask_queue"] = queue
    state["completed_subtasks"] = completed
    state["completed_subtask_ids"] = completed_ids
    if len(completed_ids) >= 2:
        state["already_resolved_subtask_B"] = True


def build_runtime_brief(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_flow": state.get("active_flow"),
        "subtask_queue": state.get("subtask_queue", [])[:4],
        "completed_subtasks": state.get("completed_subtasks", [])[-4:],
        "verified_facts": {
            "user_id": state.get("user_id"),
            "reservation_id": state.get("reservation_id"),
            "known_reservation_ids": state.get("known_reservation_ids", [])[:6],
            "requested_cabin": state.get("requested_cabin"),
            "travel_dates": state.get("travel_dates", [])[:4],
            "membership": state.get("membership"),
        },
        "open_questions": [
            label
            for label, value in (
                ("awaiting_choice_between_options", state.get("awaiting_choice_between_options")),
                ("awaiting_flight_selection", state.get("awaiting_flight_selection")),
                ("awaiting_payment_choice", state.get("awaiting_payment_choice")),
                ("pending_confirmation_action", state.get("pending_confirmation_action")),
            )
            if value
        ],
        "completed_actions": state.get("completed_actions", [])[-6:],
        "last_tool_error": state.get("last_tool_error"),
        "error_streak": state.get("error_streak"),
    }


def _success_tool_result(latest_user_text: str) -> dict[str, Any] | None:
    if not latest_user_text.strip().startswith("tool:"):
        return None
    payload_text = latest_user_text.split("tool:", 1)[1].strip()
    if not payload_text or "error" in payload_text.lower():
        return None
    return {"ok": True, "raw": payload_text}


def _redirect_to_next_subtask(state: dict[str, Any], queue: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an action that drives the agent toward the next pending subtask."""
    import sys
    next_kind = queue[0].get("kind") if queue else None
    print(f"[TC] blocking close, next subtask kind={next_kind}", file=sys.stderr)
    # For booking subtask: redirect to search if we have origin/destination/date
    if next_kind == "booking":
        origin = state.get("origin")
        destination = state.get("destination")
        travel_dates = state.get("travel_dates") or []
        if origin and destination and travel_dates:
            return {
                "name": "search_direct_flight",
                "arguments": {"origin": origin, "destination": destination, "date": travel_dates[0]},
            }
        if origin and destination:
            return {
                "name": "respond",
                "arguments": {
                    "content": f"Now let me help with your booking request. I have the route {origin} → {destination} noted. Please confirm the travel date (YYYY-MM-DD) and cabin preference so I can search available flights."
                },
            }
    # For cancel subtask: redirect to next reservation
    if next_kind == "cancel":
        known_ids = state.get("known_reservation_ids", [])
        inventory = state.get("reservation_inventory", {})
        if isinstance(known_ids, list) and isinstance(inventory, dict):
            for rid in known_ids:
                if isinstance(rid, str) and rid not in inventory:
                    return {"name": "get_reservation_details", "arguments": {"reservation_id": rid}}
    # For modify subtask: redirect to reservation lookup
    if next_kind in {"modify", "baggage"}:
        reservation_id = state.get("reservation_id")
        if isinstance(reservation_id, str):
            return {"name": "get_reservation_details", "arguments": {"reservation_id": reservation_id}}
    # Generic fallback
    return {
        "name": "respond",
        "arguments": {
            "content": f"Let me now address the pending {next_kind} request before we finish."
        },
    }


def termination_controller(state: dict[str, Any], action: dict[str, Any], latest_user_text: str) -> dict[str, Any]:
    import sys
    if state.get("terminal_user_intent") == "stop_without_changes":
        return {
            "name": "respond",
            "arguments": {
                "content": "Understood. I will leave the reservation unchanged and close out this request."
            },
        }
    if int(state.get("error_streak") or 0) >= 2:
        last_error = state.get("last_tool_error") or "The last lookup failed."
        return {
            "name": "respond",
            "arguments": {
                "content": f"{last_error} Please provide a corrected reservation ID, flight number, date, or tell me which reservation to inspect next."
            },
        }
    success = _success_tool_result(latest_user_text)
    if not success:
        queue = state.get("subtask_queue")
        if (
            action.get("name") == "respond"
            and isinstance(queue, list)
            and queue
            and any(
                token in str(action.get("arguments", {}).get("content", "")).lower()
                for token in ("all set", "done", "completed", "have a great day")
            )
        ):
            print(f"[TC] agent tried to close with pending queue={[s.get('kind') for s in queue]}", file=sys.stderr)
            return _redirect_to_next_subtask(state, queue)
        return action
    last_tool = state.get("last_tool_name")
    if last_tool in SUCCESSFUL_WRITE_TOOLS and action.get("name") in {"search_direct_flight", "search_onestop_flight", "get_user_details", "get_reservation_details"}:
        queue = state.get("subtask_queue")
        if isinstance(queue, list) and queue:
            next_kind = queue[0].get("kind")
            return {
                "name": "respond",
                "arguments": {
                    "content": (
                        "The previous change completed successfully. I should summarize that result"
                        + (f" and then continue with the next {next_kind} subtask." if isinstance(next_kind, str) else ".")
                    )
                },
            }
        return {
            "name": "respond",
            "arguments": {
                "content": "The requested change has already been completed successfully. I should summarize the result to the user now."
            },
        }
    if last_tool in SUCCESSFUL_WRITE_TOOLS and action.get("name") == "respond":
        state["already_summarized_write_result"] = True
        state["reservation_locked"] = False
        state["terminal_user_intent"] = None
    return action


def history_snapshot(state: dict[str, Any]) -> str:
    brief = build_runtime_brief(state)
    return json.dumps(brief, ensure_ascii=False)


def compressed_history_summary(messages: list[dict[str, Any]], state: dict[str, Any]) -> str:
    recent_user = [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "user" and isinstance(msg.get("content"), str)
    ][-3:]
    summary = {
        "verified_facts": state.get("verified_facts_cache", {}),
        "completed_actions": state.get("completed_actions", [])[-6:],
        "completed_subtask_ids": state.get("completed_subtask_ids", [])[-6:],
        "executed_actions_by_reservation": state.get("executed_actions_by_reservation", {}),
        "recent_user_goals": recent_user,
        "last_tool_error": state.get("last_tool_error"),
    }
    return json.dumps(summary, ensure_ascii=False)
