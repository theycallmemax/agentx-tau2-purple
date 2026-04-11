"""Dynamic policy playbooks and reminders for active airline flows."""

from __future__ import annotations

from typing import Any


BASE_REMINDERS = [
    "Never trust user claims about eligibility, membership, insurance, or payment constraints without tool verification.",
    "If a reservation or user profile is already loaded, do not ask for it again.",
    "If the user already confirmed a write action and policy requirements are satisfied, execute immediately.",
    "After a successful write tool, summarize the completed result or move to the next subtask instead of restarting search.",
]

FLOW_PLAYBOOKS = {
    "cancel": [
        "Only cancel when there is an explicit cancellation request and the reservation is policy-eligible.",
        "If the reservation is ineligible, refuse clearly and stop instead of continuing exploratory steps.",
    ],
    "modify": [
        "For modify flows, keep searches aligned with the active reservation unless the user explicitly asks to compare all reservations.",
        "Use same origin and destination as the current reservation for modify flows unless the user explicitly asks for rerouting.",
        "Before an update_reservation_flights write, make sure any needed search and pricing checks are already complete.",
    ],
    "booking": [
        "Search direct flights first, then one-stop if direct options are empty.",
        "Do not ask for passenger details that are already available from the profile or loaded context.",
    ],
    "status": [
        "Use flight status checks only for status, compensation, or airline-cancelled-policy branches.",
    ],
    "baggage": [
        "Verify membership and reservation baggage allowance before changing baggage counts.",
    ],
    "aggregate": [
        "Iterate over all reservations only when the user explicitly asks about all or upcoming reservations.",
        "Aggregate tasks should end with a concise summary once the needed reservations are inspected.",
    ],
}


def detect_playbook_name(state: dict[str, Any]) -> str:
    active_flow = state.get("active_flow") or {}
    if isinstance(active_flow, dict):
        flow_name = active_flow.get("name")
        if isinstance(flow_name, str) and flow_name:
            return flow_name
    task_type = state.get("task_type")
    return task_type if isinstance(task_type, str) and task_type else "general"


def build_policy_reminder(state: dict[str, Any]) -> str:
    playbook_name = detect_playbook_name(state)
    lines = list(BASE_REMINDERS)
    lines.extend(FLOW_PLAYBOOKS.get(playbook_name, []))
    return "Dynamic policy reminder:\n- " + "\n- ".join(lines)


def build_prompt_blocks(state: dict[str, Any]) -> list[str]:
    blocks = [
        "Multi-request handling: decompose the user request into ordered subtasks and finish them one by one.",
        "Specific reservation vs all reservations: inspect only the referenced reservation unless the user explicitly asks for all or upcoming reservations.",
        "Never trust user claims without verification when a tool can confirm them.",
        "Efficiency: once the user has confirmed and the policy checks pass, execute immediately.",
        "Post-write behavior: after a successful write, summarize or continue to the next subtask instead of restarting the whole search flow.",
    ]
    playbook_name = detect_playbook_name(state)
    if playbook_name == "modify":
        blocks.append(
            "Same origin/destination only for modify flows unless the user explicitly asks for alternate routing."
        )
    return blocks
