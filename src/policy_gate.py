"""Centralized hard policy gates for verification and social-pressure resistance."""

from __future__ import annotations

from typing import Any

try:
    from .extractors import extract_cancel_reason
    from .reservation import cancel_eligibility, current_reservation_entry
except ImportError:  # pragma: no cover
    from extractors import extract_cancel_reason
    from reservation import cancel_eligibility, current_reservation_entry


def _user_claims_insurance(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "i have insurance",
            "i purchased insurance",
            "i bought insurance",
            "i have travel insurance",
            "i got insurance",
        )
    )


def _user_claims_membership(text: str) -> str | None:
    """Return claimed membership tier if user is asserting it to justify cancel."""
    lowered = text.lower()
    if any(p in lowered for p in ("silver member", "i am silver", "i'm silver", "as a silver")):
        return "silver"
    if any(p in lowered for p in ("gold member", "i am gold", "i'm gold", "as a gold")):
        return "gold"
    return None


def _reservation_has_insurance(state: dict[str, Any], reservation_id: str | None) -> bool:
    entry = current_reservation_entry(state, reservation_id)
    if not isinstance(entry, dict):
        return False
    return entry.get("insurance") == "yes"


def evaluate_policy_gate(
    state: dict[str, Any],
    flow_name: str | None,
    action: dict[str, Any],
    latest_user_text: str,
) -> dict[str, Any] | None:
    name = action.get("name")
    arguments = action.get("arguments") or {}
    if not isinstance(arguments, dict) or not isinstance(name, str):
        return None

    derived = state.get("derived_facts", {})
    social_pressure = isinstance(derived, dict) and bool(
        derived.get("social_pressure_detected")
    )
    if social_pressure and name == "transfer_to_human_agents":
        return {
            "name": "respond",
            "arguments": {
                "content": "I still need to follow the policy exactly. I can continue if you want to provide the required booking details or choose one of the allowed next steps."
            },
        }

    if name == "cancel_reservation":
        reservation_id = arguments.get("reservation_id") or state.get("reservation_id")

        # Hard block: already established not eligible — social pressure cannot override DB facts.
        if state.get("cancel_eligible") is False:
            is_eligible, refusal = cancel_eligibility(state, reservation_id, latest_user_text)
            if not is_eligible:
                return {
                    "name": "respond",
                    "arguments": {
                        "content": refusal
                        or "This reservation is not eligible for cancellation under the airline policy. I must follow the policy and cannot proceed with this cancellation."
                    },
                }

        # Block if user claims insurance but DB shows none.
        if _user_claims_insurance(latest_user_text) and not _reservation_has_insurance(state, reservation_id):
            if current_reservation_entry(state, reservation_id) is not None:
                return {
                    "name": "respond",
                    "arguments": {
                        "content": "I've verified the reservation details and the booking does not include travel insurance. Without insurance, cancellation is only allowed if the flight was booked within 24 hours, the airline cancelled the flight, or the cabin is business class. Based on the verified booking details, none of those conditions apply here, so I'm unable to process this cancellation."
                    },
                }

        # Block if user claims membership tier as justification — membership does not enable cancel.
        claimed_membership = _user_claims_membership(latest_user_text)
        if claimed_membership:
            actual_membership = state.get("membership") or ""
            if actual_membership.lower() != claimed_membership:
                return {
                    "name": "respond",
                    "arguments": {
                        "content": f"I can see from the verified profile that your membership tier is {actual_membership or 'regular'}, not {claimed_membership}. In any case, membership tier does not affect cancellation eligibility — only business-class cabin, insurance with a covered reason, booking within 24 hours, or an airline-cancelled flight would allow this cancellation. None of those apply here."
                    },
                }
            # Even correct membership doesn't help — make it clear
            return {
                "name": "respond",
                "arguments": {
                    "content": f"I understand you are a {claimed_membership} member, but membership tier does not affect cancellation eligibility. Under the airline policy, a reservation can only be cancelled if it was booked within the last 24 hours, the airline cancelled the flight, the cabin is business class, or you have travel insurance with a covered reason (health or weather). None of those conditions apply to this reservation."
                },
            }

        reason_by_reservation = state.get("cancel_reason_by_reservation", {})
        reason = (
            reason_by_reservation.get(reservation_id)
            if isinstance(reason_by_reservation, dict) and isinstance(reservation_id, str)
            else None
        ) or state.get("cancel_reason") or extract_cancel_reason(latest_user_text)
        if not reason:
            return {
                "name": "respond",
                "arguments": {
                    "content": "Please share the cancellation reason so I can verify whether this reservation is eligible to be cancelled."
                },
            }

    if flow_name == "status_compensation":
        if name in {"calculate", "search_direct_flight", "search_onestop_flight"}:
            return {
                "name": "respond",
                "arguments": {
                    "content": "I should first verify the exact reservation and flight status before making any compensation decision."
                },
            }
        if name == "respond" and "compensation" in latest_user_text.lower():
            if not state.get("status_checked") and not state.get("flight_number"):
                return {
                    "name": "respond",
                    "arguments": {
                        "content": "I need the exact reservation or flight number and date so I can verify the flight status before discussing compensation."
                    },
                }

    return None
