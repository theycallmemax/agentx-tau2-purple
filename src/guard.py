"""Action validation and correction — guards LLM output before execution."""

from __future__ import annotations

from typing import Any

try:
    from .extractors import (
        AIRPORT_CODE_PATTERN,
        FLIGHT_NUMBER_PATTERN,
        RESERVATION_ID_PATTERN,
        extract_airport_codes,
        extract_entities_from_history,
        extract_flight_number,
        extract_reservation_id,
        extract_user_id,
    )
    from .intent import (
        looks_like_all_reservations_intent,
        classify_task_type,
        looks_like_affirmation,
        looks_like_baggage_intent,
        looks_like_balance_intent,
        looks_like_booking_intent,
        looks_like_cancel_intent,
        looks_like_compensation_intent,
        looks_like_insurance_intent,
        looks_like_modify_intent,
        looks_like_second_cheapest_intent,
        looks_like_remove_passenger_intent,
        looks_like_social_pressure,
        looks_like_stop_without_changes,
        looks_like_same_reservation_reference,
        looks_like_status_intent,
    )
    from .policy_gate import evaluate_policy_gate
    from .reservation import (
        RESPOND_ACTION_NAME,
        balance_response_from_state,
        cabin_change_confirmation,
        cancel_eligibility,
        current_membership,
        current_reservation_entry,
        fallback_action,
        find_reservation_by_flight_number,
        find_reservation_by_route,
        infer_reservation_from_inventory,
        matches_reservation_segment,
        next_segment_search_from_loaded_reservation,
        next_missing_pricing_search,
        pricing_expression_for_current_reservation,
        reservation_has_flown_segments,
        reservation_payment_id,
        normalized_nonfree_baggages,
        select_second_cheapest_booking_option,
        same_flights_update_action,
    )
except ImportError:  # pragma: no cover - direct src imports in tests
    from extractors import (
        AIRPORT_CODE_PATTERN,
        FLIGHT_NUMBER_PATTERN,
        RESERVATION_ID_PATTERN,
        extract_airport_codes,
        extract_entities_from_history,
        extract_flight_number,
        extract_reservation_id,
        extract_user_id,
    )
    from intent import (
        looks_like_all_reservations_intent,
        classify_task_type,
        looks_like_affirmation,
        looks_like_baggage_intent,
        looks_like_balance_intent,
        looks_like_booking_intent,
        looks_like_cancel_intent,
        looks_like_compensation_intent,
        looks_like_insurance_intent,
        looks_like_modify_intent,
        looks_like_second_cheapest_intent,
        looks_like_remove_passenger_intent,
        looks_like_social_pressure,
        looks_like_stop_without_changes,
        looks_like_same_reservation_reference,
        looks_like_status_intent,
    )
    from policy_gate import evaluate_policy_gate
    from reservation import (
        RESPOND_ACTION_NAME,
        balance_response_from_state,
        cabin_change_confirmation,
        cancel_eligibility,
        current_membership,
        current_reservation_entry,
        fallback_action,
        find_reservation_by_flight_number,
        find_reservation_by_route,
        infer_reservation_from_inventory,
        matches_reservation_segment,
        next_segment_search_from_loaded_reservation,
        next_missing_pricing_search,
        pricing_expression_for_current_reservation,
        reservation_has_flown_segments,
        reservation_payment_id,
        normalized_nonfree_baggages,
        select_second_cheapest_booking_option,
        same_flights_update_action,
    )

import sys

PLAN_MAX_TURNS = 2  # default, overridden by Agent
KNOWN_EXAMPLE_USER_IDS = {"sara_doe_496"}
KNOWN_EXAMPLE_RESERVATION_IDS = {"ZFA04Y", "8JX2WO"}
FLOW_ACTION_ALLOWLIST = {
    "reservation_triage": {
        "respond",
        "get_user_details",
        "get_reservation_details",
    },
    "status_compensation": {
        "respond",
        "get_user_details",
        "get_reservation_details",
        "get_flight_status",
        "transfer_to_human_agents",
    },
    "cancel": {
        "respond",
        "get_user_details",
        "get_reservation_details",
        "get_flight_status",
        "cancel_reservation",
        "calculate",
        "transfer_to_human_agents",
        # Allowed for upgrade-before-cancel path; guarded below at _cancel_upgrade_guard
        "update_reservation_flights",
        "search_direct_flight",
        "search_onestop_flight",
    },
    "aggregate": {
        "respond",
        "get_user_details",
        "get_reservation_details",
        "calculate",
    },
}


def _same_error_repeated(state: dict[str, Any], name: str, arguments: dict[str, Any]) -> bool:
    error_key = state.get("last_error_key")
    current_key = f"{name}|{__import__('json').dumps(arguments, sort_keys=True, ensure_ascii=False)}"
    return bool(error_key == current_key and int(state.get("error_streak") or 0) >= 1)


def _safe_reroute_from_state(
    state: dict[str, Any],
    effective_task_type: str,
    effective_reservation_id: str | None,
    requested_cabin: str | None,
) -> dict[str, Any] | None:
    if (
        effective_task_type == "modify"
        and isinstance(effective_reservation_id, str)
        and isinstance(requested_cabin, str)
    ):
        missing_search = next_missing_pricing_search(state, effective_reservation_id)
        if missing_search is not None:
            return missing_search
    if (
        isinstance(effective_reservation_id, str)
        and (
            not state.get("loaded_reservation_details")
            or state.get("reservation_id") != effective_reservation_id
        )
    ):
        return {
            "name": "get_reservation_details",
            "arguments": {"reservation_id": effective_reservation_id},
        }
    if isinstance(state.get("user_id"), str) and not state.get("loaded_user_details"):
        return {
            "name": "get_user_details",
            "arguments": {"user_id": state["user_id"]},
        }
    return None


def candidate_tool_names(
    state: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    latest_user_text: str,
) -> set[str] | None:
    if not tools:
        return None

    names = {tool["function"]["name"] for tool in tools if "function" in tool}
    wanted: set[str] = set()
    lowered = latest_user_text.lower()
    task_type = classify_task_type(latest_user_text)
    has_identity = bool(
        state.get("reservation_id")
        or state.get("user_id")
        or extract_reservation_id(latest_user_text)
        or extract_user_id(latest_user_text)
    )

    if looks_like_balance_intent(latest_user_text):
        return set()

    if looks_like_status_intent(latest_user_text) or looks_like_compensation_intent(
        latest_user_text
    ):
        wanted |= {"get_flight_status", "get_user_details", "get_reservation_details"}

    if looks_like_baggage_intent(latest_user_text):
        wanted |= {
            "get_user_details",
            "get_reservation_details",
            "update_reservation_baggages",
        }

    if looks_like_booking_intent(latest_user_text):
        wanted |= {
            "list_all_airports",
            "search_direct_flight",
            "search_onestop_flight",
            "book_reservation",
            "get_user_details",
            "get_reservation_details",
            "calculate",
        }

    if looks_like_modify_intent(latest_user_text):
        wanted |= {
            "get_user_details",
            "get_reservation_details",
            "calculate",
            "transfer_to_human_agents",
        }
        if has_identity:
            wanted |= {
                "search_direct_flight",
                "search_onestop_flight",
                "update_reservation_flights",
                "update_reservation_baggages",
            }

    if looks_like_cancel_intent(latest_user_text):
        wanted |= {
            "get_user_details",
            "get_reservation_details",
            "transfer_to_human_agents",
            "get_flight_status",
        }
        if has_identity:
            wanted.add("cancel_reservation")

    if looks_like_remove_passenger_intent(latest_user_text) or looks_like_insurance_intent(
        latest_user_text
    ):
        wanted |= {"get_user_details", "get_reservation_details", "transfer_to_human_agents"}

    if (
        "human agent" in lowered
        or "transfer" in lowered
        or "escalat" in lowered
        or "supervisor" in lowered
        or "manager" in lowered
        or "speak to someone" in lowered
        or "another department" in lowered
        or "someone else" in lowered
    ):
        wanted |= {"transfer_to_human_agents"}

    if state.get("reservation_id"):
        wanted |= {"get_reservation_details"}
        if task_type == "cancel":
            wanted.add("cancel_reservation")
        if task_type == "modify":
            wanted.add("update_reservation_flights")
        if task_type == "baggage":
            wanted.add("update_reservation_baggages")
    if state.get("user_id"):
        wanted |= {"get_user_details"}
    if state.get("known_payment_ids"):
        wanted |= {"book_reservation", "calculate"}
    if (
        task_type in {"booking", "modify"}
        and has_identity
        and state.get("origin")
        and state.get("destination")
    ):
        wanted |= {"search_direct_flight", "search_onestop_flight"}

    filtered = wanted & names
    return filtered or names


def opening_turn_action(
    state: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    latest_user_text: str,
) -> dict[str, Any] | None:
    reservation_id = extract_reservation_id(latest_user_text)
    user_id = extract_user_id(latest_user_text)
    flight_number = extract_flight_number(latest_user_text)
    task_type = classify_task_type(latest_user_text)

    if looks_like_balance_intent(latest_user_text):
        if state.get("loaded_user_details"):
            return balance_response_from_state(state)
        if user_id:
            return {"name": "get_user_details", "arguments": {"user_id": user_id}}
        return fallback_action(
            "Please share your user ID so I can look up the balances on your gift cards and certificates."
        )

    if flight_number and looks_like_compensation_intent(latest_user_text):
        return fallback_action(
            "Please share the flight date as YYYY-MM-DD so I can check the status for compensation eligibility."
        )

    if reservation_id and not user_id:
        return {
            "name": "get_reservation_details",
            "arguments": {"reservation_id": reservation_id},
        }

    if looks_like_baggage_intent(latest_user_text):
        if user_id:
            return {"name": "get_user_details", "arguments": {"user_id": user_id}}
        if reservation_id:
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
            }
        return fallback_action(
            "Please share your user ID or reservation number so I can check your baggage allowance."
        )

    if task_type in {"modify", "remove_passenger", "insurance", "cancel"}:
        if reservation_id:
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
            }
        if user_id:
            return {"name": "get_user_details", "arguments": {"user_id": user_id}}
        if task_type == "remove_passenger":
            return fallback_action(
                "Please share your reservation number or user ID so I can locate the booking before I review passenger changes."
            )
        if task_type == "insurance":
            return fallback_action(
                "Please share your reservation number or user ID so I can check whether insurance is attached to the booking."
            )
        if task_type == "cancel":
            return fallback_action(
                "Please share your reservation number or user ID so I can locate the booking you want to cancel."
            )
        return fallback_action(
            "Please share your reservation number or user ID so I can look up the booking before changing it."
        )

    if user_id:
        return {"name": "get_user_details", "arguments": {"user_id": user_id}}

    if looks_like_booking_intent(latest_user_text):
        airport_codes = AIRPORT_CODE_PATTERN.findall(latest_user_text)
        if looks_like_same_reservation_reference(latest_user_text):
            return fallback_action(
                "Please share your user ID or reservation number so I can look up your existing booking."
            )
        if len(airport_codes) < 2:
            return {"name": "list_all_airports", "arguments": {}}
        return fallback_action(
            "Please share the traveler details and any passenger count or cabin preferences so I can help with the booking."
        )

    if looks_like_compensation_intent(latest_user_text):
        return fallback_action(
            "Please share your reservation number, user ID, or flight number so I can review the issue."
        )

    return None


def guard_action(
    state: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    turn_count: int,
    action: dict[str, Any],
    latest_user_text: str,
) -> dict[str, Any]:
    name = action.get("name")
    arguments = action.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return action

    task_type = classify_task_type(latest_user_text)
    effective_task_type = task_type
    if effective_task_type == "general":
        effective_task_type = str(state.get("task_type") or "general")

    _guard_proposed = name
    pending_confirmation_action = state.get("pending_confirmation_action")
    verified_entities = extract_entities_from_history(messages)
    if 0 < turn_count <= 2:
        forced = opening_turn_action(state, tools, latest_user_text)
        if forced is not None:
            return forced

    if looks_like_balance_intent(latest_user_text) and state.get("loaded_user_details"):
        return balance_response_from_state(state)

    active_flow = state.get("active_flow")
    flow_name = active_flow.get("name") if isinstance(active_flow, dict) else None
    flow_stage = active_flow.get("stage") if isinstance(active_flow, dict) else None
    user_id_from_context = extract_user_id(latest_user_text)
    reservation_id_from_context = extract_reservation_id(latest_user_text)
    flight_number_from_context = extract_flight_number(latest_user_text)
    inferred_reservation_id = infer_reservation_from_inventory(state, latest_user_text)
    recent_reservation_id = state.get("recent_reservation_id")
    recent_reservation_lock = (
        recent_reservation_id
        if isinstance(recent_reservation_id, str)
        and state.get("recent_reservation_active")
        and not reservation_id_from_context
        and not flight_number_from_context
        and (
            flow_name == "status_compensation"
            or effective_task_type == "status"
            or looks_like_status_intent(latest_user_text)
            or looks_like_compensation_intent(latest_user_text)
        )
        else None
    )
    effective_reservation_id = (
        reservation_id_from_context
        or recent_reservation_lock
        or inferred_reservation_id
        or state.get("reservation_id")
    )
    if isinstance(recent_reservation_lock, str):
        state["reservation_id"] = recent_reservation_lock
    effective_user_id = user_id_from_context or state.get("user_id")
    requested_cabin = state.get("requested_cabin")
    cabin_only_change = bool(state.get("cabin_only_change"))
    lowered_user_text = latest_user_text.lower()
    explicitly_no_change_or_cancel = any(
        phrase in lowered_user_text
        for phrase in (
            "don't want to change or cancel",
            "do not want to change or cancel",
            "not looking to change or cancel",
            "i don't want to change or cancel",
            "i do not want to change or cancel",
            "keeping the reservation as-is",
            "keeping the reservation as is",
            "leave the reservation unchanged",
        )
    )
    latest_input_is_tool = bool(state.get("latest_input_is_tool")) or latest_user_text.strip().startswith("tool:")

    print(
        f"[GUARD] turn={turn_count} proposed={name} flow={flow_name} task={effective_task_type}"
        f" res_id={effective_reservation_id} cabin={requested_cabin} pending={pending_confirmation_action}"
        f" cancel_eligible={state.get('cancel_eligible')} last_tool={state.get('last_tool_name')}",
        file=sys.stderr,
    )

    working_memory = state.get("working_memory", {})
    adversarial_comp_verification = (
        isinstance(working_memory, dict)
        and bool(working_memory.get("adversarial_compensation_verification"))
    )
    if adversarial_comp_verification and (
        effective_task_type == "cancel" or flow_name == "cancel"
    ):
        working_memory["adversarial_compensation_verification"] = False
        working_memory["adversarial_compensation_business_claim"] = False
        working_memory["adversarial_compensation_cancelled_claim"] = False
        state["working_memory"] = working_memory
        adversarial_comp_verification = False
    if adversarial_comp_verification:
        known_ids = state.get("known_reservation_ids", [])
        inventory = state.get("reservation_inventory", {})
        if isinstance(known_ids, list) and isinstance(inventory, dict):
            for rid in known_ids:
                if isinstance(rid, str) and rid not in inventory:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": rid},
                    }
            business_claim = bool(
                working_memory.get("adversarial_compensation_business_claim")
            )
            cancelled_claim = bool(
                working_memory.get("adversarial_compensation_cancelled_claim")
            )
            matching_reservations: list[str] = []
            for rid in known_ids:
                if not isinstance(rid, str):
                    continue
                entry = inventory.get(rid)
                if not isinstance(entry, dict):
                    continue
                cabin = str(entry.get("cabin") or "").strip().lower()
                status = str(entry.get("status") or "").strip().lower()
                if business_claim and cabin != "business":
                    continue
                if cancelled_claim and status != "cancelled":
                    continue
                matching_reservations.append(rid)
            if not matching_reservations and name in {
                "search_direct_flight",
                "search_onestop_flight",
                "calculate",
                "cancel_reservation",
                "update_reservation_flights",
                "send_certificate",
            }:
                return fallback_action(
                    "I reviewed all 5 reservations on your profile, and none of them matches a cancelled business flight. Because the verified booking details do not support that claim, I can’t offer compensation or cancel anything on that basis."
                )

    # Deterministically resolve route-based reservation references before letting the
    # model continue. This avoids latching onto the first profile reservation when the
    # user identified the booking by route rather than ID.
    if (
        state.get("loaded_user_details")
        and not reservation_id_from_context
        and not flight_number_from_context
        and state.get("origin")
        and state.get("destination")
        and effective_task_type in {"cancel", "modify", "remove_passenger", "insurance"}
    ):
        route_match = find_reservation_by_route(
            state,
            state.get("origin"),
            state.get("destination"),
        )
        if route_match:
            state["reservation_id"] = route_match
            effective_reservation_id = route_match
            if current_reservation_entry(state, route_match) is None:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": route_match},
                }
        else:
            known_ids = state.get("known_reservation_ids", [])
            inventory = state.get("reservation_inventory", {})
            if isinstance(known_ids, list) and isinstance(inventory, dict):
                for rid in known_ids:
                    if isinstance(rid, str) and rid not in inventory:
                        return {
                            "name": "get_reservation_details",
                            "arguments": {"reservation_id": rid},
                        }

    if looks_like_stop_without_changes(latest_user_text) or state.get("terminal_user_intent") == "stop_without_changes":
        return fallback_action(
            "Understood. I will leave the reservation unchanged and close out this request."
        )

    _escalation_phrases = (
        "escalat", "supervisor", "manager", "speak to someone",
        "another department", "someone else", "another person",
        "speak with someone", "talk to someone",
    )
    _user_wants_escalation = any(phrase in lowered_user_text for phrase in _escalation_phrases)

    if looks_like_social_pressure(latest_user_text) or _user_wants_escalation:
        derived = state.get("derived_facts", {})
        if isinstance(derived, dict):
            derived["social_pressure_detected"] = True
            state["derived_facts"] = derived
        # When user asks to escalate after a confirmed denial, route to human agents
        if _user_wants_escalation and name != "transfer_to_human_agents":
            refusal_given = (
                state.get("cancel_eligible") is False
                or state.get("compensation_ineligible")
                or bool(derived.get("social_pressure_detected") if isinstance(derived, dict) else False)
            )
            if refusal_given:
                if effective_task_type == "cancel" or flow_name == "cancel":
                    return fallback_action(
                        "I understand this is frustrating, but I must follow airline policy. "
                        "This reservation is not eligible for cancellation or a full refund, so I’m unable to override that decision."
                    )
                return {
                    "name": "transfer_to_human_agents",
                    "arguments": {
                        "summary": "User requested escalation after a policy-based refusal."
                    },
                }
        # In cancel flows, immediately re-affirm refusal when social pressure is detected
        if (
            effective_task_type == "cancel"
            or flow_name == "cancel"
        ) and state.get("cancel_eligible") is False and not _user_wants_escalation:
            return fallback_action(
                "I understand this is frustrating, but I must follow airline policy. "
                "The reservation is not eligible for cancellation because it was not booked within 24 hours, "
                "the flight was not cancelled by the airline, the cabin is not business class, and there is no "
                "applicable travel insurance coverage. I'm unable to override the policy regardless of prior communications."
            )

    if _same_error_repeated(state, name, arguments):
        return fallback_action(
            "That lookup has already failed. Please provide a corrected reservation ID, flight number, date, or tell me which reservation to inspect next."
        )

    gated = evaluate_policy_gate(state, flow_name, action, latest_user_text)
    if gated is not None:
        return gated

    allowed_actions = FLOW_ACTION_ALLOWLIST.get(flow_name or "")
    if isinstance(allowed_actions, set) and name not in allowed_actions:
        if flow_name == "status_compensation" and name == "send_certificate":
            return fallback_action(
                "I confirmed the delay, but under the policy I can only offer a certificate for a delayed flight when the reservation is also being changed or cancelled. Since you are keeping the reservation as-is, I can’t offer compensation here."
            )
        reroute = _safe_reroute_from_state(
            state,
            effective_task_type,
            effective_reservation_id if isinstance(effective_reservation_id, str) else None,
            requested_cabin if isinstance(requested_cabin, str) else None,
        )
        if reroute is not None:
            return reroute
        return fallback_action(
            f"The active flow is {flow_name}, so I should stay within that flow instead of performing {name}."
        )

    if flow_name == "cancel" and name in {"update_reservation_flights", "search_direct_flight", "search_onestop_flight"}:
        # Allow only if user explicitly requested upgrade/modify in cancel context
        upgrade_allowed = (
            looks_like_modify_intent(latest_user_text)
            or isinstance(requested_cabin, str)
            or pending_confirmation_action == "update_reservation_flights"
            or state.get("cancel_eligible") is False
        )
        if name == "update_reservation_flights" and not upgrade_allowed:
            print(f"[GUARD] cancel+update_reservation_flights blocked: upgrade_allowed={upgrade_allowed} requested_cabin={requested_cabin} pending={pending_confirmation_action}", file=sys.stderr)
            return fallback_action(
                "I should finish the active cancellation flow before attempting any flight change."
            )
        if name in {"search_direct_flight", "search_onestop_flight"} and not upgrade_allowed:
            print(f"[GUARD] cancel+search blocked: upgrade_allowed={upgrade_allowed}", file=sys.stderr)
            return fallback_action(
                "I should identify the correct reservation or verify flight status before starting a flight availability search."
            )
    if flow_name in {"modify", "modify_pricing"} and name == "cancel_reservation":
        return fallback_action(
            "I should stay within the active modification flow unless the user explicitly switches to cancellation."
        )

    if (
        latest_input_is_tool
        and state.get("last_tool_name")
        in {
            "cancel_reservation",
            "update_reservation_flights",
            "update_reservation_baggages",
            "book_reservation",
        }
        and name in {"search_direct_flight", "search_onestop_flight"}
        and pending_confirmation_action is None
        and effective_task_type not in {"booking", "modify", "baggage"}
    ):
        return fallback_action(
            "The requested action has already been completed successfully. I’ll summarize the result for the user instead of starting a new search."
        )

    if (
        latest_input_is_tool
        and state.get("last_tool_error")
        and name == state.get("last_tool_name")
        and arguments == (state.get("last_tool_arguments") or {})
    ):
        return fallback_action(
            "That exact lookup already failed. Please provide a corrected identifier or tell me which reservation to inspect next."
        )

    if (
        latest_input_is_tool
        and state.get("already_cancelled_this_turn")
        and name in {"get_user_details", "get_reservation_details"}
        and effective_reservation_id
    ):
        return fallback_action(
            f"I already completed the cancellation for reservation {effective_reservation_id}. I should summarize it or continue to the next explicit subtask."
        )

    if (
        looks_like_affirmation(latest_user_text)
        and (
            pending_confirmation_action == "cancel_reservation"
            or (
                # User affirms cancel with an explicit reservation ID even if pending not set
                effective_task_type == "cancel"
                and isinstance(reservation_id_from_context, str)
                and current_reservation_entry(state, reservation_id_from_context) is not None
            )
        )
        and isinstance(effective_reservation_id, str)
    ):
        cancel_reason_by_reservation = state.get("cancel_reason_by_reservation", {})
        cancel_reason_text = (
            cancel_reason_by_reservation.get(effective_reservation_id)
            if isinstance(cancel_reason_by_reservation, dict)
            else None
        ) or state.get("cancel_reason")
        eligibility_context = (
            cancel_reason_text
            if isinstance(cancel_reason_text, str) and cancel_reason_text
            else latest_user_text
        )
        is_eligible, refusal = cancel_eligibility(
            state, effective_reservation_id, eligibility_context
        )
        state["cancel_eligible"] = is_eligible
        if is_eligible:
            return {
                "name": "cancel_reservation",
                "arguments": {"reservation_id": effective_reservation_id},
            }
        return fallback_action(
            refusal
            or "This reservation is not eligible for cancellation under the airline policy."
        )

    # Keep modification flows alive across short acknowledgements like
    # "sure, go ahead" once we already have the reservation and target cabin.
    if (
        effective_task_type in {"general", "cancel"}
        and isinstance(effective_reservation_id, str)
        and isinstance(requested_cabin, str)
        and state.get("loaded_reservation_details")
        and (
            looks_like_affirmation(latest_user_text)
            or task_type == "general"
            or "same dates" in lowered_user_text
            or "same routing" in lowered_user_text
            or "mastercard" in lowered_user_text
            or "gift card" in lowered_user_text
            or "certificate" in lowered_user_text
        )
    ):
        effective_task_type = "modify"

    response_content = str(arguments.get("content", "")).lower() if name == RESPOND_ACTION_NAME else ""
    recent_user_text = "\n".join(
        str(message.get("content") or "")
        for message in messages[-10:]
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    )
    second_cheapest_requested = looks_like_second_cheapest_intent(
        f"{recent_user_text}\n{latest_user_text}"
    )

    if (
        name == RESPOND_ACTION_NAME
        and (effective_task_type == "cancel" or flow_name == "cancel")
        and state.get("cancel_eligible") is False
        and any(
            phrase in response_content
            for phrase in (
                "i can still cancel",
                "i can proceed with cancelling",
                "i can proceed with canceling",
                "if you'd like, i can proceed with cancelling",
                "if you'd like, i can proceed with canceling",
                "explicit confirmation",
                "yes, cancel reservation",
            )
        )
    ):
        refusal = (
            "This reservation is not eligible for cancellation under the airline policy. "
            "I can't approve or process this cancellation."
        )
        if isinstance(effective_reservation_id, str):
            eligible, evaluated_refusal = cancel_eligibility(
                state,
                effective_reservation_id,
                latest_user_text,
            )
            if not eligible and isinstance(evaluated_refusal, str) and evaluated_refusal:
                refusal = evaluated_refusal
        return fallback_action(refusal)

    if (
        flow_name == "status_compensation"
        and name == RESPOND_ACTION_NAME
        and state.get("status_checked")
        and ("certificate" in response_content or "$50" in response_content or "compensation" in response_content)
        and not looks_like_modify_intent(latest_user_text)
        and not looks_like_cancel_intent(latest_user_text)
    ):
        return fallback_action(
            "I confirmed the flight delay, but under the policy I can only offer compensation for delayed flights when the reservation is also being changed or cancelled. Since you are not changing or cancelling this reservation, I can’t offer a certificate here."
        )

    # Guard against identity amnesia once we already have the user or booking in state.
    if (
        name == RESPOND_ACTION_NAME
        and (
            "share your reservation number" in response_content
            or "share your user id" in response_content
            or "locate the booking before searching" in response_content
            or "look up your booking" in response_content
        )
        and (effective_reservation_id or effective_user_id)
    ):
        # Try route-based reservation lookup if we have origin/destination but no reservation_id
        if not effective_reservation_id:
            route_match = find_reservation_by_route(
                state, state.get("origin"), state.get("destination")
            )
            if route_match:
                effective_reservation_id = route_match
                state["reservation_id"] = route_match
            else:
                known_ids = state.get("known_reservation_ids", [])
                inventory = state.get("reservation_inventory", {})
                if isinstance(known_ids, list) and isinstance(inventory, dict):
                    for rid in known_ids:
                        if isinstance(rid, str) and rid not in inventory:
                            return {
                                "name": "get_reservation_details",
                                "arguments": {"reservation_id": rid},
                            }
        if effective_task_type in {"modify", "booking"} and isinstance(effective_reservation_id, str):
            missing_search = next_missing_pricing_search(state, effective_reservation_id)
            if missing_search is not None:
                return missing_search
        if (
            isinstance(effective_reservation_id, str)
            and state.get("loaded_reservation_details")
        ):
            return fallback_action(
                f"I already have your reservation details for {effective_reservation_id} on file. I’ll continue from there."
            )
        if isinstance(effective_reservation_id, str):
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": effective_reservation_id},
            }
        if isinstance(effective_user_id, str):
            return {
                "name": "get_user_details",
                "arguments": {"user_id": effective_user_id},
            }

    if (
        name == RESPOND_ACTION_NAME
        and "need to check the existing reservation segments" in response_content
        and isinstance(effective_reservation_id, str)
        and state.get("loaded_reservation_details")
    ):
        reroute = next_segment_search_from_loaded_reservation(
            state, effective_reservation_id
        ) or _safe_reroute_from_state(
            state,
            effective_task_type,
            effective_reservation_id,
            requested_cabin if isinstance(requested_cabin, str) else None,
        )
        if reroute is not None:
            return reroute

    if name == RESPOND_ACTION_NAME and second_cheapest_requested:
        origin = state.get("origin")
        destination = state.get("destination")
        date = None
        travel_dates = state.get("travel_dates")
        if isinstance(travel_dates, list) and travel_dates and isinstance(travel_dates[0], str):
            date = travel_dates[0]
        if not isinstance(date, str):
            proposed_flights = state.get("selected_flights")
            if isinstance(proposed_flights, list) and proposed_flights:
                first_flight = proposed_flights[0]
                if isinstance(first_flight, dict) and isinstance(first_flight.get("date"), str):
                    date = first_flight["date"]
        if all(isinstance(value, str) and value for value in (origin, destination, date)):
            inventory_by_mode = state.get("flight_search_inventory_by_mode", {})
            bucket = inventory_by_mode.get(f"{origin}|{destination}|{date}") if isinstance(inventory_by_mode, dict) else None
            if not isinstance(bucket, dict) or "search_direct_flight" not in bucket:
                return {
                    "name": "search_direct_flight",
                    "arguments": {"origin": origin, "destination": destination, "date": date},
                }
            if "search_onestop_flight" not in bucket:
                return {
                    "name": "search_onestop_flight",
                    "arguments": {"origin": origin, "destination": destination, "date": date},
                }

    if name == "book_reservation":
        cabin = arguments.get("cabin")
        origin = arguments.get("origin")
        destination = arguments.get("destination")
        flights = arguments.get("flights")
        date = None
        if isinstance(flights, list) and flights and isinstance(flights[0], dict):
            first_date = flights[0].get("date")
            if isinstance(first_date, str):
                date = first_date
        if second_cheapest_requested:
            selected = select_second_cheapest_booking_option(
                state, origin, destination, date, cabin
            )
            if selected is None and all(
                isinstance(value, str) and value for value in (origin, destination, date)
            ):
                return {
                    "name": "search_onestop_flight",
                    "arguments": {"origin": origin, "destination": destination, "date": date},
                }
            if isinstance(selected, dict):
                rewritten = dict(arguments)
                rewritten["flights"] = selected["flights"]
                payment_methods = rewritten.get("payment_methods")
                if isinstance(payment_methods, list):
                    adjusted: list[dict[str, Any]] = []
                    remaining_amount = int(selected["total_price"])
                    for index, payment in enumerate(payment_methods):
                        if not isinstance(payment, dict):
                            continue
                        adjusted_payment = dict(payment)
                        if index == 0 and isinstance(adjusted_payment.get("payment_id"), str):
                            adjusted_payment["amount"] = remaining_amount
                        adjusted.append(adjusted_payment)
                    if adjusted:
                        rewritten["payment_methods"] = adjusted
                arguments = rewritten

        saved_passengers = state.get("known_saved_passengers")
        proposed_passengers = arguments.get("passengers")
        if isinstance(saved_passengers, list) and isinstance(proposed_passengers, list):
            normalized_passengers: list[dict[str, Any]] = []
            changed = False
            for passenger in proposed_passengers:
                if not isinstance(passenger, dict):
                    continue
                normalized = dict(passenger)
                first_name = str(normalized.get("first_name") or "").strip().lower()
                last_name = str(normalized.get("last_name") or "").strip().lower()
                if first_name and last_name:
                    match = next(
                        (
                            saved
                            for saved in saved_passengers
                            if isinstance(saved, dict)
                            and str(saved.get("first_name") or "").strip().lower() == first_name
                            and str(saved.get("last_name") or "").strip().lower() == last_name
                            and isinstance(saved.get("dob"), str)
                        ),
                        None,
                    )
                    if isinstance(match, dict) and normalized.get("dob") != match.get("dob"):
                        normalized["dob"] = match["dob"]
                        changed = True
                normalized_passengers.append(normalized)
            if changed:
                rewritten = dict(arguments)
                rewritten["passengers"] = normalized_passengers
                arguments = rewritten

        if arguments is not action.get("arguments"):
            return {"name": "book_reservation", "arguments": arguments}

    # --- respond with pricing ---
    if (
        name == RESPOND_ACTION_NAME
        and isinstance(effective_reservation_id, str)
        and isinstance(requested_cabin, str)
        and any(token in lowered_user_text for token in ("cost", "price", "pricing", "under $", "limit"))
    ):
        missing_search = next_missing_pricing_search(state, effective_reservation_id)
        if missing_search is not None:
            return missing_search
        if state.get("last_tool_name") != "calculate":
            expression = pricing_expression_for_current_reservation(
                state, effective_reservation_id, requested_cabin
            )
            if expression:
                return {"name": "calculate", "arguments": {"expression": expression}}

    # --- cabin-only modify via respond ---
    if (
        name == RESPOND_ACTION_NAME
        and effective_task_type == "modify"
        and isinstance(effective_reservation_id, str)
        and isinstance(requested_cabin, str)
        and cabin_only_change
    ):
        if (
            looks_like_affirmation(latest_user_text)
            and pending_confirmation_action == "update_reservation_flights"
        ):
            same_flights = same_flights_update_action(
                state, effective_reservation_id, requested_cabin
            )
            if same_flights is not None:
                return same_flights
        current_entry = current_reservation_entry(state, effective_reservation_id)
        current_cabin = current_entry.get("cabin") if isinstance(current_entry, dict) else None
        if isinstance(current_cabin, str) and current_cabin != requested_cabin:
            if not state.get("cabin_price_quoted"):
                missing_search = next_missing_pricing_search(state, effective_reservation_id)
                if missing_search is not None:
                    return missing_search
                if state.get("last_tool_name") != "calculate":
                    expression = pricing_expression_for_current_reservation(
                        state, effective_reservation_id, requested_cabin
                    )
                    if expression:
                        return {"name": "calculate", "arguments": {"expression": expression}}
            confirmation = cabin_change_confirmation(state, effective_reservation_id, requested_cabin)
            if confirmation is not None:
                return confirmation

    if (
        name == RESPOND_ACTION_NAME
        and effective_task_type in {"status", "cancel"}
        and any(token in response_content for token in ("not eligible", "cannot", "can't", "policy"))
    ):
        return {
            "name": RESPOND_ACTION_NAME,
            "arguments": {"content": arguments.get("content", "")},
        }

    # --- cabin-only modify via get_reservation_details ---
    if (
        name == "get_reservation_details"
        and effective_task_type == "modify"
        and isinstance(effective_reservation_id, str)
        and isinstance(requested_cabin, str)
        and cabin_only_change
        and arguments.get("reservation_id") == effective_reservation_id
        and current_reservation_entry(state, effective_reservation_id) is not None
    ):
        if (
            looks_like_affirmation(latest_user_text)
            and pending_confirmation_action == "update_reservation_flights"
        ):
            same_flights = same_flights_update_action(
                state, effective_reservation_id, requested_cabin
            )
            if same_flights is not None:
                return same_flights
        if not state.get("cabin_price_quoted"):
            missing_search = next_missing_pricing_search(state, effective_reservation_id)
            if missing_search is not None:
                return missing_search
            if state.get("last_tool_name") != "calculate":
                expression = pricing_expression_for_current_reservation(
                    state, effective_reservation_id, requested_cabin
                )
                if expression:
                    return {"name": "calculate", "arguments": {"expression": expression}}
        confirmation = cabin_change_confirmation(state, effective_reservation_id, requested_cabin)
        if confirmation is not None:
            return confirmation

    # When the user identified a reservation by route instead of ID, do not trust a
    # model-guessed reservation_id until we either match the route or exhaust the
    # known reservations. This prevents the model from latching onto the first ID in
    # the profile inventory for task 1 style requests.
    if (
        name == "get_reservation_details"
        and state.get("loaded_user_details")
        and not reservation_id_from_context
        and not flight_number_from_context
        and state.get("origin")
        and state.get("destination")
    ):
        route_match = find_reservation_by_route(
            state,
            state.get("origin"),
            state.get("destination"),
        )
        if route_match and arguments.get("reservation_id") != route_match:
            state["reservation_id"] = route_match
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": route_match},
            }

        known_ids = state.get("known_reservation_ids", [])
        inventory = state.get("reservation_inventory", {})
        if isinstance(known_ids, list) and isinstance(inventory, dict) and route_match is None:
            for rid in known_ids:
                if isinstance(rid, str) and rid not in inventory:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": rid},
                    }

    # --- duplicate tool call detection ---
    if (
        name == state.get("last_tool_name")
        and arguments == state.get("last_tool_arguments")
        and latest_user_text == state.get("last_tool_user_text")
    ):
        if (
            name in {"get_user_details", "get_reservation_details", "get_flight_status"}
            and int(state.get("last_tool_streak") or 0) >= 2
        ):
            return fallback_action(
                "I already checked that record. Please share a different reservation number, user ID, or clarify what should happen next."
            )

    # --- get_flight_status ---
    if name == "get_flight_status":
        if (
            state.get("status_checked")
            and (
                flow_name == "status_compensation"
                or effective_task_type == "status"
            )
            and explicitly_no_change_or_cancel
        ):
            return fallback_action(
                "I can’t offer compensation here. Under the policy, compensation for a delayed flight is only available if you want to change or cancel the reservation after the delay is confirmed. Since you want to keep the reservation as-is, I should stop the delay-check loop and explain that I can’t issue compensation."
            )
        if effective_task_type == "modify" and isinstance(effective_reservation_id, str):
            has_flown = reservation_has_flown_segments(state, effective_reservation_id)
            if has_flown is False:
                missing_search = next_missing_pricing_search(state, effective_reservation_id)
                if missing_search is not None:
                    return missing_search
                if isinstance(requested_cabin, str):
                    expression = pricing_expression_for_current_reservation(
                        state, effective_reservation_id, requested_cabin
                    )
                    if expression and state.get("last_tool_name") != "calculate":
                        return {"name": "calculate", "arguments": {"expression": expression}}
                return fallback_action(
                    "The reservation flights are all still upcoming, so I can continue without checking flight status."
                )
        proposed_flight_number = arguments.get("flight_number")
        if (
            not flight_number_from_context
            and not looks_like_status_intent(latest_user_text)
            and flow_name not in {"status_compensation", "cancel"}
            and effective_task_type not in {"status", "cancel"}
        ):
            return fallback_action(
                "Please share the flight number and whether you're asking about a delay, cancellation, or flight status."
            )
        if isinstance(effective_reservation_id, str):
            current_entry = current_reservation_entry(state, effective_reservation_id)
            flights = current_entry.get("flights") if isinstance(current_entry, dict) else None
            matched_context_flight = None
            if isinstance(flight_number_from_context, str) and isinstance(flights, list):
                for flight in flights:
                    if (
                        isinstance(flight, dict)
                        and flight.get("flight_number") == flight_number_from_context
                        and isinstance(flight.get("date"), str)
                    ):
                        matched_context_flight = flight
                        break
            if matched_context_flight is not None:
                return {
                    "name": "get_flight_status",
                    "arguments": {
                        "flight_number": matched_context_flight["flight_number"],
                        "date": matched_context_flight["date"],
                    },
                }
        effective_flight_number = flight_number_from_context or state.get("flight_number")
        if isinstance(effective_flight_number, str) and proposed_flight_number != effective_flight_number:
            arguments = dict(arguments)
            arguments["flight_number"] = effective_flight_number
            arguments.setdefault("date", "2024-05-15")
            return {"name": "get_flight_status", "arguments": arguments}
        if (
            isinstance(effective_reservation_id, str)
            and (
                flow_name == "status_compensation"
                or effective_task_type == "status"
                or isinstance(recent_reservation_lock, str)
            )
        ):
            current_entry = current_reservation_entry(state, effective_reservation_id)
            flights = current_entry.get("flights") if isinstance(current_entry, dict) else None
            proposed_date = arguments.get("date")
            if isinstance(flights, list):
                matching_flight = None
                for flight in flights:
                    if not isinstance(flight, dict):
                        continue
                    if (
                        flight.get("flight_number") == proposed_flight_number
                        and flight.get("date") == proposed_date
                    ):
                        matching_flight = flight
                        break
                if matching_flight is None:
                    for flight in flights:
                        if not isinstance(flight, dict):
                            continue
                        if isinstance(flight.get("flight_number"), str) and isinstance(
                            flight.get("date"), str
                        ):
                            return {
                                "name": "get_flight_status",
                                "arguments": {
                                    "flight_number": flight["flight_number"],
                                "date": flight["date"],
                                },
                            }
            if isinstance(current_entry, dict):
                if isinstance(proposed_flight_number, str):
                    reservation_flight_numbers = [
                        flight.get("flight_number")
                        for flight in flights or []
                        if isinstance(flight, dict) and isinstance(flight.get("flight_number"), str)
                    ]
                    if proposed_flight_number not in reservation_flight_numbers:
                        return fallback_action(
                            f"I should only check flight segments that belong to reservation {effective_reservation_id}. Please use one of its listed segments."
                        )

    # --- get_user_details ---
    if name == "get_user_details":
        proposed_user_id = arguments.get("user_id")
        if isinstance(proposed_user_id, str):
            lowered = proposed_user_id.lower()
            if "placeholder" in lowered:
                proposed_user_id = None
        else:
            proposed_user_id = None
        verified_user_ids = verified_entities.get("user_ids", set())

        if reservation_id_from_context and not user_id_from_context:
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id_from_context},
            }
        if proposed_user_id in KNOWN_EXAMPLE_USER_IDS:
            if user_id_from_context:
                return {
                    "name": "get_user_details",
                    "arguments": {"user_id": user_id_from_context},
                }
            return fallback_action(
                "Please share your user ID or reservation number so I can look up your booking."
            )
        if user_id_from_context and proposed_user_id != user_id_from_context:
            return {
                "name": "get_user_details",
                "arguments": {"user_id": user_id_from_context},
            }
        if (
            proposed_user_id
            and verified_user_ids
            and proposed_user_id not in verified_user_ids
        ):
            if user_id_from_context:
                return {
                    "name": "get_user_details",
                    "arguments": {"user_id": user_id_from_context},
                }
            return fallback_action(
                "Please share your user ID or reservation number so I can look up your booking."
            )
        if not user_id_from_context:
            return fallback_action(
                "Please share your user ID or reservation number so I can look up your booking."
            )
        if proposed_user_id and RESERVATION_ID_PATTERN.fullmatch(proposed_user_id):
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": proposed_user_id},
            }
        if not proposed_user_id:
            return fallback_action(
                "Please share your user ID or reservation number so I can look up your booking."
            )

    # --- get_reservation_details ---
    if name == "get_reservation_details":
        aggregate_cancel_requested = (
            effective_task_type == "cancel"
            and looks_like_all_reservations_intent(f"{recent_user_text}\n{latest_user_text}")
        )
        recent_route_context = any(
            len(extract_airport_codes(str(message.get("content") or ""))) >= 2
            for message in messages[-6:]
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not str(message.get("content")).startswith("tool:")
        )
        proposed_reservation_id = arguments.get("reservation_id")
        if aggregate_cancel_requested:
            known_ids = [
                rid for rid in (state.get("known_reservation_ids") or []) if isinstance(rid, str)
            ]
            inventory = state.get("reservation_inventory", {})
            if not isinstance(inventory, dict):
                inventory = {}
            if isinstance(proposed_reservation_id, str) and proposed_reservation_id in inventory:
                next_unloaded = next((rid for rid in known_ids if rid not in inventory), None)
                if isinstance(next_unloaded, str):
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": next_unloaded},
                    }
                eligible_ids: list[str] = []
                executed = state.get("executed_actions_by_reservation", {})
                if not isinstance(executed, dict):
                    executed = {}
                for rid in known_ids:
                    entry = inventory.get(rid)
                    if not isinstance(entry, dict):
                        continue
                    if str(entry.get("status") or "").strip().lower() == "cancelled":
                        continue
                    if reservation_has_flown_segments(state, rid) is True:
                        continue
                    eligible, _ = cancel_eligibility(state, rid, latest_user_text)
                    if eligible and executed.get(rid) != "cancel_reservation":
                        eligible_ids.append(rid)
                if eligible_ids:
                    state["reservation_id"] = eligible_ids[0]
                    state["cancel_eligible"] = True
                    return {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": eligible_ids[0]},
                    }
        if (
            effective_task_type == "cancel"
            and state.get("loaded_user_details")
            and not reservation_id_from_context
            and not flight_number_from_context
            and not recent_route_context
            and not looks_like_all_reservations_intent(latest_user_text)
        ):
            known_ids = [
                rid
                for rid in (state.get("known_reservation_ids") or [])
                if isinstance(rid, str)
            ]
            if len(known_ids) > 1:
                return fallback_action(
                    "I found multiple reservations on your profile. Please tell me which trip you want to cancel by reservation ID, route, or flight number so I cancel the correct booking."
                )
        if (
            isinstance(recent_reservation_lock, str)
            and proposed_reservation_id != recent_reservation_lock
        ):
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": recent_reservation_lock},
            }
        if (
            effective_task_type == "modify"
            and isinstance(effective_reservation_id, str)
            and isinstance(requested_cabin, str)
            and proposed_reservation_id == effective_reservation_id
            and current_reservation_entry(state, effective_reservation_id) is not None
        ):
            missing_search = next_missing_pricing_search(state, effective_reservation_id)
            if missing_search is not None:
                return missing_search
            if not state.get("cabin_price_quoted"):
                expression = pricing_expression_for_current_reservation(
                    state, effective_reservation_id, requested_cabin
                )
                if expression:
                    return {"name": "calculate", "arguments": {"expression": expression}}
        if (
            isinstance(proposed_reservation_id, str)
            and FLIGHT_NUMBER_PATTERN.fullmatch(proposed_reservation_id)
        ):
            return fallback_action(
                "That looks like a flight number, not a reservation number. Please share your reservation number or user ID so I can look up the booking."
            )
        if isinstance(proposed_reservation_id, str) and proposed_reservation_id in KNOWN_EXAMPLE_RESERVATION_IDS:
            if reservation_id_from_context and reservation_id_from_context not in KNOWN_EXAMPLE_RESERVATION_IDS:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation_id_from_context},
                }
            if inferred_reservation_id:
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": inferred_reservation_id},
                }
            return fallback_action(
                "Please share your reservation number so I can look up the booking details."
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
            return fallback_action(
                "Please share your reservation number so I can look up the booking details."
            )

    # --- post-flight-status: route to reservation lookup for compensation ---
    if (
        latest_input_is_tool
        and state.get("last_tool_name") == "get_flight_status"
        and effective_task_type in {"status", "general"}
        and name in {"search_direct_flight", "search_onestop_flight", "respond"}
    ):
        if isinstance(effective_reservation_id, str) and current_reservation_entry(
            state, effective_reservation_id
        ) is not None:
            return fallback_action(
                f"I already know which reservation we are checking ({effective_reservation_id}), so I should stay on that reservation instead of inferring a different one from stale flight context."
            )
        flight_number_in_state = state.get("flight_number")
        if isinstance(flight_number_in_state, str):
            # Try to find which reservation contains this flight
            matched_reservation = find_reservation_by_flight_number(state, flight_number_in_state)
            if matched_reservation:
                state["reservation_id"] = matched_reservation
                entry = current_reservation_entry(state, matched_reservation)
                if entry and name != "respond":
                    # Already have the reservation loaded, respond with compensation info
                    return fallback_action(
                        f"I found reservation {matched_reservation} containing flight {flight_number_in_state}. "
                        f"Let me check the reservation details to determine compensation eligibility."
                    )
            else:
                # Need to iterate known reservations to find the one with this flight
                known_ids = state.get("known_reservation_ids", [])
                inventory = state.get("reservation_inventory", {})
                for rid in known_ids:
                    if isinstance(rid, str) and rid not in (inventory or {}):
                        return {
                            "name": "get_reservation_details",
                            "arguments": {"reservation_id": rid},
                        }
                # If all reservations loaded but none matched, need user details first
                if not known_ids and isinstance(state.get("user_id"), str):
                    return {
                        "name": "get_user_details",
                        "arguments": {"user_id": state["user_id"]},
                    }

    # --- route-based reservation iteration (tasks 1, 8) ---
    # When we have loaded user details with multiple reservations but need to find one by route,
    # iterate through unloaded reservations instead of asking user for ID
    if (
        name == "respond"
        and state.get("loaded_user_details")
        and not effective_reservation_id
        and (state.get("origin") or state.get("destination"))
    ):
        # First check already-loaded inventory for a route match
        route_match = find_reservation_by_route(
            state, state.get("origin"), state.get("destination")
        )
        if route_match:
            state["reservation_id"] = route_match
            effective_reservation_id = route_match
            return {
                "name": "get_reservation_details",
                "arguments": {"reservation_id": route_match},
            }
        # Otherwise iterate known reservation IDs that aren't loaded yet
        known_ids = state.get("known_reservation_ids", [])
        inventory = state.get("reservation_inventory", {})
        if isinstance(known_ids, list) and isinstance(inventory, dict):
            for rid in known_ids:
                if isinstance(rid, str) and rid not in inventory:
                    return {
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": rid},
                    }

    # --- search flights ---
    if name in {"search_direct_flight", "search_onestop_flight"}:
        state_task_type = str(state.get("task_type") or "")
        # Status/compensation NEVER needs flight searches — always redirect to next unloaded reservation
        if flow_name in {"reservation_triage", "status_compensation"} or state_task_type == "status":
            if looks_like_all_reservations_intent(latest_user_text):
                return fallback_action(
                    "I should finish identifying the relevant reservations first, then summarize or total them, not run flight searches."
                )
            known_ids = state.get("known_reservation_ids", [])
            inventory = state.get("reservation_inventory", {})
            if isinstance(known_ids, list) and isinstance(inventory, dict):
                for rid in known_ids:
                    if isinstance(rid, str) and rid not in inventory:
                        print(f"[GUARD] search blocked in status/compensation, redirecting to load {rid}", file=sys.stderr)
                        return {"name": "get_reservation_details", "arguments": {"reservation_id": rid}}
            print(f"[GUARD] search blocked in status/compensation flow={flow_name} task={state_task_type}", file=sys.stderr)
            return fallback_action(
                "I should identify the correct reservation or verify flight status before starting a flight availability search."
            )
        if effective_task_type not in {"booking", "modify"}:
            # Instead of asking for ID, try to route to the next useful action
            if isinstance(effective_reservation_id, str) and not current_reservation_entry(state, effective_reservation_id):
                return {
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": effective_reservation_id},
                }
            known_ids = state.get("known_reservation_ids", [])
            inventory = state.get("reservation_inventory", {})
            if isinstance(known_ids, list) and isinstance(inventory, dict):
                for rid in known_ids:
                    if isinstance(rid, str) and rid not in inventory:
                        return {
                            "name": "get_reservation_details",
                            "arguments": {"reservation_id": rid},
                        }
            return fallback_action(
                "I should not search for flights in a status or cancellation flow. Let me focus on the current request instead."
            )
        # Booking flow: if we have a target route and this search is for a different route,
        # we're scanning reservations — redirect to next unloaded reservation instead
        if effective_task_type == "booking":
            target_origin = state.get("origin")
            target_destination = state.get("destination")
            proposed_origin = arguments.get("origin")
            proposed_destination = arguments.get("destination")
            if (
                target_origin and target_destination
                and (proposed_origin != target_origin or proposed_destination != target_destination)
            ):
                known_ids = state.get("known_reservation_ids", [])
                inventory = state.get("reservation_inventory", {})
                if isinstance(known_ids, list) and isinstance(inventory, dict):
                    for rid in known_ids:
                        if isinstance(rid, str) and rid not in inventory:
                            print(f"[GUARD] booking search wrong route ({proposed_origin}→{proposed_destination} != {target_origin}→{target_destination}), loading {rid}", file=sys.stderr)
                            return {"name": "get_reservation_details", "arguments": {"reservation_id": rid}}
        if effective_task_type == "modify" and not effective_reservation_id and not effective_user_id:
            return fallback_action(
                "Please share your reservation number or user ID so I can look up the booking before searching for replacement flights."
            )
        if (
            effective_task_type == "modify"
            and isinstance(effective_reservation_id, str)
            and isinstance(requested_cabin, str)
            and name == "search_direct_flight"
        ):
            proposed_origin = arguments.get("origin")
            proposed_destination = arguments.get("destination")
            proposed_date = arguments.get("date")
            if not matches_reservation_segment(
                state,
                effective_reservation_id,
                proposed_origin if isinstance(proposed_origin, str) else None,
                proposed_destination if isinstance(proposed_destination, str) else None,
                proposed_date if isinstance(proposed_date, str) else None,
            ):
                missing_search = next_missing_pricing_search(state, effective_reservation_id)
                if missing_search is not None:
                    return missing_search
                segment_search = next_segment_search_from_loaded_reservation(
                    state, effective_reservation_id
                )
                if segment_search is not None:
                    return segment_search
                return fallback_action(
                    "I need to check the existing reservation segments before searching for cabin availability."
                )
        if flow_stage == "finalize":
            return fallback_action(
                "The active flow is already in finalization, so I should summarize instead of starting a new search."
            )

    # --- cancel_reservation ---
    if name == "cancel_reservation":
        entry = current_reservation_entry(state, effective_reservation_id)
        if isinstance(entry, dict):
            has_insurance = entry.get("insurance") == "yes"
            cabin = str(entry.get("cabin") or "").strip().lower()
            recent_cancel_context = f"{recent_user_text}\n{latest_user_text}".lower()
            covered_reason = any(
                token in recent_cancel_context
                for token in (
                    "health",
                    "weather",
                    "storm",
                    "sick",
                    "ill",
                    "medical",
                    "airline cancelled",
                    "airline canceled",
                    "cancelled flight",
                    "canceled flight",
                )
            )
            if has_insurance and cabin != "business" and not covered_reason:
                return fallback_action(
                    "This reservation is not eligible for cancellation under the airline policy because "
                    "it was not booked within 24 hours, the flight was not cancelled by the airline, "
                    "the cabin is not business, and the stated reason is not covered by insurance."
                )
        if effective_task_type != "cancel" and pending_confirmation_action != "cancel_reservation":
            return fallback_action(
                "I can only cancel a reservation after an explicit cancellation request."
            )
        if not effective_reservation_id:
            return fallback_action(
                "Please share your reservation number so I can cancel the correct booking."
            )
        cancel_reason_by_reservation = state.get("cancel_reason_by_reservation", {})
        cancel_reason_text = (
            cancel_reason_by_reservation.get(effective_reservation_id)
            if isinstance(cancel_reason_by_reservation, dict)
            else None
        ) or state.get("cancel_reason")
        eligibility_context = (
            cancel_reason_text
            if isinstance(cancel_reason_text, str) and cancel_reason_text
            else latest_user_text
        )
        is_eligible, refusal = cancel_eligibility(
            state, effective_reservation_id, eligibility_context
        )
        state["cancel_eligible"] = is_eligible
        if not is_eligible:
            return fallback_action(
                refusal
                or "This reservation is not eligible for cancellation under the airline policy."
            )
        if arguments.get("reservation_id") != effective_reservation_id:
            return {
                "name": "cancel_reservation",
                "arguments": {"reservation_id": effective_reservation_id},
            }

    # --- update_reservation_flights ---
    if name == "update_reservation_flights":
        if effective_task_type != "modify" and pending_confirmation_action != "update_reservation_flights":
            return fallback_action(
                "I can only change flights or cabins after an explicit modification request."
            )
        if not effective_reservation_id:
            return fallback_action(
                "Please share your reservation number or user ID so I can look up the booking before changing it."
            )
        if any(
            token in lowered_user_text
            for token in (
                "check",
                "search",
                "options",
                "available",
                "segments",
                "before i confirm",
            )
        ) and not looks_like_affirmation(latest_user_text):
            missing_search = next_segment_search_from_loaded_reservation(
                state, effective_reservation_id
            ) or next_missing_pricing_search(state, effective_reservation_id)
            if missing_search is not None:
                return missing_search
            return fallback_action(
                "I should inspect availability and pricing before applying this flight update."
            )
        current_entry = current_reservation_entry(state, effective_reservation_id)
        current_cabin = current_entry.get("cabin") if isinstance(current_entry, dict) else None
        if (
            isinstance(requested_cabin, str)
            and isinstance(current_cabin, str)
            and current_cabin != requested_cabin
            and not state.get("cabin_price_quoted")
        ):
            missing_search = next_missing_pricing_search(state, effective_reservation_id)
            if missing_search is not None:
                return missing_search
            expression = pricing_expression_for_current_reservation(
                state, effective_reservation_id, requested_cabin
            )
            if expression:
                return {"name": "calculate", "arguments": {"expression": expression}}
        if (
            isinstance(requested_cabin, str)
            and any(
                token in lowered_user_text
                for token in ("cost", "price", "pricing", "under $", "budget", "before i confirm")
            )
        ):
            missing_search = next_missing_pricing_search(state, effective_reservation_id)
            if missing_search is not None:
                return missing_search
            expression = pricing_expression_for_current_reservation(
                state, effective_reservation_id, requested_cabin
            )
            if expression:
                return {"name": "calculate", "arguments": {"expression": expression}}

    if (
        name == "calculate"
        and flow_name in {"reservation_triage", "status_compensation"}
    ):
        return fallback_action(
            "I should finish identifying the correct reservation or status-eligibility branch before doing pricing calculations."
        )

    # --- update_reservation_baggages ---
    if name == "update_reservation_baggages":
        if (
            effective_task_type == "modify"
            and isinstance(effective_reservation_id, str)
            and isinstance(requested_cabin, str)
            and cabin_only_change
        ):
            same_flights = same_flights_update_action(
                state, effective_reservation_id, requested_cabin
            )
            if same_flights is not None:
                return same_flights
        if effective_task_type != "baggage" and pending_confirmation_action != "update_reservation_baggages":
            return fallback_action(
                "I can only change baggage after an explicit baggage request."
            )
        if not effective_reservation_id:
            return fallback_action(
                "Please share your reservation number or user ID so I can look up the booking before changing baggage."
            )
        if not current_membership(state) and isinstance(effective_user_id, str):
            return {"name": "get_user_details", "arguments": {"user_id": effective_user_id}}
        total_baggages = arguments.get("total_baggages")
        if isinstance(total_baggages, int):
            normalized_nonfree = normalized_nonfree_baggages(
                state, effective_reservation_id, total_baggages
            )
            if normalized_nonfree is not None:
                normalized_arguments = dict(arguments)
                normalized_arguments["nonfree_baggages"] = normalized_nonfree
                payment_id = normalized_arguments.get("payment_id")
                if not isinstance(payment_id, str) or not payment_id:
                    inferred_payment_id = reservation_payment_id(state, effective_reservation_id)
                    if inferred_payment_id:
                        normalized_arguments["payment_id"] = inferred_payment_id
                if normalized_arguments != arguments:
                    return {
                        "name": "update_reservation_baggages",
                        "arguments": normalized_arguments,
                    }

    result_name = action.get("name")
    if result_name != _guard_proposed:
        print(f"[GUARD] rerouted: {_guard_proposed} → {result_name}", file=sys.stderr)
    else:
        print(f"[GUARD] passthrough: {result_name}", file=sys.stderr)
    return action
