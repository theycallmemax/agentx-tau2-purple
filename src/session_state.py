"""Session state management: tracking, updating, and summarizing conversation state."""

from __future__ import annotations

import copy
import json
from typing import Any

try:
    from .extractors import (
        AIRPORT_CODE_PATTERN,
        dedupe_keep_order,
        extract_airport_codes,
        extract_cancel_reason,
        extract_dates,
        extract_flight_number,
        extract_requested_cabin,
        extract_reservation_id,
        extract_user_id,
    )
    from .intent import (
        classify_task_type,
        looks_like_affirmation,
        looks_like_social_pressure,
        looks_like_stop_without_changes,
    )
except ImportError:  # pragma: no cover - direct src imports in tests
    from extractors import (
        AIRPORT_CODE_PATTERN,
        dedupe_keep_order,
        extract_airport_codes,
        extract_cancel_reason,
        extract_dates,
        extract_flight_number,
        extract_requested_cabin,
        extract_reservation_id,
        extract_user_id,
    )
    from intent import (
        classify_task_type,
        looks_like_affirmation,
        looks_like_social_pressure,
        looks_like_stop_without_changes,
    )


def create_initial_state() -> dict[str, Any]:
    return {
        "user_id": None,
        "reservation_id": None,
        "flight_number": None,
        "origin": None,
        "destination": None,
        "travel_dates": [],
        "known_reservation_ids": [],
        "known_payment_ids": [],
        "known_payment_balances": {},
        "known_saved_passengers": [],
        "loaded_user_details": False,
        "loaded_reservation_details": False,
        "last_tool_name": None,
        "last_tool_arguments": None,
        "last_tool_user_text": None,
        "last_tool_streak": 0,
        "reservation_inventory": {},
        "reservation_baseline_inventory": {},
        "flight_search_inventory": {},
        "flight_search_inventory_by_mode": {},
        "membership": None,
        "pending_confirmation_action": None,
        "pending_confirmation_by_reservation": {},
        "confirmed_actions_by_reservation": {},
        "executed_actions_by_reservation": {},
        "requested_cabin": None,
        "cabin_only_change": False,
        "cabin_price_quoted": False,
        "last_calculation_result": None,
        "cancel_reason": None,
        "cancel_reason_by_reservation": {},
        "cancel_eligible": None,
        "compensation_ineligible": False,
        "status_checked": False,
        "last_flight_status_result": None,
        "reservation_identification_incomplete": False,
        "route_resolution_active": False,
        "route_resolution_target": None,
        "route_resolution_queue": [],
        "recent_reservation_active": False,
        "recent_reservation_id": None,
        "nonstop_unavailable": False,
        "reservation_locked": False,
        "already_cancelled_this_turn": False,
        "already_summarized_write_result": False,
        "already_resolved_subtask_B": False,
        "awaiting_choice_between_options": False,
        "awaiting_flight_selection": False,
        "awaiting_payment_choice": False,
        "flow_version": 0,
        "pricing_context_version": 0,
        "last_tool_error": None,
        "last_error_key": None,
        "error_streak": 0,
        "latest_input_is_tool": False,
        "terminal_user_intent": None,
        "verified_facts_cache": {},
        "active_flow": {"name": "general", "stage": "analyze"},
        "subtask_queue": [],
        "completed_subtasks": [],
        "completed_subtask_ids": [],
        "completed_actions": [],
        "raw_facts": {},
        "working_memory": {},
        "derived_facts": {},
        "history_summary": None,
        "history_compression_summary": None,
        "task_type": "general",
    }


def merge_state_lists(state: dict[str, Any], key: str, values: list[str]) -> None:
    existing = state.get(key, [])
    if not isinstance(existing, list):
        existing = []
    state[key] = dedupe_keep_order(existing + values)


def clear_modify_flow_state(state: dict[str, Any]) -> None:
    state["requested_cabin"] = None
    state["cabin_only_change"] = False
    state["cabin_price_quoted"] = False
    state["last_calculation_result"] = None
    state["awaiting_choice_between_options"] = False
    state["awaiting_flight_selection"] = False
    if state.get("pending_confirmation_action") == "update_reservation_flights":
        state["pending_confirmation_action"] = None


def clear_cancel_flow_state(state: dict[str, Any]) -> None:
    state["cancel_reason"] = None
    state["cancel_eligible"] = None
    state["already_cancelled_this_turn"] = False
    if state.get("pending_confirmation_action") == "cancel_reservation":
        state["pending_confirmation_action"] = None


def clear_adversarial_compensation_state(state: dict[str, Any]) -> None:
    working = state.get("working_memory")
    if not isinstance(working, dict):
        working = {}
    working["adversarial_compensation_verification"] = False
    working["adversarial_compensation_business_claim"] = False
    working["adversarial_compensation_cancelled_claim"] = False
    state["working_memory"] = working


def update_state_from_text(state: dict[str, Any], text: str) -> None:
    if text.startswith("tool:"):
        return
    marker = "User message:"
    if marker in text:
        embedded = text.rsplit(marker, 1)[1].strip()
        try:
            parsed = json.loads(embedded)
            if isinstance(parsed, str):
                text = parsed
            else:
                text = embedded
        except Exception:
            quoted = embedded.strip()
            if len(quoted) >= 2 and quoted[0] == quoted[-1] == '"':
                try:
                    text = json.loads(quoted)
                except Exception:
                    text = embedded
            else:
                text = embedded
    elif text.lower().startswith("user:"):
        text = text.split(":", 1)[1].strip()
    latest_task_type = classify_task_type(text)
    if latest_task_type != "general":
        state["task_type"] = latest_task_type
        if latest_task_type != "modify":
            clear_modify_flow_state(state)
        if latest_task_type != "cancel":
            clear_cancel_flow_state(state)
        if latest_task_type != "status":
            clear_adversarial_compensation_state(state)
    requested_cabin = extract_requested_cabin(text)
    if requested_cabin and requested_cabin != state.get("requested_cabin"):
        state["cabin_price_quoted"] = False
        state["last_calculation_result"] = None
    if requested_cabin:
        state["requested_cabin"] = requested_cabin
    cancel_reason = extract_cancel_reason(text)
    if cancel_reason:
        state["cancel_reason"] = cancel_reason

    user_id = extract_user_id(text)
    reservation_id = extract_reservation_id(text)
    flight_number = extract_flight_number(text)
    dates = extract_dates(text)
    airport_codes = extract_airport_codes(text)

    if user_id:
        state["user_id"] = user_id
    if reservation_id:
        state["reservation_id"] = reservation_id
        merge_state_lists(state, "known_reservation_ids", [reservation_id])
        state["route_resolution_active"] = False
        state["route_resolution_target"] = None
        state["route_resolution_queue"] = []
        state["recent_reservation_active"] = False
        state["recent_reservation_id"] = None
        if cancel_reason:
            cancel_reason_by_reservation = state.get("cancel_reason_by_reservation", {})
            if not isinstance(cancel_reason_by_reservation, dict):
                cancel_reason_by_reservation = {}
            cancel_reason_by_reservation[reservation_id] = cancel_reason
            state["cancel_reason_by_reservation"] = cancel_reason_by_reservation
    if flight_number:
        state["flight_number"] = flight_number
    if dates:
        merge_state_lists(state, "travel_dates", dates)
    if len(airport_codes) >= 2:
        state["origin"] = airport_codes[0]
        state["destination"] = airport_codes[1]

    if requested_cabin and not dates and len(airport_codes) < 2:
        state["cabin_only_change"] = True
    elif not looks_like_affirmation(text) and (dates or len(airport_codes) >= 2):
        state["cabin_only_change"] = False

    lowered = text.lower()
    aggregate_summary_request = any(
        phrase in lowered
        for phrase in (
            "all upcoming",
            "include all",
            "total cost of those upcoming flights",
            "total cost of my upcoming flights",
            "let me know the total cost",
            "how much are my upcoming flights",
        )
    )
    keep_current_plan = any(
        phrase in lowered
        for phrase in (
            "keep it as-is",
            "keep it as is",
            "keep the current flights",
            "keep it as currently booked",
            "remain as-is",
            "leave it as-is",
            "leave it as is",
        )
    )
    if aggregate_summary_request or keep_current_plan:
        clear_modify_flow_state(state)
    if "don't" in lowered or "do not" in lowered or "instead" in lowered:
        state["pending_confirmation_action"] = None
    decline_upgrade = any(
        phrase in lowered
        for phrase in (
            "can't proceed",
            "cannot proceed",
            "won't proceed",
            "will not proceed",
            "too expensive",
            "over my budget",
            "keep the current",
            "keep my current",
        )
    ) or ("over $" in lowered)
    baggage_only_followup = (
        "add" in lowered and "bag" in lowered and "upgrade" not in lowered and "business" not in lowered
    )
    if decline_upgrade or baggage_only_followup:
        clear_modify_flow_state(state)

    if looks_like_stop_without_changes(text):
        state["terminal_user_intent"] = "stop_without_changes"
    else:
        # Treat stop-without-changes as a single-turn instruction. If the user asks a
        # fresh follow-up question afterwards, the sticky terminal flag must clear so
        # we do not keep auto-closing the conversation on unrelated turns.
        state["terminal_user_intent"] = None

    if looks_like_social_pressure(text):
        derived = state.get("derived_facts", {})
        if not isinstance(derived, dict):
            derived = {}
        derived["social_pressure_detected"] = True
        state["derived_facts"] = derived

    if looks_like_affirmation(text):
        return
    if any(
        phrase in lowered
        for phrase in (
            "which option",
            "which one",
            "choose",
            "select one",
        )
    ):
        state["awaiting_choice_between_options"] = True
    if any(
        phrase in lowered
        for phrase in (
            "which flight",
            "which segment",
            "flight selection",
        )
    ):
        state["awaiting_flight_selection"] = True
    if any(
        phrase in lowered
        for phrase in (
            "which payment",
            "payment method",
            "which card",
        )
    ):
        state["awaiting_payment_choice"] = True


def update_state_from_tool_payload(state: dict[str, Any], input_text: str) -> None:
    if not input_text.startswith("tool:"):
        return
    payload_text = input_text.split("tool:", 1)[1].strip()
    if payload_text.lower().startswith("error:"):
        error_key = f"{state.get('last_tool_name')}|{json.dumps(state.get('last_tool_arguments') or {}, sort_keys=True, ensure_ascii=False)}"
        if state.get("last_error_key") == error_key:
            state["error_streak"] = int(state.get("error_streak") or 0) + 1
        else:
            state["error_streak"] = 1
        state["last_error_key"] = error_key
        state["last_tool_error"] = payload_text
        state["reservation_identification_incomplete"] = True
        return
    try:
        payload = json.loads(payload_text)
    except Exception:
        if state.get("last_tool_name") == "get_flight_status":
            state["status_checked"] = True
            state["last_flight_status_result"] = payload_text.strip()
        _reconcile_successful_write_tool(state, payload_text)
        return
    state["last_tool_error"] = None
    state["last_error_key"] = None
    state["error_streak"] = 0
    if state.get("last_tool_name") == "get_flight_status":
        state["status_checked"] = True
        # Save the flight number that was checked so the loop guard can use it
        # even after last_tool_name is changed by a subsequent respond action
        last_args = state.get("last_tool_arguments") or {}
        if isinstance(last_args, dict) and last_args.get("flight_number"):
            state["_last_flight_status_checked_flight"] = last_args["flight_number"]
        if isinstance(payload, dict):
            status_value = payload.get("status")
            if isinstance(status_value, str) and status_value:
                state["last_flight_status_result"] = status_value
    if (
        isinstance(payload, (int, float))
        and state.get("last_tool_name") == "calculate"
        and isinstance(state.get("requested_cabin"), str)
    ):
        state["cabin_price_quoted"] = True
        state["last_calculation_result"] = float(payload)
        state["awaiting_choice_between_options"] = False
        state["pricing_context_version"] = int(state.get("flow_version") or 0)
        return
    if isinstance(payload, list):
        last_tool_name = state.get("last_tool_name")
        last_tool_arguments = state.get("last_tool_arguments")
        if last_tool_name not in {"search_direct_flight", "search_onestop_flight"}:
            return
        if not isinstance(last_tool_arguments, dict):
            return
        origin = last_tool_arguments.get("origin")
        destination = last_tool_arguments.get("destination")
        date = last_tool_arguments.get("date")
        if not all(isinstance(value, str) and value for value in (origin, destination, date)):
            return
        if not all(isinstance(item, dict) for item in payload):
            return
        inventory = state.get("flight_search_inventory", {})
        if not isinstance(inventory, dict):
            inventory = {}
        inventory[f"{origin}|{destination}|{date}"] = payload
        state["flight_search_inventory"] = inventory
        inventory_by_mode = state.get("flight_search_inventory_by_mode", {})
        if not isinstance(inventory_by_mode, dict):
            inventory_by_mode = {}
        key = f"{origin}|{destination}|{date}"
        bucket = inventory_by_mode.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
        bucket[str(last_tool_name)] = payload
        inventory_by_mode[key] = bucket
        state["flight_search_inventory_by_mode"] = inventory_by_mode
        if not payload and state.get("task_type") == "modify":
            state["nonstop_unavailable"] = True
        if payload and state.get("task_type") == "status":
            state["reservation_identification_incomplete"] = False
        return
    if not isinstance(payload, dict):
        return

    user_id = payload.get("user_id")
    reservation_id = payload.get("reservation_id")
    flight_number = payload.get("flight_number")
    origin = payload.get("origin")
    destination = payload.get("destination")
    membership = payload.get("membership")

    if isinstance(user_id, str) and user_id:
        state["user_id"] = user_id
        state["loaded_user_details"] = True
    if isinstance(membership, str) and membership:
        state["membership"] = membership
    if isinstance(reservation_id, str) and reservation_id:
        state["reservation_id"] = reservation_id
        state["loaded_reservation_details"] = True
        merge_state_lists(state, "known_reservation_ids", [reservation_id])
        state["reservation_identification_incomplete"] = False
    if isinstance(flight_number, str) and flight_number:
        state["flight_number"] = flight_number
    if isinstance(origin, str) and origin:
        state["origin"] = origin
    if isinstance(destination, str) and destination:
        state["destination"] = destination

    reservations = payload.get("reservations")
    if isinstance(reservations, list):
        merge_state_lists(
            state,
            "known_reservation_ids",
            [value for value in reservations if isinstance(value, str)],
        )

    payment_methods = payload.get("payment_methods")
    if isinstance(payment_methods, dict):
        merge_state_lists(
            state,
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
            existing = state.get("known_payment_balances", {})
            if not isinstance(existing, dict):
                existing = {}
            existing.update(balances)
            state["known_payment_balances"] = existing

    saved_passengers = payload.get("saved_passengers")
    if isinstance(saved_passengers, list):
        normalized_saved: list[dict[str, str]] = []
        for passenger in saved_passengers:
            if not isinstance(passenger, dict):
                continue
            first_name = passenger.get("first_name")
            last_name = passenger.get("last_name")
            dob = passenger.get("dob")
            if all(isinstance(value, str) and value for value in (first_name, last_name, dob)):
                normalized_saved.append(
                    {
                        "first_name": first_name,
                        "last_name": last_name,
                        "dob": dob,
                    }
                )
        if normalized_saved:
            state["known_saved_passengers"] = normalized_saved

    payment_history = payload.get("payment_history")
    if isinstance(payment_history, list):
        merge_state_lists(
            state,
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
                state["flight_number"] = flight["flight_number"]
            if isinstance(flight.get("date"), str):
                dates.append(flight["date"])
        if dates:
            merge_state_lists(state, "travel_dates", dates)

    if isinstance(reservation_id, str) and reservation_id:
        inventory = state.get("reservation_inventory", {})
        if not isinstance(inventory, dict):
            inventory = {}
        reservation_snapshot = {
            "origin": origin,
            "destination": destination,
            "dates": dedupe_keep_order(
                [
                    flight.get("date")
                    for flight in flights or []
                    if isinstance(flight, dict) and isinstance(flight.get("date"), str)
                ]
            ),
            "flight_numbers": dedupe_keep_order(
                [
                    flight.get("flight_number")
                    for flight in flights or []
                    if isinstance(flight, dict)
                    and isinstance(flight.get("flight_number"), str)
                ]
            ),
            "flights": [
                {
                    "flight_number": flight.get("flight_number"),
                    "origin": flight.get("origin"),
                    "destination": flight.get("destination"),
                    "date": flight.get("date"),
                    "price": flight.get("price"),
                }
                for flight in flights or []
                if isinstance(flight, dict)
                and isinstance(flight.get("flight_number"), str)
                and isinstance(flight.get("date"), str)
            ],
            "created_at": payload.get("created_at"),
            "status": payload.get("status"),
            "cabin": payload.get("cabin"),
            "insurance": payload.get("insurance"),
            "payment_id": (
                payment_history[0].get("payment_id")
                if isinstance(payment_history, list)
                and payment_history
                and isinstance(payment_history[0], dict)
                and isinstance(payment_history[0].get("payment_id"), str)
                else None
            ),
            "passenger_count": len(payload.get("passengers", []))
            if isinstance(payload.get("passengers"), list)
            else None,
            "total_baggages": payload.get("total_baggages"),
            "nonfree_baggages": payload.get("nonfree_baggages"),
        }
        inventory[reservation_id] = reservation_snapshot
        state["reservation_inventory"] = inventory
        baseline_inventory = state.get("reservation_baseline_inventory", {})
        if not isinstance(baseline_inventory, dict):
            baseline_inventory = {}
        baseline_inventory.setdefault(reservation_id, copy.deepcopy(reservation_snapshot))
        state["reservation_baseline_inventory"] = baseline_inventory
    verified_cache = state.get("verified_facts_cache", {})
    if not isinstance(verified_cache, dict):
        verified_cache = {}
    for key in ("user_id", "reservation_id", "origin", "destination", "membership"):
        value = state.get(key)
        if value:
            verified_cache[key] = value
    state["verified_facts_cache"] = verified_cache


def _reconcile_successful_write_tool(state: dict[str, Any], payload_text: str) -> None:
    lowered = payload_text.lower()
    if not payload_text or "error" in lowered or "failed" in lowered:
        return

    last_tool_name = state.get("last_tool_name")
    last_tool_arguments = state.get("last_tool_arguments")
    if not isinstance(last_tool_arguments, dict):
        return

    if last_tool_name == "cancel_reservation":
        reservation_id = last_tool_arguments.get("reservation_id")
        entry = _mutable_reservation_entry(state, reservation_id)
        if entry is not None:
            entry["status"] = "cancelled"
        state["already_cancelled_this_turn"] = True
        state["already_summarized_write_result"] = False
        state["cabin_price_quoted"] = False
        state["last_calculation_result"] = None
        state["flow_version"] = int(state.get("flow_version") or 0) + 1
        state["reservation_locked"] = True
        return

    if last_tool_name == "update_reservation_baggages":
        reservation_id = last_tool_arguments.get("reservation_id")
        entry = _mutable_reservation_entry(state, reservation_id)
        if entry is None:
            return
        total_baggages = last_tool_arguments.get("total_baggages")
        nonfree_baggages = last_tool_arguments.get("nonfree_baggages")
        if isinstance(total_baggages, int):
            entry["total_baggages"] = total_baggages
        if isinstance(nonfree_baggages, int):
            entry["nonfree_baggages"] = nonfree_baggages
        state["already_summarized_write_result"] = False
        state["flow_version"] = int(state.get("flow_version") or 0) + 1
        state["reservation_locked"] = True
        return

    if last_tool_name == "update_reservation_flights":
        reservation_id = last_tool_arguments.get("reservation_id")
        entry = _mutable_reservation_entry(state, reservation_id)
        if entry is None:
            return
        flights = last_tool_arguments.get("flights")
        cabin = last_tool_arguments.get("cabin")
        if isinstance(cabin, str) and cabin:
            entry["cabin"] = cabin
        if isinstance(flights, list):
            entry["flights"] = [
                {
                    "flight_number": flight.get("flight_number"),
                    "origin": existing.get("origin") if isinstance(existing, dict) else None,
                    "destination": existing.get("destination") if isinstance(existing, dict) else None,
                    "date": flight.get("date"),
                    "price": existing.get("price") if isinstance(existing, dict) else None,
                }
                for flight in flights
                for existing in [_match_existing_flight(entry, flight)]
                if isinstance(flight, dict)
                and isinstance(flight.get("flight_number"), str)
                and isinstance(flight.get("date"), str)
            ]
            entry["flight_numbers"] = dedupe_keep_order(
                [
                    flight["flight_number"]
                    for flight in entry["flights"]
                    if isinstance(flight.get("flight_number"), str)
                ]
            )
            entry["dates"] = dedupe_keep_order(
                [
                    flight["date"]
                    for flight in entry["flights"]
                    if isinstance(flight.get("date"), str)
                ]
            )
            if entry["flights"]:
                first = entry["flights"][0]
                last = entry["flights"][-1]
                if isinstance(first.get("origin"), str):
                    entry["origin"] = first["origin"]
                if isinstance(last.get("destination"), str):
                    entry["destination"] = last["destination"]
        state["already_summarized_write_result"] = False
        state["flow_version"] = int(state.get("flow_version") or 0) + 1
        state["reservation_locked"] = True
        state["cabin_price_quoted"] = False
        state["last_calculation_result"] = None


def _mutable_reservation_entry(
    state: dict[str, Any], reservation_id: str | None
) -> dict[str, Any] | None:
    inventory = state.get("reservation_inventory", {})
    if not isinstance(reservation_id, str) or not isinstance(inventory, dict):
        return None
    entry = inventory.get(reservation_id)
    return entry if isinstance(entry, dict) else None


def _match_existing_flight(
    entry: dict[str, Any], requested_flight: dict[str, Any]
) -> dict[str, Any] | None:
    flights = entry.get("flights")
    if not isinstance(flights, list):
        return None
    requested_number = requested_flight.get("flight_number")
    requested_date = requested_flight.get("date")
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        if (
            flight.get("flight_number") == requested_number
            and flight.get("date") == requested_date
        ):
            return flight
    return None


def build_state_summary(state: dict[str, Any]) -> dict[str, str]:
    inventory = state.get("reservation_inventory", {})
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
        "user_id": state.get("user_id"),
        "reservation_id": state.get("reservation_id"),
        "flight_number": state.get("flight_number"),
        "origin": state.get("origin"),
        "destination": state.get("destination"),
        "travel_dates": state.get("travel_dates", [])[:4],
        "known_reservation_ids": state.get("known_reservation_ids", [])[:6],
        "known_payment_ids": state.get("known_payment_ids", [])[:8],
        "loaded_user_details": state.get("loaded_user_details"),
        "loaded_reservation_details": state.get("loaded_reservation_details"),
        "last_tool_name": state.get("last_tool_name"),
        "last_tool_arguments": state.get("last_tool_arguments"),
        "known_payment_balances": state.get("known_payment_balances", {}),
        "reservation_inventory": inventory_summary,
        "pending_confirmation_action": state.get("pending_confirmation_action"),
        "last_calculation_result": state.get("last_calculation_result"),
        "cancel_reason": state.get("cancel_reason"),
        "cancel_reason_by_reservation": state.get("cancel_reason_by_reservation", {}),
        "cancel_eligible": state.get("cancel_eligible"),
        "compensation_ineligible": state.get("compensation_ineligible"),
        "status_checked": state.get("status_checked"),
        "flow_version": state.get("flow_version"),
        "pricing_context_version": state.get("pricing_context_version"),
        "last_tool_error": state.get("last_tool_error"),
        "error_streak": state.get("error_streak"),
        "terminal_user_intent": state.get("terminal_user_intent"),
        "active_flow": state.get("active_flow"),
        "subtask_queue": state.get("subtask_queue", [])[:4],
        "completed_subtasks": state.get("completed_subtasks", [])[-4:],
        "completed_subtask_ids": state.get("completed_subtask_ids", [])[-8:],
        "completed_actions": state.get("completed_actions", [])[-6:],
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
        "raw_facts": state.get("raw_facts", {}),
        "working_memory": state.get("working_memory", {}),
        "derived_facts": state.get("derived_facts", {}),
        "verified_facts_cache": state.get("verified_facts_cache", {}),
        "history_compression_summary": state.get("history_compression_summary"),
        "task_type": state.get("task_type"),
    }
    return {
        "role": "system",
        "content": (
            "Structured session state for reuse across steps: "
            + json.dumps(summary, ensure_ascii=False)
        ),
    }


def remember_pending_confirmation(state: dict[str, Any], action: dict[str, Any]) -> None:
    if action.get("name") != "respond":
        return
    content = str(action.get("arguments", {}).get("content", "")).lower()
    if not content:
        return
    if "reply" not in content and "confirm" not in content and "proceed" not in content:
        return
    if "cancel" in content:
        state["pending_confirmation_action"] = "cancel_reservation"
        reservation_id = state.get("reservation_id")
        if isinstance(reservation_id, str):
            pending_by_reservation = state.get("pending_confirmation_by_reservation", {})
            if not isinstance(pending_by_reservation, dict):
                pending_by_reservation = {}
            pending_by_reservation[reservation_id] = "cancel_reservation"
            state["pending_confirmation_by_reservation"] = pending_by_reservation
        return
    if "bag" in content:
        state["pending_confirmation_action"] = "update_reservation_baggages"
        return
    if any(word in content for word in ("upgrade", "change", "modify", "cabin")):
        state["pending_confirmation_action"] = "update_reservation_flights"


def clear_pending_confirmation_if_completed(state: dict[str, Any], action: dict[str, Any]) -> None:
    if action.get("name") in {
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
    }:
        state["pending_confirmation_action"] = None
        reservation_id = action.get("arguments", {}).get("reservation_id")
        if isinstance(reservation_id, str):
            executed = state.get("executed_actions_by_reservation", {})
            if not isinstance(executed, dict):
                executed = {}
            executed[reservation_id] = action.get("name")
            state["executed_actions_by_reservation"] = executed
    if action.get("name") == "cancel_reservation":
        state["cancel_reason"] = None
        state["cancel_eligible"] = None
        state["awaiting_choice_between_options"] = False
        reservation_id = action.get("arguments", {}).get("reservation_id")
        if isinstance(reservation_id, str):
            pending_by_reservation = state.get("pending_confirmation_by_reservation", {})
            if isinstance(pending_by_reservation, dict):
                pending_by_reservation.pop(reservation_id, None)
    if action.get("name") == "update_reservation_flights":
        clear_modify_flow_state(state)
