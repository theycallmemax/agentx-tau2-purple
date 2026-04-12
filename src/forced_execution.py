"""Forced Execution Layer — bypass LLM when state is complete for action.

Architecture: When session_state contains all required information for a write action,
this layer forces the action directly without consulting the LLM, saving valuable steps.

Completeness Flags:
- cancel_ready: user_id + reservation_id + reservation_details + eligibility checked
- baggage_ready: user_id + reservation_id + reservation_details + baggage_count_requested
- modify_ready: user_id + reservation_id + reservation_details + target specified
- status_ready: user_id + reservation_id + reservation_details + flight_status checked
"""

from __future__ import annotations

from typing import Any

try:
    from .reservation import (
        cancel_eligibility,
        current_reservation_entry,
        reservation_payment_id,
        normalized_nonfree_baggages,
    )
except ImportError:
    from reservation import (
        cancel_eligibility,
        current_reservation_entry,
        reservation_payment_id,
        normalized_nonfree_baggages,
    )


def should_force_execution(state: dict[str, Any]) -> bool:
    """Check if any completeness flag is set indicating action is ready."""
    # Check completeness flags
    if state.get("cancel_ready"):
        return True
    if state.get("baggage_ready"):
        return True
    if state.get("modify_ready"):
        return True
    if state.get("status_ready"):
        return True
    
    return False


def _mark_completeness_flags(state: dict[str, Any]) -> None:
    """Update completeness flags based on current state.
    
    This should be called during state updates to track when we have
    all required information for each action type.
    """
    task_type = state.get("task_type", "general")
    has_user = state.get("user_id") is not None
    has_reservation = state.get("reservation_id") is not None
    has_reservation_details = state.get("reservation_inventory", {}).get(state.get("reservation_id")) is not None
    
    # Mark cancel_ready
    if task_type == "cancel" and has_user and has_reservation and has_reservation_details:
        entry = current_reservation_entry(state, state["reservation_id"])
        if entry:
            state["cancel_ready"] = True
    
    # Mark baggage_ready
    if task_type == "baggage" and has_user and has_reservation and has_reservation_details:
        if state.get("baggage_count_requested"):
            state["baggage_ready"] = True
    
    # Mark modify_ready  
    if task_type == "modify" and has_user and has_reservation and has_reservation_details:
        if state.get("requested_cabin") or state.get("target_flights"):
            state["modify_ready"] = True
    
    # Mark status_ready
    if task_type == "status" and has_user and has_reservation and has_reservation_details:
        if state.get("last_flight_status"):
            state["status_ready"] = True
    
    # NEW: Mark aggregate_ready when all known reservations are loaded
    # This handles multi-intent tasks where agent needs to check all reservations
    known_ids = state.get("known_reservation_ids", [])
    inventory = state.get("reservation_inventory", {})
    if isinstance(known_ids, list) and len(known_ids) > 0:
        loaded_count = sum(1 for rid in known_ids if rid in inventory)
        total_count = len(known_ids)
        if loaded_count == total_count and total_count > 0:
            state["all_reservations_loaded"] = True
            # Mark all action types as ready since we have complete picture
            state["cancel_ready"] = True
            state["baggage_ready"] = True
            state["modify_ready"] = True
            state["aggregate_ready"] = True


def force_pending_action(state: dict[str, Any]) -> dict[str, Any] | None:
    """Force execute a pending action when all prerequisites are met.
    
    This bypasses the LLM to save steps when the state machine has already
    determined what needs to be done.
    """
    if not should_force_execution(state):
        return None
    
    task_type = state.get("task_type", "general")
    reservation_id = state.get("reservation_id")
    active_flow = state.get("active_flow", {})
    flow_stage = active_flow.get("stage", "unknown") if isinstance(active_flow, dict) else "unknown"
    
    # Rule 1: If we have reservation details loaded and it's a cancel task, check eligibility and act
    if task_type == "cancel" and reservation_id:
        entry = current_reservation_entry(state, reservation_id)
        if entry:  # We have the reservation details
            eligible, refusal = cancel_eligibility(state, reservation_id, "")
            if eligible:
                # Force cancel directly
                return {
                    "name": "cancel_reservation",
                    "arguments": {"reservation_id": reservation_id},
                }
            elif refusal:
                # Policy forbids — respond with refusal
                return {
                    "name": "respond",
                    "arguments": {"content": refusal},
                }
    
    # Rule 2: If modify flow and we have pricing info, force the modify action
    if task_type == "modify" and reservation_id and state.get("pricing_calculated"):
        target_cabin = state.get("requested_cabin")
        if target_cabin and state.get("modify_action_ready"):
            payment_id = reservation_payment_id(state, reservation_id)
            if not payment_id:
                payment_id = state.get("payment_id")
            
            if payment_id:
                return {
                    "name": "update_reservation_flights",
                    "arguments": {
                        "reservation_id": reservation_id,
                        "cabin": target_cabin,
                        "flights": entry.get("flights", []),
                        "payment_id": payment_id,
                    },
                }
    
    # Rule 3: If baggage flow and we have reservation details, force baggage update
    if task_type == "baggage" and reservation_id:
        entry = current_reservation_entry(state, reservation_id)
        if entry and state.get("baggage_count_requested"):
            total_bags = state["baggage_count_requested"]
            user_membership = state.get("membership", "regular")
            cabin = entry.get("cabin", "economy")
            passengers = entry.get("passengers", [])
            passenger_count = len(passengers) if isinstance(passengers, list) else 1
            
            # Calculate free allowance
            free_per_passenger = {
                ("regular", "basic_economy"): 0,
                ("regular", "economy"): 1,
                ("regular", "business"): 2,
                ("silver", "basic_economy"): 1,
                ("silver", "economy"): 2,
                ("silver", "business"): 3,
                ("gold", "basic_economy"): 2,
                ("gold", "economy"): 3,
                ("gold", "business"): 4,
            }.get((user_membership, cabin), 0)
            
            free_total = free_per_passenger * passenger_count
            nonfree = max(0, total_bags - free_total)
            
            payment_id = state.get("payment_id") or reservation_payment_id(state, reservation_id)
            if payment_id:
                return {
                    "name": "update_reservation_baggages",
                    "arguments": {
                        "reservation_id": reservation_id,
                        "total_baggages": total_bags,
                        "nonfree_baggages": nonfree,
                        "payment_id": payment_id,
                    },
                }
    
    # Rule 4: If status flow and we checked flight status, report compensation
    if task_type == "status" and state.get("status_checked") and reservation_id:
        entry = current_reservation_entry(state, reservation_id)
        if entry:
            flight_status = state.get("last_flight_status")
            if flight_status == "delayed":
                user_wants_change = state.get("user_wants_change_or_cancel")
                if user_wants_change:
                    # Compensation available
                    passengers = entry.get("passengers", [])
                    passenger_count = len(passengers) if isinstance(passengers, list) else 1
                    cert_amount = 50 * passenger_count
                    return {
                        "name": "send_certificate",
                        "arguments": {
                            "user_id": state["user_id"],
                            "amount": cert_amount,
                        },
                    }
                else:
                    # User doesn't want change — no compensation per policy
                    return {
                        "name": "respond",
                        "arguments": {
                            "content": "Your flight was delayed. However, under the policy, compensation for a delayed flight is only available if you want to change or cancel the reservation. Since you want to keep the reservation as-is, I cannot offer compensation."
                        },
                    }
            elif flight_status == "cancelled":
                passengers = entry.get("passengers", [])
                passenger_count = len(passengers) if isinstance(passengers, list) else 1
                cert_amount = 100 * passenger_count
                return {
                    "name": "send_certificate",
                    "arguments": {
                        "user_id": state["user_id"],
                        "amount": cert_amount,
                    },
                }

    # Rule 5: Aggregate lookup complete — all reservations loaded, proceed with actions
    if state.get("all_reservations_loaded") and state.get("user_id"):
        known_ids = state.get("known_reservation_ids", [])
        inventory = state.get("reservation_inventory", {})
        
        # Find first reservation that hasn't been acted upon
        acted_on = state.get("acted_on_reservations", set())
        if not isinstance(acted_on, set):
            acted_on = set()
        
        for rid in known_ids:
            if rid not in acted_on and rid in inventory:
                entry = inventory[rid]
                if not isinstance(entry, dict):
                    continue
                
                # Check cancel eligibility first
                eligible, _ = cancel_eligibility(state, rid, "")
                if eligible:
                    state.setdefault("acted_on_reservations", set()).add(rid)
                    return {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": rid},
                    }
                
                # If not eligible for cancel, just mark as processed
                state.setdefault("acted_on_reservations", set()).add(rid)
        
        # All reservations processed — respond with summary
        return {
            "name": "respond",
            "arguments": {
                "content": "I've reviewed all your reservations. Based on the policy, I've completed the eligible actions. Is there anything specific you'd like me to help with next?"
            },
        }

    return None


def get_step_budget_remaining(state: dict[str, Any], max_steps: int = 30) -> int:
    """Calculate remaining steps budget. Returns 0 when about to exceed."""
    turn_count = state.get("turn_count", 0)
    return max_steps - turn_count


def should_force_transfer_on_budget_exhaust(state: dict[str, Any], max_steps: int = 30) -> bool:
    """Check if we should force transfer when budget is nearly exhausted."""
    remaining = get_step_budget_remaining(state, max_steps)
    return remaining <= 3 and state.get("active_flow", {}).get("stage") != "complete"
