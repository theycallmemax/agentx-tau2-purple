"""Reservation business logic: pricing, eligibility, inventory queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from .extractors import extract_airport_codes, extract_cancel_reason
except ImportError:  # pragma: no cover - direct src imports in tests
    from extractors import extract_airport_codes, extract_cancel_reason

BENCHMARK_NOW = datetime.fromisoformat("2024-05-15T15:00:00")
RESPOND_ACTION_NAME = "respond"


def fallback_action(content: str) -> dict[str, Any]:
    return {"name": RESPOND_ACTION_NAME, "arguments": {"content": content}}


def current_reservation_entry(
    state: dict[str, Any], reservation_id: str | None = None
) -> dict[str, Any] | None:
    candidate = reservation_id or state.get("reservation_id")
    inventory = state.get("reservation_inventory", {})
    if not isinstance(candidate, str) or not isinstance(inventory, dict):
        return None
    entry = inventory.get(candidate)
    return entry if isinstance(entry, dict) else None


def reservation_payment_id(
    state: dict[str, Any], reservation_id: str | None = None
) -> str | None:
    entry = current_reservation_entry(state, reservation_id)
    if not isinstance(entry, dict):
        return None
    payment_id = entry.get("payment_id")
    return payment_id if isinstance(payment_id, str) and payment_id else None


def current_membership(state: dict[str, Any]) -> str | None:
    membership = state.get("membership")
    return membership if isinstance(membership, str) and membership else None


def free_bags_per_passenger(membership: str | None, cabin: str | None) -> int:
    table = {
        "regular": {"basic_economy": 0, "economy": 1, "business": 2},
        "silver": {"basic_economy": 1, "economy": 2, "business": 3},
        "gold": {"basic_economy": 2, "economy": 3, "business": 4},
    }
    if not isinstance(membership, str) or not isinstance(cabin, str):
        return 0
    return table.get(membership, {}).get(cabin, 0)


def normalized_nonfree_baggages(
    state: dict[str, Any], reservation_id: str | None, total_baggages: int
) -> int | None:
    entry = current_reservation_entry(state, reservation_id)
    membership = current_membership(state)
    if not isinstance(entry, dict) or not isinstance(membership, str):
        return None
    passenger_count = entry.get("passenger_count")
    cabin = entry.get("cabin")
    if not isinstance(passenger_count, int) or not isinstance(cabin, str):
        return None
    allowance = free_bags_per_passenger(membership, cabin) * passenger_count
    return max(0, total_baggages - allowance)


def pricing_expression_for_current_reservation(
    state: dict[str, Any], reservation_id: str | None, target_cabin: str | None
) -> str | None:
    if not isinstance(reservation_id, str) or not isinstance(target_cabin, str):
        return None
    baseline_inventory = state.get("reservation_baseline_inventory", {})
    entry = current_reservation_entry(state, reservation_id)
    baseline_entry = (
        baseline_inventory.get(reservation_id)
        if isinstance(baseline_inventory, dict)
        else None
    )
    search_inventory = state.get("flight_search_inventory", {})
    if (
        not isinstance(entry, dict)
        or not isinstance(baseline_entry, dict)
        or not isinstance(search_inventory, dict)
    ):
        return None
    flights = entry.get("flights")
    baseline_flights = baseline_entry.get("flights")
    passenger_count = entry.get("passenger_count")
    if (
        not isinstance(flights, list)
        or not isinstance(baseline_flights, list)
        or not isinstance(passenger_count, int)
    ):
        return None

    terms: list[str] = []
    for flight in flights:
        if not isinstance(flight, dict):
            return None
        origin = flight.get("origin")
        destination = flight.get("destination")
        date = flight.get("date")
        flight_number = flight.get("flight_number")
        if not all(isinstance(value, str) and value for value in (origin, destination, date, flight_number)):
            return None

        baseline_match = next(
            (
                candidate
                for candidate in baseline_flights
                if isinstance(candidate, dict)
                and candidate.get("flight_number") == flight_number
                and candidate.get("date") == date
            ),
            None,
        )
        current_price = baseline_match.get("price") if isinstance(baseline_match, dict) else None
        if not isinstance(current_price, (int, float)):
            return None

        options = search_inventory.get(f"{origin}|{destination}|{date}")
        if not isinstance(options, list):
            return None
        match = next(
            (
                option
                for option in options
                if isinstance(option, dict) and option.get("flight_number") == flight_number
            ),
            None,
        )
        if not isinstance(match, dict):
            return None
        prices = match.get("prices")
        if not isinstance(prices, dict):
            return None
        new_price = prices.get(target_cabin)
        if not isinstance(new_price, (int, float)):
            return None
        terms.append(f"({int(new_price)} - {int(current_price)})")

    if not terms:
        return None
    joined = " + ".join(terms)
    if passenger_count == 1:
        return joined
    return f"{passenger_count} * ({joined})"


def next_missing_pricing_search(
    state: dict[str, Any], reservation_id: str | None
) -> dict[str, Any] | None:
    entry = current_reservation_entry(state, reservation_id)
    search_inventory = state.get("flight_search_inventory", {})
    if not isinstance(entry, dict) or not isinstance(search_inventory, dict):
        return None
    flights = entry.get("flights")
    if not isinstance(flights, list):
        return None
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        origin = flight.get("origin")
        destination = flight.get("destination")
        date = flight.get("date")
        if not all(isinstance(value, str) and value for value in (origin, destination, date)):
            continue
        key = f"{origin}|{destination}|{date}"
        if key in search_inventory:
            continue
        return {
            "name": "search_direct_flight",
            "arguments": {
                "origin": origin,
                "destination": destination,
                "date": date,
            },
        }
    return None


def next_segment_search_from_loaded_reservation(
    state: dict[str, Any], reservation_id: str | None
) -> dict[str, Any] | None:
    entry = current_reservation_entry(state, reservation_id)
    if not isinstance(entry, dict):
        return None
    flights = entry.get("flights")
    search_inventory = state.get("flight_search_inventory", {})
    if not isinstance(flights, list):
        return None
    if not isinstance(search_inventory, dict):
        search_inventory = {}
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        origin = flight.get("origin")
        destination = flight.get("destination")
        date = flight.get("date")
        if not all(isinstance(value, str) and value for value in (origin, destination, date)):
            continue
        key = f"{origin}|{destination}|{date}"
        if key in search_inventory:
            continue
        return {
            "name": "search_direct_flight",
            "arguments": {
                "origin": origin,
                "destination": destination,
                "date": date,
            },
        }
    return None


def matches_reservation_segment(
    state: dict[str, Any],
    reservation_id: str | None,
    origin: str | None,
    destination: str | None,
    date: str | None,
) -> bool:
    entry = current_reservation_entry(state, reservation_id)
    if not isinstance(entry, dict):
        return False
    flights = entry.get("flights")
    if not isinstance(flights, list):
        return False
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        if (
            flight.get("origin") == origin
            and flight.get("destination") == destination
            and flight.get("date") == date
        ):
            return True
    return False


def reservation_has_flown_segments(
    state: dict[str, Any], reservation_id: str | None
) -> bool | None:
    entry = current_reservation_entry(state, reservation_id)
    if not isinstance(entry, dict):
        return None
    flights = entry.get("flights")
    if not isinstance(flights, list) or not flights:
        return None
    benchmark_date = BENCHMARK_NOW.date()
    for flight in flights:
        if not isinstance(flight, dict):
            return None
        date_text = flight.get("date")
        if not isinstance(date_text, str):
            return None
        try:
            flight_date = datetime.fromisoformat(date_text).date()
        except ValueError:
            return None
        if flight_date < benchmark_date:
            return True
    return False


def select_second_cheapest_booking_option(
    state: dict[str, Any],
    origin: str | None,
    destination: str | None,
    date: str | None,
    cabin: str | None,
) -> dict[str, Any] | None:
    if not all(isinstance(value, str) and value for value in (origin, destination, date, cabin)):
        return None
    inventory_by_mode = state.get("flight_search_inventory_by_mode", {})
    if not isinstance(inventory_by_mode, dict):
        return None
    key = f"{origin}|{destination}|{date}"
    bucket = inventory_by_mode.get(key)
    if not isinstance(bucket, dict):
        return None

    options: list[dict[str, Any]] = []

    direct_options = bucket.get("search_direct_flight")
    if isinstance(direct_options, list):
        for option in direct_options:
            if not isinstance(option, dict):
                continue
            prices = option.get("prices")
            flight_number = option.get("flight_number")
            if not isinstance(prices, dict) or not isinstance(flight_number, str):
                continue
            price = prices.get(cabin)
            if not isinstance(price, (int, float)):
                continue
            options.append(
                {
                    "total_price": int(price),
                    "flights": [{"flight_number": flight_number, "date": date}],
                }
            )

    onestop_options = bucket.get("search_onestop_flight")
    if isinstance(onestop_options, list):
        for option in onestop_options:
            if not isinstance(option, list) or len(option) != 2:
                continue
            segment_a, segment_b = option
            if not isinstance(segment_a, dict) or not isinstance(segment_b, dict):
                continue
            prices_a = segment_a.get("prices")
            prices_b = segment_b.get("prices")
            flight_a = segment_a.get("flight_number")
            flight_b = segment_b.get("flight_number")
            date_a = segment_a.get("date") or date
            date_b = segment_b.get("date") or date
            if (
                not isinstance(prices_a, dict)
                or not isinstance(prices_b, dict)
                or not isinstance(flight_a, str)
                or not isinstance(flight_b, str)
                or not isinstance(date_a, str)
                or not isinstance(date_b, str)
            ):
                continue
            price_a = prices_a.get(cabin)
            price_b = prices_b.get(cabin)
            if not isinstance(price_a, (int, float)) or not isinstance(price_b, (int, float)):
                continue
            options.append(
                {
                    "total_price": int(price_a) + int(price_b),
                    "flights": [
                        {"flight_number": flight_a, "date": date_a},
                        {"flight_number": flight_b, "date": date_b},
                    ],
                }
            )

    if len(options) < 2:
        return None
    options.sort(
        key=lambda option: (
            int(option["total_price"]),
            len(option["flights"]),
            tuple(flight["flight_number"] for flight in option["flights"]),
        )
    )
    return options[1]


def same_flights_update_action(
    state: dict[str, Any], reservation_id: str, cabin: str
) -> dict[str, Any] | None:
    entry = current_reservation_entry(state, reservation_id)
    if not isinstance(entry, dict):
        return None
    flights = entry.get("flights")
    payment_id = reservation_payment_id(state, reservation_id)
    if not isinstance(flights, list) or not flights or not payment_id:
        return None
    normalized_flights = []
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        flight_number = flight.get("flight_number")
        date = flight.get("date")
        if isinstance(flight_number, str) and isinstance(date, str):
            normalized_flights.append({"flight_number": flight_number, "date": date})
    if not normalized_flights:
        return None
    return {
        "name": "update_reservation_flights",
        "arguments": {
            "reservation_id": reservation_id,
            "cabin": cabin,
            "flights": normalized_flights,
            "payment_id": payment_id,
        },
    }


def cabin_change_confirmation(
    state: dict[str, Any], reservation_id: str, cabin: str
) -> dict[str, Any] | None:
    entry = current_reservation_entry(state, reservation_id)
    if not isinstance(entry, dict):
        return None
    current_cabin = entry.get("cabin")
    flights = entry.get("flights") or []
    payment_id = reservation_payment_id(state, reservation_id)
    if not payment_id or not isinstance(current_cabin, str):
        return None
    lines = []
    for flight in flights:
        if not isinstance(flight, dict):
            continue
        flight_number = flight.get("flight_number")
        date = flight.get("date")
        if isinstance(flight_number, str) and isinstance(date, str):
            lines.append(f"- {flight_number} on {date}")
    if not lines:
        return None
    total_cost = state.get("last_calculation_result")
    price_line = ""
    if isinstance(total_cost, (int, float)):
        price_line = f"\n\nThe total fare difference for this change is ${float(total_cost):.2f}."
    return fallback_action(
        f"I can change reservation {reservation_id} from {current_cabin} to {cabin} for the same flights:\n"
        + "\n".join(lines)
        + price_line
        + f"\n\nI would use the original payment method on file ({payment_id}) for any fare difference or refund.\n\n"
        + 'Reply "yes" to confirm I should apply this cabin change.'
    )


def cancel_eligibility(
    state: dict[str, Any], reservation_id: str | None, latest_user_text: str
) -> tuple[bool, str | None]:
    entry = current_reservation_entry(state, reservation_id)
    if not entry:
        return False, "Please share your reservation number so I can verify whether the booking can be cancelled."

    status = entry.get("status")
    if isinstance(status, str) and status.lower() == "cancelled":
        return False, "That reservation is already cancelled."

    created_at = entry.get("created_at")
    booked_within_24h = False
    if isinstance(created_at, str):
        try:
            created_dt = datetime.fromisoformat(created_at)
            booked_within_24h = (BENCHMARK_NOW - created_dt).total_seconds() <= 24 * 3600
        except ValueError:
            booked_within_24h = False

    cabin = entry.get("cabin")
    is_business = isinstance(cabin, str) and cabin == "business"
    has_insurance = entry.get("insurance") == "yes"
    airline_cancelled = isinstance(status, str) and status.lower() == "cancelled"

    if booked_within_24h or airline_cancelled or is_business:
        return True, None

    # If the booking is already older than 24h, not business, and has no insurance,
    # the cancellation is disallowed regardless of the user's stated reason.
    if not has_insurance:
        return (
            False,
            "This reservation is not eligible for cancellation under the airline policy because "
            "it was not booked within 24 hours, the flight was not cancelled by the airline, "
            "the cabin is not business, and the booking does not include travel insurance.",
        )

    cancel_reason = extract_cancel_reason(latest_user_text)
    if cancel_reason is None and latest_user_text in {
        "change_of_plan",
        "airline_cancelled",
        "health",
        "weather",
        "other",
    }:
        cancel_reason = latest_user_text
    # Fall back to state-stored cancel reason (set from earlier in the conversation)
    if cancel_reason is None:
        state_reason = state.get("cancel_reason")
        if isinstance(state_reason, str) and state_reason:
            cancel_reason = state_reason
    # Also check per-reservation cancel reason
    if cancel_reason is None and isinstance(reservation_id, str):
        reason_by_res = state.get("cancel_reason_by_reservation", {})
        if isinstance(reason_by_res, dict):
            r = reason_by_res.get(reservation_id)
            if isinstance(r, str) and r:
                cancel_reason = r
    covered_reason = cancel_reason in {"health", "weather"}
    airline_cancelled = cancel_reason == "airline_cancelled"

    if airline_cancelled or (has_insurance and covered_reason):
        return True, None

    if not cancel_reason:
        return (
            False,
            "Please share the reason for cancellation so I can verify whether this reservation is eligible to be cancelled.",
        )

    return (
        False,
        "This reservation is not eligible for cancellation under the airline policy because "
        "it was not booked within 24 hours, the flight was not cancelled by the airline, "
        "the cabin is not business, and the stated reason is not covered by insurance.",
    )


def balance_response_from_state(state: dict[str, Any]) -> dict[str, Any]:
    balances = state.get("known_payment_balances", {})
    if not isinstance(balances, dict) or not balances:
        return fallback_action(
            "Please share your user ID so I can look up the balances on your gift cards and certificates."
        )

    lines = []
    gift_total = 0.0
    certificate_total = 0.0
    for payment_id, amount in sorted(balances.items()):
        if payment_id.startswith(("gift_card_", "certificate_")):
            lines.append(f"- {payment_id}: ${amount:.0f}")
        if payment_id.startswith("gift_card_"):
            gift_total += float(amount)
        if payment_id.startswith("certificate_"):
            certificate_total += float(amount)
    if not lines:
        return fallback_action(
            "I couldn't find any gift cards or certificates on file in your profile."
        )
    return fallback_action(
        "Here are the gift card and certificate balances I found:\n"
        + "\n".join(lines)
        + f"\n\nTotal gift cards: ${gift_total:.2f}\nTotal certificates: ${certificate_total:.2f}"
    )


def infer_reservation_from_inventory(
    state: dict[str, Any], text: str
) -> str | None:
    try:
        from .extractors import AIRPORT_CODE_PATTERN, extract_dates
    except ImportError:  # pragma: no cover - direct src imports in tests
        from extractors import AIRPORT_CODE_PATTERN, extract_dates

    inventory = state.get("reservation_inventory", {})
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

    dates = set(extract_dates(text))
    airport_codes = extract_airport_codes(text)
    current_reservation = state.get("reservation_id")

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


def find_reservation_by_route(
    state: dict[str, Any],
    origin: str | None,
    destination: str | None,
    date: str | None = None,
) -> str | None:
    """Find a reservation_id by matching itinerary origin/destination across loaded reservations."""
    inventory = state.get("reservation_inventory", {})
    if not isinstance(inventory, dict) or not origin or not destination:
        return None
    for reservation_id, entry in inventory.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("origin") == origin and entry.get("destination") == destination:
            if date and isinstance(entry.get("dates"), list) and date not in entry["dates"]:
                continue
            return reservation_id
        flights = entry.get("flights")
        if not isinstance(flights, list):
            continue
        for flight in flights:
            if not isinstance(flight, dict):
                continue
            if flight.get("origin") == origin and flight.get("destination") == destination:
                if date and flight.get("date") != date:
                    continue
                return reservation_id
    return None


def find_reservation_by_flight_number(
    state: dict[str, Any], flight_number: str
) -> str | None:
    """Find a reservation_id that contains the given flight_number."""
    inventory = state.get("reservation_inventory", {})
    if not isinstance(inventory, dict) or not flight_number:
        return None
    for reservation_id, entry in inventory.items():
        if not isinstance(entry, dict):
            continue
        flight_numbers = entry.get("flight_numbers", [])
        if isinstance(flight_numbers, list) and flight_number in flight_numbers:
            return reservation_id
        flights = entry.get("flights")
        if isinstance(flights, list):
            for flight in flights:
                if isinstance(flight, dict) and flight.get("flight_number") == flight_number:
                    return reservation_id
    return None
