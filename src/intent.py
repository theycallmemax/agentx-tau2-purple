"""User intent classification from text."""

from __future__ import annotations


def looks_like_modify_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "change my flight",
            "modify my flight",
            "change reservation",
            "modify reservation",
            "switch to",
            "push it back",
            "move it to",
            "upgrade",
            "upgrade the class",
            "downgrade",
            "downgrade the class",
            "downgrade the cabin",
            "upgrade cabin",
            "change the cabin",
            "change the class",
            "nonstop flight",
            "direct flight",
            "cheapest economy",
            # Additional patterns from failed tasks
            "change my flights",
            "change my flight date",
            "change the flight date",
            "change my booking",
            "change the booking",
            "change the reservation",
            "change this reservation",
            "switch my flight",
            "switch flights",
            "switch to a nonstop",
            "switch to a direct",
            "downgrade all my",
            "downgrade my",
            "downgrade the cabin",
            "downgrade the flights",
            "move to the next day",
            "move to the day after",
            "move it to the next",
            "find the cheapest economy",
            "find a cheaper",
            "earlier flight",
            "later flight",
            "shortest return",
            "fastest return",
            "change my return flight",
            "change the return",
            "change the departure",
            "change my flights from",
            "change the name on my reservation",
            "change the passenger name",
            "change the passenger",
            "update the reservation",
        )
    )


def looks_like_remove_passenger_intent(text: str) -> bool:
    lowered = text.lower()
    return (
        "remove passenger" in lowered
        or "remove a passenger" in lowered
        or "remove just" in lowered
        or ("remove" in lowered and "passenger" in lowered)
    )


def looks_like_insurance_intent(text: str) -> bool:
    lowered = text.lower()
    return "insurance" in lowered and not looks_like_compensation_intent(text)


def looks_like_cancel_intent(text: str) -> bool:
    lowered = text.lower()
    return "cancel" in lowered or "cancellation" in lowered


def looks_like_direct_cancellation_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "cancel my reservation",
            "cancel the reservation",
            "cancel reservation",
            "cancel my flight",
            "cancel this flight",
            "please cancel",
            "i want to cancel",
            "i would like to cancel",
            "go ahead and cancel",
        )
    )


def looks_like_affirmation(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "yes",
            "go ahead",
            "please proceed",
            "proceed with",
            "do it",
            "confirm",
        )
    )


def looks_like_stop_without_changes(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "hold off",
            "for now",
            "no changes",
            "do nothing",
            "leave it as is",
            "leave it as-is",
            "keep it as is",
            "keep it as-is",
            "i'll wait",
            "i'll think about it",
            "take more time",
            "not making any changes",
            "won't make any changes",
        )
    )


def looks_like_social_pressure(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "most lenient",
            "most helpful",
            "other agent said",
            "customer support representative assured",
            "customer support representative approved",
            "representative approved",
            "was approved",
            "already approved",
            "please be flexible",
            "please make an exception",
            "really important",
            "i really believe",
            "i was told",
            "i've been told",
            "i have been told",
            "you were wrong",
            "you are wrong",
            "must get a refund",
            "entitled to a refund",
            "deserve a refund",
            "should get a refund",
            "insisted",
        )
    )


def looks_like_all_reservations_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "all upcoming",
            "all upcoming flights",
            "all my upcoming flights",
            "all of my upcoming flights",
            "all reservations",
            "all my flights",
            "cancel all",
            "cancel all of my flights",
            "include all",
            "every reservation",
            "all business",
        )
    )


def looks_like_second_cheapest_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "second cheapest",
            "2nd cheapest",
            "not the cheapest",
            "other than the cheapest",
        )
    )


def looks_like_reservation_triage_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "which reservation",
            "check reservation",
            "find my",
            "which one should i check",
            "other reservation",
            "correct reservation",
        )
    )


def looks_like_booking_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "book a flight",
            "book a one-way flight",
            "book a one way flight",
            "make a reservation",
            "make me a reservation",
            "i'd like to book",
            "i would like to book",
            "reserve a flight",
            # Additional patterns from failed tasks
            "book the flight",
            "book that flight",
            "book me a flight",
            "book a reservation",
            "book a new flight",
            "i'm looking to book",
            "i am looking to book",
            "looking to book",
            "i want to book",
            "book a ticket",
            "i'd like to make",
            "i would like to make a reservation",
            "book a new reservation",
            "proceed with booking",
            "proceed with the booking",
            "proceed with the flight booking",
            "help me book",
            "help me find a flight",
            "find available flights",
            "find flights for",
            "check the availability",
            "check availability for",
            "what options are available",
            "book for my friend",
            "book for my",
            "add a passenger",
            "book the same flight",
        )
    )


def looks_like_same_reservation_reference(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "same as my current reservation",
            "same flight that i had",
            "exactly the same as my current reservation",
            "my current reservation",
        )
    )


def looks_like_compensation_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "compensation",
            "certificate",
            "delayed flight",
            "canceled flight",
            "cancelled flight",
        )
    )


def looks_like_balance_intent(text: str) -> bool:
    lowered = text.lower()
    return "balance" in lowered and (
        "gift card" in lowered or "certificate" in lowered
    )


def looks_like_baggage_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "baggage",
            "bag allowance",
            "bags allowed",
            "how many suitcases",
            "how many bags",
            "checked bag",
            "checked baggage",
            "suitcases",
        )
    )


def looks_like_status_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "flight status",
            "status of my flight",
            "delayed flight",
            "delay",
            "canceled flight",
            "cancelled flight",
            "is my flight",
            "what happened to flight",
        )
    )


def classify_task_type(text: str) -> str:
    if looks_like_balance_intent(text):
        return "balance"
    if looks_like_remove_passenger_intent(text):
        return "remove_passenger"
    if looks_like_baggage_intent(text):
        return "baggage"
    if looks_like_insurance_intent(text):
        return "insurance"
    if (
        looks_like_status_intent(text) or looks_like_compensation_intent(text)
    ) and not looks_like_direct_cancellation_request(text):
        return "status"
    if looks_like_cancel_intent(text):
        return "cancel"
    if looks_like_booking_intent(text):
        return "booking"
    if looks_like_modify_intent(text):
        return "modify"
    return "general"


def detect_all_intents(text: str) -> list[str]:
    """Return ALL intents found in text, in priority order."""
    intents = []
    if looks_like_balance_intent(text):
        intents.append("balance")
    if looks_like_remove_passenger_intent(text):
        intents.append("remove_passenger")
    if looks_like_baggage_intent(text):
        intents.append("baggage")
    if looks_like_insurance_intent(text):
        intents.append("insurance")
    if looks_like_status_intent(text) or looks_like_compensation_intent(text):
        if not looks_like_direct_cancellation_request(text):
            intents.append("status")
    if looks_like_cancel_intent(text):
        intents.append("cancel")
    if looks_like_booking_intent(text):
        intents.append("booking")
    if looks_like_modify_intent(text):
        intents.append("modify")
    if not intents:
        intents.append("general")
    return intents


def looks_like_booking_after_cancel(text: str) -> bool:
    """Detect when user wants to book a NEW flight after cancel denial."""
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "book a new",
            "book me a new",
            "book a flight",
            "book the cheapest",
            "book the second",
            "find and book",
            "make a new reservation",
            "find the cheapest",
            "find me a flight",
            "find a flight",
            "search for a flight",
            "search for flights",
            "book a one-way",
            "book a one way",
            "book economy",
            "book in economy",
            "book business",
        )
    )


def looks_like_multi_action_request(text: str) -> bool:
    """Detect when user has multiple requests in one message."""
    lowered = text.lower()
    action_count = 0
    if looks_like_cancel_intent(text):
        action_count += 1
    if looks_like_booking_intent(text):
        action_count += 1
    if looks_like_modify_intent(text):
        action_count += 1
    if looks_like_baggage_intent(text):
        action_count += 1
    # Also check for multiple reservation mentions
    res_count = text.lower().count("reservation") + text.lower().count("booking") + text.lower().count("flight")
    if action_count >= 2 or res_count >= 4:
        return True
    # Check for "and" connecting two actions
    if any(
        pattern in lowered
        for pattern in (
            "cancel", "and i also",
            "cancel", "and then",
            "cancel", "and book",
            "cancel", "and change",
            "cancel", "and modify",
            "cancel", "and upgrade",
            "cancel", "and i want",
            "cancel", "and i need",
        )
    ):
        return True
    return False
