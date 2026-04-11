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
