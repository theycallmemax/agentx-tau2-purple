"""LLM-based intent classifier — dual classifier with regex fallback (phantom-agent principle)."""

from __future__ import annotations

import json
import re
from typing import Any

LLM_CLASSIFY_PROMPT = """<ROLE>You are an intent classifier for an airline customer service agent.</ROLE>

<TASK>
{task_text}
</TASK>

<INSTRUCTIONS>
Classify this task into exactly one intent from the list below.

Available intents:
- "cancel": User wants to cancel a reservation (cancel, refund, undo booking)
- "booking": User wants to book a new flight (book, reservation, new flight, search)
- "modify": User wants to change an existing reservation (change, modify, upgrade, downgrade, switch)
- "baggage": User asks about baggage (bag, baggage, luggage, suitcase)
- "status": User asks about flight status or compensation (status, compensation, delay, cancelled)
- "general": Anything else

IMPORTANT: ALL CAPS text is NOT ambiguous — it's just emphasis.
Terse requests like "cancel all" or "check baggage" are NOT ambiguous.

Return ONLY a JSON object: {{"intent": "...", "confidence": 0.0-1.0, "reason": "..."}}
</INSTRUCTIONS>"""

_VALID_INTENTS = {"cancel", "booking", "modify", "baggage", "status", "general"}


def llm_classify_intent(llm_call_fn, task_text: str) -> dict[str, Any] | None:
    """Classify intent using LLM. Returns None on error.

    Args:
        llm_call_fn: A function that takes messages=[{"role":"user","content":...}]
                     and max_tokens, returns a response object with
                     response.choices[0].message.content
        task_text: The user's task text
    """
    try:
        prompt = LLM_CLASSIFY_PROMPT.format(task_text=task_text[:500])
        response = llm_call_fn(messages=[{"role": "user", "content": prompt}], max_tokens=150)

        text = ""
        if response:
            try:
                text = response.choices[0].message.content or ""
            except Exception:
                text = str(response)

        for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
            try:
                obj = json.loads(m.group())
                intent = obj.get("intent", "")
                confidence = float(obj.get("confidence", 0.8))
                if intent in _VALID_INTENTS:
                    return {"intent": intent, "confidence": confidence}
            except Exception:
                continue
    except Exception:
        pass
    return None


def classify_intent(task_text: str) -> dict[str, Any]:
    """Regex-based intent classifier (fallback from phantom-agent).

    Returns {"intent": str, "confidence": float}
    """
    lowered = task_text.lower()

    # Cancel intent
    cancel_patterns = [
        r"cancel",
        r"reservat.*cancel",
        r"undo.*booking",
        r"undo.*reservation",
        r"request.*cancel",
        r"would like to cancel",
        r"want to cancel",
        r"need to cancel",
        r"cancel all",
        r"cancel every",
        r"cancel both",
        r"cancel my",
        r"cancel the reservation",
        r"cancel reservation",
        r"can i cancel",
        r"could i cancel",
    ]
    for pat in cancel_patterns:
        if re.search(pat, lowered):
            return {"intent": "cancel", "confidence": 0.9}

    # Baggage intent
    baggage_patterns = [
        r"baggage",
        r"bag ",
        r"bags?",
        r"luggage",
        r"suitcase",
        r"carry.?on",
        r"checked bag",
        r"bag allowance",
        r"baggage allowance",
        r"how many bag",
        r"free bag",
        r"free baggage",
        r"additional bag",
        r"extra bag",
    ]
    for pat in baggage_patterns:
        if re.search(pat, lowered):
            return {"intent": "baggage", "confidence": 0.85}

    # Booking intent
    booking_patterns = [
        r"book a flight",
        r"book.*flight",
        r"new reservat",
        r"make a reservat",
        r"search.*flight",
        r"looking for a flight",
        r"find.*flight",
        r"available.*flight",
        r"flights? from",
        r"flights? to",
        r"flights? on",
        r"i want to fly",
        r"i need to fly",
        r"flights?\?",
    ]
    for pat in booking_patterns:
        if re.search(pat, lowered):
            return {"intent": "booking", "confidence": 0.85}

    # Modify intent
    modify_patterns = [
        r"change my flight",
        r"change.*reservat",
        r"modify.*reservat",
        r"switch to",
        r"upgrade",
        r"downgrade",
        r"change cabin",
        r"change the class",
        r"change.*date",
        r"move.*flight",
        r"push.*back",
        r"reschedule",
        r"change my flights",
    ]
    for pat in modify_patterns:
        if re.search(pat, lowered):
            return {"intent": "modify", "confidence": 0.85}

    # Status intent
    status_patterns = [
        r"flight status",
        r"status of flight",
        r"status for flight",
        r"compensation",
        r"eligible.*compensation",
        r"compensation eligib",
        r"flight.*delay",
        r"delay.*flight",
        r"cancelled by airline",
        r"canceled by airline",
        r"airline cancel",
    ]
    for pat in status_patterns:
        if re.search(pat, lowered):
            return {"intent": "status", "confidence": 0.85}

    # No confident match
    return {"intent": "general", "confidence": 0.5}


def dual_classify_intent(
    llm_call_fn, task_text: str, regex_fn=classify_intent
) -> dict[str, Any]:
    """Dual classifier: LLM first, regex fallback (phantom-agent principle).

    1. Try LLM classification
    2. If LLM returns "general" with low confidence, or fails → use regex
    3. If LLM returns a confident non-general intent → use it

    Args:
        llm_call_fn: Function to call LLM (see llm_classify_intent)
        task_text: User's task text
        regex_fn: Regex classifier function (default: classify_intent)

    Returns:
        {"intent": str, "confidence": float, "classifier": str}
    """
    # Try LLM first
    llm_result = llm_classify_intent(llm_call_fn, task_text)

    if llm_result:
        # If LLM is confident and not "general" → use it
        if llm_result["confidence"] >= 0.7 and llm_result["intent"] != "general":
            return {
                "intent": llm_result["intent"],
                "confidence": llm_result["confidence"],
                "classifier": "llm",
            }

    # Fallback to regex
    regex_result = regex_fn(task_text)
    return {
        "intent": regex_result["intent"],
        "confidence": regex_result["confidence"],
        "classifier": "regex",
    }
