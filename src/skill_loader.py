"""Skill loader and dual classifier (phantom-agent principle).

Loads skill prompts from .md files and classifies tasks using LLM + regex fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SKILLS_DIR = Path(__file__).parent


def load_skill_prompt(skill_id: str) -> str:
    """Load skill prompt from .md file — hot-reloadable."""
    skill_file = SKILLS_DIR / f"{skill_id}.md"
    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8").strip()
    return ""


def get_skills_menu() -> str:
    """Build skills menu for system prompt."""
    skills = {
        "cancel": "Cancel one or more reservations",
        "booking": "Book a new flight reservation",
        "modify": "Change an existing reservation (date, cabin, route)",
        "baggage": "Check or update baggage allowance",
        "status": "Check flight status or compensation eligibility",
        "general": "General assistance",
    }
    lines = ["\n<AVAILABLE_SKILLS>"]
    for sid, desc in skills.items():
        lines.append(f"- {sid}: {desc}")
    lines.append("</AVAILABLE_SKILLS>")
    return "\n".join(lines)


# ============================================================================
# REGEX CLASSIFIER (fallback)
# ============================================================================

def classify_intent_regex(task_text: str) -> dict[str, Any]:
    """Regex-based intent classifier (fallback from phantom-agent)."""
    lowered = task_text.lower()

    # Cancel
    cancel_patterns = [
        r"cancel", r"reservat.*cancel", r"undo.*booking", r"undo.*reservation",
        r"request.*cancel", r"would like to cancel", r"want to cancel",
        r"need to cancel", r"cancel all", r"cancel every", r"cancel both",
        r"cancel my", r"cancel the reservation", r"can i cancel",
        r"could i cancel",
    ]
    for pat in cancel_patterns:
        if re.search(pat, lowered):
            return {"intent": "cancel", "confidence": 0.9}

    # Baggage
    baggage_patterns = [
        r"baggage", r"\bbag\b", r"\bbags\b", r"luggage", r"suitcase",
        r"carry.?on", r"checked bag", r"bag allowance", r"baggage allowance",
        r"how many bag", r"free bag", r"additional bag", r"extra bag",
    ]
    for pat in baggage_patterns:
        if re.search(pat, lowered):
            return {"intent": "baggage", "confidence": 0.85}

    # Booking
    booking_patterns = [
        r"book a flight", r"book.*flight", r"new reservat",
        r"make a reservat", r"search.*flight", r"looking for a flight",
        r"find.*flight", r"available.*flight", r"flights? from",
        r"flights? to", r"flights? on", r"i want to fly", r"i need to fly",
        r"flights?\?",
    ]
    for pat in booking_patterns:
        if re.search(pat, lowered):
            return {"intent": "booking", "confidence": 0.85}

    # Modify
    modify_patterns = [
        r"change my flight", r"change.*reservat", r"modify.*reservat",
        r"switch to", r"upgrade", r"downgrade", r"change cabin",
        r"change the class", r"change.*date", r"move.*flight",
        r"push.*back", r"reschedule", r"change my flights",
    ]
    for pat in modify_patterns:
        if re.search(pat, lowered):
            return {"intent": "modify", "confidence": 0.85}

    # Status
    status_patterns = [
        r"flight status", r"status of flight", r"status for flight",
        r"compensation", r"eligible.*compensation", r"compensation eligib",
        r"flight.*delay", r"delay.*flight", r"cancelled by airline",
        r"canceled by airline", r"airline cancel",
    ]
    for pat in status_patterns:
        if re.search(pat, lowered):
            return {"intent": "status", "confidence": 0.85}

    return {"intent": "general", "confidence": 0.5}


# ============================================================================
# LLM CLASSIFIER
# ============================================================================

LLM_CLASSIFY_PROMPT = """<ROLE>You are an intent classifier for an airline customer service agent.</ROLE>

<TASK>
{task_text}
</TASK>

<INSTRUCTIONS>
Classify this task into exactly one intent:
- "cancel": User wants to cancel (cancel, refund, undo booking)
- "booking": User wants to book a new flight (book, new reservation, search flights)
- "modify": User wants to change existing reservation (change, modify, upgrade, downgrade, switch)
- "baggage": User asks about baggage (bag, luggage, suitcase, carry-on)
- "status": User asks about flight status or compensation (status, compensation, delay)
- "general": Anything else

IMPORTANT: ALL CAPS text is NOT ambiguous — it's just emphasis.
Return ONLY JSON: {{"intent": "...", "confidence": 0.0-1.0}}
</INSTRUCTIONS>"""


def llm_classify_intent(llm_call_fn, task_text: str) -> dict[str, Any] | None:
    """Classify intent using LLM. Returns None on error."""
    try:
        prompt = LLM_CLASSIFY_PROMPT.format(task_text=task_text[:500])
        response = llm_call_fn(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
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
                if intent in {"cancel", "booking", "modify", "baggage", "status", "general"}:
                    return {"intent": intent, "confidence": confidence}
            except Exception:
                continue
    except Exception:
        pass
    return None


# ============================================================================
# DUAL CLASSIFIER (LLM first, regex fallback)
# ============================================================================

def dual_classify_intent(llm_call_fn, task_text: str) -> str:
    """Dual classifier: LLM first, regex fallback (phantom-agent principle).

    Returns skill_id string: cancel, booking, modify, baggage, status, or general.
    """
    llm_result = llm_classify_intent(llm_call_fn, task_text)

    if llm_result and llm_result["confidence"] >= 0.7 and llm_result["intent"] != "general":
        return llm_result["intent"]

    regex_result = classify_intent_regex(task_text)
    return regex_result["intent"]
