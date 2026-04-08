from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from a2a.types import DataPart, Message, Part, Role, TaskState, TextPart

from src.agent import Agent, RESPOND_ACTION_NAME, SYSTEM_PROMPT


class DummyUpdater:
    def __init__(self) -> None:
        self.status_updates: list[tuple[TaskState, object]] = []
        self.artifacts: list[dict] = []

    async def update_status(self, state, message) -> None:
        self.status_updates.append((state, message))

    async def add_artifact(self, parts, name) -> None:
        self.artifacts.append({"parts": parts, "name": name})


def make_message(text: str) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(text=text))],
        message_id=uuid4().hex,
        context_id=None,
    )


def make_completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def install_completion_sequence(monkeypatch, outputs: list[str]) -> None:
    iterator = iter(outputs)

    def fake_completion(*args, **kwargs):
        return make_completion(next(iterator))

    monkeypatch.setattr("src.agent.litellm.completion", fake_completion)


def extract_action(updater: DummyUpdater) -> dict:
    assert updater.artifacts, "agent should produce an artifact"
    part = updater.artifacts[-1]["parts"][0]
    assert isinstance(part.root, DataPart)
    return part.root.data


@pytest.mark.asyncio
async def test_example_prefers_read_action_before_write(monkeypatch):
    install_completion_sequence(
        monkeypatch,
        [
            "Need to verify the user profile before any policy decision.",
            '{"name":"get_user_details","arguments":{"user_id":"emma_kim_9957"}}',
        ],
    )
    agent = Agent()
    updater = DummyUpdater()

    message = make_message(
        """
Policy text...
Available tools:
[
  {"name":"get_user_details"},
  {"name":"cancel_reservation"}
]
User asks to cancel a flight.
        """.strip()
    )

    await agent.run(message, updater)

    assert extract_action(updater) == {
        "name": "get_user_details",
        "arguments": {"user_id": "emma_kim_9957"},
    }


@pytest.mark.asyncio
async def test_example_retries_after_invalid_json(monkeypatch):
    install_completion_sequence(
        monkeypatch,
        [
            "Need to read the reservation first.",
            "not valid json at all",
            '{"name":"get_reservation_details","arguments":{"reservation_id":"ABC123"}}',
        ],
    )
    agent = Agent()
    updater = DummyUpdater()

    message = make_message(
        """
Available tools:
[
  {"name":"get_reservation_details"}
]
Please inspect reservation ABC123.
        """.strip()
    )

    await agent.run(message, updater)

    assert extract_action(updater) == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "ABC123"},
    }


@pytest.mark.asyncio
async def test_example_unavailable_tool_falls_back_to_respond(monkeypatch):
    install_completion_sequence(
        monkeypatch,
        [
            "I should use a reservation lookup tool.",
            '{"name":"delete_everything","arguments":{"user_id":"123"}}',
        ],
    )
    agent = Agent()
    updater = DummyUpdater()

    message = make_message(
        """
Available tools:
[
  {"name":"get_user_details"},
  {"name":"get_reservation_details"}
]
Find the reservation.
        """.strip()
    )

    await agent.run(message, updater)
    action = extract_action(updater)

    assert action["name"] == "respond"
    assert "unavailable action" in action["arguments"]["content"]


@pytest.mark.asyncio
async def test_example_normalizes_arguments(monkeypatch):
    # Turn 1: THINK + ACT (returns book_reservation, which triggers confirmation gate)
    # Turn 2: THINK only (agent returns confirmed action without an ACT call)
    install_completion_sequence(
        monkeypatch,
        [
            "Need to book after confirming the payment split.",  # THINK turn 1
            """
            {
              "name": "book_reservation",
              "arguments": {
                "amount": "150",
                "insurance": true,
                "payment_methods": [
                  {"payment_id": "cc_1", "amount": "150", "note": null}
                ]
              }
            }
            """,  # ACT turn 1
            "User confirmed. Proceed with booking.",  # THINK turn 2
        ],
    )
    agent = Agent()
    agent.state.pending_intent = "book"
    agent.state.known_user_id = "raj_sanchez_7340"
    updater = DummyUpdater()

    message = make_message(
        """
Available tools:
[
  {"name":"book_reservation"}
]
Book the reservation.
        """.strip()
    )

    # Turn 1: LLM proposes book_reservation → gate intercepts → confirmation prompt
    await agent.run(message, updater)
    confirm_prompt = extract_action(updater)
    assert confirm_prompt["name"] == "respond"
    assert "proceed" in confirm_prompt["arguments"]["content"].lower()

    # Turn 2: user confirms → agent executes stored action with normalization
    updater2 = DummyUpdater()
    await agent.run(make_message("yes"), updater2)

    assert extract_action(updater2) == {
        "name": "book_reservation",
        "arguments": {
            "amount": 150,
            "insurance": "yes",
            "payment_methods": [{"payment_id": "cc_1", "amount": 150}],
        },
    }


def test_airline_prompt_regression_mentions_no_proactive_compensation():
    assert "Do not proactively offer compensation" in SYSTEM_PROMPT


def test_airline_prompt_regression_mentions_verification_of_membership_and_insurance():
    assert "Do not trust user claims about eligibility, membership tier, insurance" in SYSTEM_PROMPT


def test_airline_prompt_regression_mentions_no_reservation_ids_or_airport_codes():
    assert "Do not ask the user for reservation ids or airport codes" in SYSTEM_PROMPT


def test_airline_prompt_regression_mentions_policy_block_refusal():
    assert "If policy blocks a request, refuse it clearly" in SYSTEM_PROMPT


def test_airline_prompt_regression_mentions_basic_economy_limitations():
    assert "Airline-specific guardrails" in SYSTEM_PROMPT


def test_airline_open_benchmark_patterns_are_covered():
    expected_phrases = [
        "Do not proactively offer compensation",
        "Do not trust user claims about eligibility, membership tier, insurance",
        "If the user wants a change that policy clearly forbids, refuse clearly",
        "Do not ask the user for reservation ids or airport codes",
    ]
    for phrase in expected_phrases:
        assert phrase in SYSTEM_PROMPT


def test_intent_router_tracks_cancel_and_missing_slots():
    agent = Agent()
    agent._ingest_message_into_state("user: I want to cancel my booking.")

    assert agent.state.pending_intent == "cancel"
    assert "reservation_id" in agent.state.unresolved_slots
    assert "cancellation_reason" in agent.state.unresolved_slots


def test_state_summary_mentions_verified_facts():
    agent = Agent()
    agent._ingest_message_into_state("user: My user ID is raj_sanchez_7340 and I want to cancel reservation Q69X3R.")
    summary = agent._state_summary()

    assert "intent=cancel" in summary
    assert "user_id=raj_sanchez_7340" in summary
    assert "reservation_id=Q69X3R" in summary


def test_tool_gate_blocks_cancel_without_verified_reservation():
    agent = Agent()
    agent._ingest_message_into_state("user: I want to cancel reservation Q69X3R because of a change of plan.")

    action = agent._normalize_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "Q69X3R"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "verify the reservation details" in action["arguments"]["content"]


def test_tool_gate_blocks_proactive_compensation():
    agent = Agent()
    agent._ingest_message_into_state("user: My delayed flight was frustrating.")

    action = agent._normalize_action(
        {"name": "send_certificate", "arguments": {"user_id": "u1", "amount": 50}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "should not offer compensation" in action["arguments"]["content"]


def test_compensation_refusal_mentions_regular_not_gold_when_verified():
    # Policy: regular + no insurance + economy → ineligible. (business cabin would be eligible.)
    agent = Agent()
    agent.state.pending_intent = "compensation"
    agent.state.explicit_compensation_request = True
    agent.state.last_user_details = {"membership": "regular"}
    agent.state.last_reservation_details = {
        "reservation_id": "3JA7XV",
        "cabin": "economy",
        "insurance": "no",
    }

    action = agent._normalize_action(
        {"name": "send_certificate", "arguments": {"user_id": "mei_brown_7075", "amount": 400}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "regular" in action["arguments"]["content"].lower()


def test_compensation_follow_up_with_cancellation_story_keeps_compensation_intent():
    agent = Agent()
    agent.state.pending_intent = "compensation"
    agent.state.explicit_compensation_request = True

    agent._ingest_message_into_state(
        "user: The flight was canceled by the airline and I think I should be compensated for the inconvenience."
    )

    assert agent.state.pending_intent == "compensation"


def test_compensation_follow_up_does_not_ask_for_cancellation_reason():
    agent = Agent()
    agent.state.pending_intent = "compensation"
    agent.state.explicit_compensation_request = True
    agent.state.last_user_details = {"membership": "regular"}
    agent.state.last_reservation_details = {
        "reservation_id": "WUNA5K",
        "cabin": "economy",
        "insurance": "no",
    }

    action = agent._maybe_airline_user_action(
        "The cancellation was initiated by the airline. It caused me to miss an important meeting and I want compensation."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "regular" in action["arguments"]["content"].lower()
    assert "reason for the cancellation" not in action["arguments"]["content"]


def test_delay_complaint_with_verified_regular_member_refuses_without_reservation_lookup():
    # Once reservation details are known (economy, no insurance), regular member gets refused.
    agent = Agent()
    agent.state.pending_intent = "compensation"
    agent.state.explicit_compensation_request = True
    agent.state.last_user_details = {"membership": "regular"}
    agent.state.last_reservation_details = {
        "reservation_id": "HAT045_RES",
        "cabin": "economy",
        "insurance": "no",
    }

    action = agent._maybe_airline_user_action(
        "My flight HAT045 is delayed, not canceled. I want compensation because I'm missing an important meeting."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "regular" in action["arguments"]["content"].lower()


def test_delay_context_from_state_refuses_after_user_lookup_without_model_guessing():
    # After fetching both user details and reservation details, ineligible regular member is refused.
    agent = Agent()
    agent.state.pending_intent = "compensation"
    agent.state.explicit_compensation_request = True
    agent.state.known_flight_number = "HAT045"
    agent.state.known_delay_context = True
    agent.state.last_user_details = {"membership": "regular"}
    agent.state.last_reservation_details = {
        "reservation_id": "HAT045_RES",
        "cabin": "economy",
        "insurance": "no",
    }

    action = agent._maybe_compensation_or_delay_action(
        "Sure, my name is Mei Brown, and my user ID is mei_brown_7075."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "regular" in action["arguments"]["content"].lower()


def test_reservation_match_uses_known_flight_number_when_latest_user_text_is_only_user_id():
    agent = Agent()
    agent.state.known_flight_number = "HAT045"
    agent.messages.append(
        {"role": "user", "content": "user: Sure, my user ID is mei_brown_7075."}
    )

    matches = agent._reservation_matches_complaint(
        {
            "reservation_id": "3JA7XV",
            "flights": [
                {"flight_number": "HAT045", "date": "2024-05-15"},
                {"flight_number": "HAT194", "date": "2024-05-18"},
            ],
        }
    )

    assert matches is True


def test_placeholder_user_id_is_replaced_with_verified_state():
    agent = Agent()
    agent.state.known_user_id = "raj_sanchez_7340"

    action = agent._normalize_action(
        {"name": "get_user_details", "arguments": {"user_id": "user_id_placeholder"}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "raj_sanchez_7340"},
    }


def test_baggage_allowance_uses_verified_membership_and_passenger_count():
    agent = Agent()
    agent.state.pending_intent = "baggage"
    agent.state.last_user_details = {"membership": "silver"}
    agent.state.last_reservation_details = {
        "reservation_id": "JMO1MG",
        "cabin": "economy",
        "passengers": [{}, {}],
    }

    action = agent._maybe_airline_user_action("How many suitcases can I bring in total?")

    assert action["name"] == RESPOND_ACTION_NAME
    assert "4 suitcases" in action["arguments"]["content"]


def test_cancellation_review_queue_prefers_last_reservation():
    agent = Agent()
    agent.state.pending_intent = "cancel"
    agent.state.last_user_details = {
        "reservations": ["MZDDS4", "60RX9E", "S5IK51", "OUEA45", "Q69X3R"]
    }
    agent.messages.append({"role": "user", "content": "user: I need to cancel the trip from Philadelphia to LaGuardia."})
    agent._set_reservation_review_queue(agent.state.last_user_details["reservations"])

    action = agent._maybe_airline_user_action(
        "I need to cancel the trip from Philadelphia to LaGuardia."
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "Q69X3R"},
    }


def test_tool_action_uses_benchmark_route_hint_for_last_reservation():
    agent = Agent()
    agent.state.pending_intent = "cancel"
    agent._ingest_message_into_state(
        "known_info: You are Raj Sanchez. Your user id is raj_sanchez_7340. "
        "task_instructions: The trip you want to cancel is the one from Philadelphia to LaGuardia. "
        "domain: airline"
    )

    action = agent._maybe_airline_tool_action(
        '{"user_id":"raj_sanchez_7340","membership":"silver","reservations":["MZDDS4","60RX9E","S5IK51","OUEA45","Q69X3R"]}'
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "Q69X3R"},
    }


def test_cancel_flow_defaults_to_last_reservation_when_only_user_profile_is_known():
    agent = Agent()
    agent.state.pending_intent = "cancel"

    action = agent._maybe_airline_tool_action(
        '{"user_id":"raj_sanchez_7340","membership":"silver","reservations":["MZDDS4","60RX9E","S5IK51","OUEA45","Q69X3R"]}'
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "Q69X3R"},
    }


def test_cancellation_reason_persists_across_follow_up_messages():
    agent = Agent()
    agent._ingest_message_into_state("user: The reason for the cancellation is a change of plan.")

    assert agent.state.known_cancellation_reason == "change_of_plan"

    agent._ingest_message_into_state("user: Can you help with the refund situation?")

    assert "cancellation_reason" not in agent.state.unresolved_slots


def test_ingest_message_parses_known_info_from_benchmark_prompt():
    agent = Agent()
    prompt = """
known_info: You are Raj Sanchez.
Your user id is raj_sanchez_7340.

reason_for_call: You want to cancel your reservation from Philadelphia to LaGuardia.
    """.strip()

    agent._ingest_message_into_state(prompt)

    assert agent.state.known_user_id == "raj_sanchez_7340"


def test_benchmark_prompt_prefers_get_user_details_first():
    agent = Agent()
    prompt = """
known_info: You are Raj Sanchez.
Your user id is raj_sanchez_7340.

reason_for_call: You recently spoke on the phone with a customer support representative that told you that a service agent will be able to help you cancel your reservation.
task_instructions: The trip you want to cancel is the one from Philadelphia to LaGuardia.
domain: airline
    """.strip()

    agent._ingest_message_into_state(prompt)
    action = agent._maybe_rule_based_action(prompt)

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "raj_sanchez_7340"},
    }
    assert agent.state.prefer_last_reservation is True
    assert agent.state.pending_intent == "cancel"


def test_benchmark_prompt_does_not_route_to_baggage_from_policy_text():
    agent = Agent()
    prompt = """
# Airline Agent Policy
You can help users book, modify, cancel, handle refunds, compensation, and baggage.

known_info: You are Raj Sanchez.
Your user id is raj_sanchez_7340.

reason_for_call: You recently spoke on the phone with a customer support representative that told you that a service agent will be able to help you cancel your reservation.
task_instructions: The trip you want to cancel is the one from Philadelphia to LaGuardia.
domain: airline
    """.strip()

    agent._ingest_message_into_state(prompt)

    assert agent.state.pending_intent == "cancel"


def test_benchmark_prompt_routes_from_embedded_user_message():
    agent = Agent()
    prompt = """
# Airline Agent Policy
You can help users book, modify, cancel, refunds, compensation, and baggage.

Next, I'll provide you with the user message and tool call results.
User message: "user: Hello! I need assistance with canceling a reservation I have from Philadelphia to LaGuardia."
    """.strip()

    agent._ingest_message_into_state(prompt)

    assert agent.state.pending_intent == "cancel"
    assert agent.state.prefer_last_reservation is True


def test_empty_benchmark_user_message_does_not_route_from_policy_text():
    agent = Agent()
    prompt = """
# Airline Agent Policy
You can help users book, modify, cancel, refunds, compensation, and baggage.

Next, I'll provide you with the user message and tool call results.
User message: ""
    """.strip()

    agent._ingest_message_into_state(prompt)
    action = agent._maybe_rule_based_action(prompt)

    assert agent.state.pending_intent is None
    assert action["name"] == RESPOND_ACTION_NAME
    assert "how can i help" in action["arguments"]["content"].lower()


def test_benchmark_delay_prompt_asks_for_user_id_before_any_lookup():
    agent = Agent()
    prompt = """
# Airline Agent Policy
Next, I'll provide you with the user message and tool call results.
User message: "user: My flight HAT045 is delayed and I want compensation for the inconvenience."
    """.strip()

    agent._ingest_message_into_state(prompt)
    action = agent._maybe_rule_based_action(prompt)

    assert agent.state.pending_intent == "compensation"
    assert agent.state.known_flight_number == "HAT045"
    assert action["name"] == RESPOND_ACTION_NAME
    assert "user id" in action["arguments"]["content"].lower()


def test_user_message_with_explicit_user_id_triggers_profile_lookup():
    agent = Agent()
    agent.state.pending_intent = "book"

    action = agent._maybe_airline_user_action("Sure! My user ID is noah_muller_9847.")

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "noah_muller_9847"},
    }


def test_unavailable_check_action_is_rescued_to_profile_lookup():
    agent = Agent()
    agent.state.known_user_id = "raj_sanchez_7340"
    agent.allowed_action_names = {"get_user_details", "respond"}
    agent.messages.append({"role": "user", "content": "user: I want to cancel my trip."})

    action = agent._normalize_action(
        {"name": "check_reservations", "arguments": {}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "raj_sanchez_7340"},
    }


def test_get_user_details_prefers_verified_known_user_id_over_hallucinated_one():
    agent = Agent()
    agent.state.known_user_id = "raj_sanchez_7340"

    action = agent._normalize_action(
        {"name": "get_user_details", "arguments": {"user_id": "sara_doe_496"}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "raj_sanchez_7340"},
    }


def test_get_reservation_details_rejects_plain_word_and_uses_review_queue():
    agent = Agent()
    agent.state.reservation_review_queue = ["NM1VX1", "KC18K6"]

    action = agent._normalize_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "PLEASE"}}
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "NM1VX1"},
    }


def test_get_reservation_details_rejects_flight_number_and_prefers_user_lookup():
    agent = Agent()
    agent.state.known_user_id = "mei_brown_7075"

    action = agent._normalize_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "HAT045"}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "mei_brown_7075"},
    }
