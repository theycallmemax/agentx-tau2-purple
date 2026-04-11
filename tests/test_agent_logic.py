import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from a2a.types import Message, Part, Role, TextPart

from agent import (
    Agent,
    RESPOND_ACTION_NAME,
    TRANSFER_HOLD_MESSAGE,
    _extract_json_object,
    _extract_openai_tools,
    _normalize_respond_content,
)


class DummyUpdater:
    def __init__(self):
        self.status_updates = []
        self.artifacts = []

    async def update_status(self, state, message):
        self.status_updates.append((state, message))

    async def add_artifact(self, parts, name):
        self.artifacts.append({"parts": parts, "name": name})


def make_message(text: str) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(text=text))],
        message_id=uuid4().hex,
        context_id=None,
    )


def test_extract_openai_tools_accepts_flat_tool_schemas():
    text = """
Policy text...
[
  {"name": "get_user_details", "description": "lookup", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}},
  {"name": "transfer_to_human_agents", "description": "transfer", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}
]
"""
    tools = _extract_openai_tools(text)
    assert tools is not None
    assert [tool["function"]["name"] for tool in tools] == [
        "get_user_details",
        "transfer_to_human_agents",
    ]


def test_extract_json_object_handles_code_fence_and_extra_text():
    payload = """
```json
{"name":"respond","arguments":{"content":"hello"}}
```
trailing text
"""
    assert _extract_json_object(payload) == {
        "name": "respond",
        "arguments": {"content": "hello"},
    }


def test_extract_json_object_handles_stringified_json_payload():
    payload = '"{\\"name\\":\\"respond\\",\\"arguments\\":{\\"content\\":\\"hello\\"}}"'
    assert _extract_json_object(payload) == {
        "name": "respond",
        "arguments": {"content": "hello"},
    }


def test_normalize_respond_content_unwraps_nested_respond_json():
    nested = '{"name":"respond","arguments":{"content":"Booked successfully."}}'
    assert _normalize_respond_content(nested) == "Booked successfully."


def test_normalize_action_rejects_unknown_tool():
    agent = Agent()
    agent.tool_names = {"get_user_details"}
    action = agent._normalize_action({"name": "delete_everything", "arguments": {}})
    assert action["name"] == RESPOND_ACTION_NAME
    assert "unavailable action" in action["arguments"]["content"]


def test_guard_action_converts_reservation_id_misused_as_user_id():
    agent = Agent()
    agent.messages.append(
        {"role": "user", "content": "user: Please cancel reservation EHGLP3."}
    )
    action = agent._guard_action(
        {"name": "get_user_details", "arguments": {"user_id": "EHGLP3"}}
    )
    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "EHGLP3"},
    }


def test_guard_action_blocks_placeholder_user_id():
    agent = Agent()
    agent.messages.append(
        {"role": "user", "content": "user: I need help canceling my booking."}
    )
    action = agent._guard_action(
        {"name": "get_user_details", "arguments": {"user_id": "user_id_placeholder"}}
    )
    assert action["name"] == RESPOND_ACTION_NAME
    assert "user ID or reservation number" in action["arguments"]["content"]


def test_guard_action_blocks_hallucinated_user_id_not_present_in_context():
    agent = Agent()
    agent.messages.append(
        {"role": "user", "content": "user: I need help with a cancellation."}
    )
    action = agent._guard_action(
        {"name": "get_user_details", "arguments": {"user_id": "sara_doe_496"}}
    )
    assert action["name"] == RESPOND_ACTION_NAME
    assert "user ID or reservation number" in action["arguments"]["content"]


def test_guard_action_rejects_example_user_id_even_after_policy_examples():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My user ID is raj_sanchez_7340. Please cancel my booking."',
        }
    )
    action = agent._guard_action(
        {"name": "get_user_details", "arguments": {"user_id": "sara_doe_496"}}
    )
    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "raj_sanchez_7340"},
    }


def test_opening_turn_routes_reservation_number_to_lookup():
    agent = Agent()
    agent.turn_count = 2
    agent.messages.append(
        {"role": "user", "content": 'User message: "Cancel reservation EHGLP3."'}
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "x"}}
    )
    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "EHGLP3"},
    }


def test_opening_turn_routes_ambiguous_booking_to_airport_list():
    agent = Agent()
    agent.turn_count = 2
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to book a one-way flight from New York to Seattle on May 20th."',
        }
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "x"}}
    )
    assert action == {"name": "list_all_airports", "arguments": {}}


def test_opening_turn_routes_compensation_with_flight_number_to_status():
    agent = Agent()
    agent.turn_count = 2
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My delayed flight HAT045 deserves compensation."',
        }
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "x"}}
    )
    assert action["name"] == RESPOND_ACTION_NAME
    assert "flight date" in action["arguments"]["content"]


def test_opening_turn_routes_baggage_question_to_lookup_request():
    agent = Agent()
    agent.turn_count = 2
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "How many suitcases am I allowed to take on my upcoming flight?"',
        }
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "x"}}
    )
    assert action["name"] == RESPOND_ACTION_NAME
    assert "baggage allowance" in action["arguments"]["content"]


def test_guard_action_blocks_get_flight_status_without_real_status_intent():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to book a flight from SFO to New York."',
        }
    )
    action = agent._guard_action(
        {
            "name": "get_flight_status",
            "arguments": {"flight_number": "HAT001", "date": "2024-05-15"},
        }
    )
    assert action["name"] == RESPOND_ACTION_NAME
    assert "flight number" in action["arguments"]["content"]


def test_call_llm_uses_tool_call_payload(monkeypatch):
    agent = Agent()
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="get_user_details",
            arguments=json.dumps({"user_id": "abc"}),
        )
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))]
    )
    monkeypatch.setattr("agent.litellm.completion", lambda **kwargs: completion)

    assert agent._call_llm() == {
        "name": "get_user_details",
        "arguments": {"user_id": "abc"},
    }


def test_call_llm_unwraps_nested_respond_json_content(monkeypatch):
    agent = Agent()
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[],
                    content='{"name":"respond","arguments":{"content":"{\\"name\\":\\"respond\\",\\"arguments\\":{\\"content\\":\\"hello\\"}}"}}',
                )
            )
        ]
    )
    monkeypatch.setattr("agent.litellm.completion", lambda **kwargs: completion)

    assert agent._call_llm() == {
        "name": "respond",
        "arguments": {"content": "hello"},
    }


def test_update_state_from_tool_payload_extracts_reservations_and_payments():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"user_id":"mohamed_silva_9265","reservations":["K1NW8N"],"payment_methods":{"gift_card_1":{"id":"gift_card_1"},"certificate_2":{"id":"certificate_2"}}}'
    )
    assert agent.session_state["user_id"] == "mohamed_silva_9265"
    assert agent.session_state["loaded_user_details"] is True
    assert agent.session_state["known_reservation_ids"] == ["K1NW8N"]
    assert agent.session_state["known_payment_ids"] == [
        "gift_card_1",
        "certificate_2",
    ]


def test_update_state_from_tool_payload_extracts_payment_balances():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"user_id":"mohamed_silva_9265","payment_methods":{"gift_card_1":{"id":"gift_card_1","amount":198},"certificate_2":{"id":"certificate_2","amount":250}}}'
    )
    assert agent.session_state["known_payment_balances"] == {
        "gift_card_1": 198.0,
        "certificate_2": 250.0,
    }


def test_opening_turn_balance_intent_routes_to_user_lookup():
    agent = Agent()
    agent.turn_count = 2
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My user ID is mohamed_silva_9265. What are my gift card and certificate balances?"',
        }
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "x"}}
    )
    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "mohamed_silva_9265"},
    }


def test_guard_action_responds_with_known_balances():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["known_payment_balances"] = {
        "gift_card_1": 198.0,
        "certificate_2": 250.0,
        "credit_card_9": 0.0,
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Can you tell me my gift card and certificate balances?"',
        }
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "x"}}
    )
    assert action["name"] == RESPOND_ACTION_NAME
    assert "gift_card_1" in action["arguments"]["content"]
    assert "certificate_2" in action["arguments"]["content"]


def test_guard_action_blocks_flight_number_used_as_reservation_id():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I want flight HAT083."',
        }
    )
    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "HAT083"}}
    )
    assert action["name"] == RESPOND_ACTION_NAME
    assert "flight number" in action["arguments"]["content"]


def test_messages_for_model_injects_verified_entities_constraints():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My user ID is raj_sanchez_7340 and reservation is EHGLP3."',
        }
    )

    messages = agent._messages_for_model()
    joined = "\n".join(
        msg["content"] for msg in messages if isinstance(msg.get("content"), str)
    )

    assert "Verified entities from the user conversation" in joined
    assert "raj_sanchez_7340" in joined
    assert "EHGLP3" in joined


def test_inventory_can_infer_other_reservation_by_date():
    agent = Agent()
    agent.session_state["reservation_id"] = "D1EW9B"
    agent.session_state["reservation_inventory"] = {
        "D1EW9B": {
            "origin": "ATL",
            "destination": "JFK",
            "dates": ["2024-05-17"],
            "created_at": "2024-05-04T07:38:29",
            "status": None,
            "passenger_count": 1,
        },
        "9HBUV8": {
            "origin": "DEN",
            "destination": "BOS",
            "dates": ["2024-05-17"],
            "created_at": "2024-05-12T17:08:42",
            "status": None,
            "passenger_count": 1,
        },
    }
    inferred = agent._infer_reservation_from_inventory(
        "Please check the other flight I have on 2024-05-17."
    )
    assert inferred == "9HBUV8"


def test_active_tools_returns_all_tools():
    """_active_tools should always return all tools, never filter by intent."""
    agent = Agent()
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in (
            "get_user_details",
            "get_reservation_details",
            "cancel_reservation",
            "book_reservation",
            "search_direct_flight",
            "transfer_to_human_agents",
        )
    ]
    agent.session_state["reservation_id"] = "K1NW8N"

    active = agent._active_tools()

    tool_names = {t["function"]["name"] for t in active}
    assert "cancel_reservation" in tool_names
    assert "get_reservation_details" in tool_names
    assert "book_reservation" in tool_names  # all tools always available


@pytest.mark.asyncio
async def test_transfer_emits_hold_message_on_next_turn(monkeypatch):
    agent = Agent()
    updater = DummyUpdater()
    first_turn_text = """
Policy
[
  {"name": "transfer_to_human_agents", "description": "transfer", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
  {"name": "get_user_details", "description": "lookup", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}
]
"""

    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda: {"name": "transfer_to_human_agents", "arguments": {"summary": "needs help"}},
    )

    await agent.run(make_message(first_turn_text), updater)
    first_action = updater.artifacts[-1]["parts"][0].root.data
    assert first_action["name"] == "transfer_to_human_agents"

    await agent.run(make_message("user: hello?"), updater)
    second_action = updater.artifacts[-1]["parts"][0].root.data
    assert second_action == {
        "name": "respond",
        "arguments": {"content": TRANSFER_HOLD_MESSAGE},
    }
