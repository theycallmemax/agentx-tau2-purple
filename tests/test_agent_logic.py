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


def test_opening_turn_routes_reservation_number_to_lookup():
    agent = Agent()
    agent.turn_count = 1
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
    agent.turn_count = 1
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
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My delayed flight HAT045 deserves compensation."',
        }
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "x"}}
    )
    assert action == {
        "name": "get_flight_status",
        "arguments": {"flight_number": "HAT045", "date": "2024-05-15"},
    }


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
