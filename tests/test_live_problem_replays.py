import json
import os
from uuid import uuid4

import pytest
from a2a.types import Message, Part, Role, TextPart

from agent import Agent, RESPOND_ACTION_NAME


LIVE_LLM_ENABLED = bool(os.getenv("TAU2_RUN_LIVE_LLM_TESTS")) and bool(
    os.getenv("OPENAI_API_KEY")
)

pytestmark = pytest.mark.skipif(
    not LIVE_LLM_ENABLED,
    reason="Live LLM regression tests require TAU2_RUN_LIVE_LLM_TESTS=1 and OPENAI_API_KEY.",
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_details",
            "description": "Look up a customer profile by user ID.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reservation_details",
            "description": "Look up a reservation by confirmation number.",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": "Check delay, cancellation, or operational status for a flight on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["flight_number", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reservation",
            "description": "Cancel an existing reservation.",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_reservation_flights",
            "description": "Modify a reservation's cabin or flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {"type": "string"},
                    "cabin": {"type": "string"},
                    "flights": {"type": "array"},
                    "payment_id": {"type": "string"},
                },
                "required": ["reservation_id", "cabin", "flights", "payment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_airports",
            "description": "List airports and their codes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_direct_flight",
            "description": "Search direct flights between airports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_onestop_flight",
            "description": "Search one-stop flights between airports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_reservation",
            "description": "Create a new reservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                    "cabin": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "description": "Escalate the case to a human agent.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


class DummyUpdater:
    def __init__(self) -> None:
        self.artifacts = []

    async def update_status(self, state, message) -> None:
        return None

    async def add_artifact(self, parts, name) -> None:
        self.artifacts.append({"parts": parts, "name": name})


def make_wrapped_first_prompt(user_message: str) -> str:
    policy = """
The first user message contains domain policy and tool schemas.
Follow the policy strictly.
- Use at most one tool at a time.
- Reply directly only with action name "respond".
- Avoid redundant reads.
- Do not confuse flight numbers with reservation IDs.
- Do not take destructive actions unless the user actually asked for them.
    """.strip()
    return f"""
{policy}
Here's a list of tools you can use (you can use at most one tool at a time):
{json.dumps(TOOLS, ensure_ascii=False, indent=2)}
Please response in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON should contain:
- "name": the tool call function name, or "respond" if you want to respond directly.
- "arguments": the arguments for the tool call, or {{"content": "your message here"}} if you want to respond directly.
You should only use one tool at a time!!
You cannot respond to user and use a tool at the same time!!

Next, I'll provide you with the user message and tool call results.
User message: {json.dumps(user_message, ensure_ascii=False)}
    """.strip()


@pytest.mark.asyncio
async def test_live_first_turn_cancel_example_routes_to_reservation_lookup():
    agent = Agent()
    updater = DummyUpdater()
    prompt = make_wrapped_first_prompt(
        "Hello! I would like to cancel my reservation. The confirmation number is EHGLP3. Could you assist me with that?"
    )
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(text=prompt))],
        message_id=uuid4().hex,
        context_id=None,
    )

    await agent.run(msg, updater)
    action = updater.artifacts[-1]["parts"][0].root.data

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "EHGLP3"},
    }


@pytest.mark.asyncio
async def test_live_first_turn_baggage_example_does_not_jump_to_compensation():
    agent = Agent()
    updater = DummyUpdater()
    prompt = make_wrapped_first_prompt(
        "I actually just need to know the total number of suitcases allowed for my reservation. My user ID is anya_garcia_5901, and my confirmation number is JMO1MG. Can you help me find that information?"
    )
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(text=prompt))],
        message_id=uuid4().hex,
        context_id=None,
    )

    await agent.run(msg, updater)
    action = updater.artifacts[-1]["parts"][0].root.data

    assert action["name"] in {"get_reservation_details", "get_user_details", RESPOND_ACTION_NAME}
    if action["name"] == RESPOND_ACTION_NAME:
        assert "compensation eligibility" not in action["arguments"]["content"].lower()
        assert "flight date" not in action["arguments"]["content"].lower()


@pytest.mark.asyncio
async def test_live_first_turn_modify_example_does_not_jump_to_compensation():
    agent = Agent()
    updater = DummyUpdater()
    prompt = make_wrapped_first_prompt(
        "Hi! I need assistance with modifying my upcoming flight. I have a flight from IAH to SEA on May 23, and I would like to push it back to May 24. Also, I want to upgrade the class to business for all passengers. Can you help with that?"
    )
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(text=prompt))],
        message_id=uuid4().hex,
        context_id=None,
    )

    await agent.run(msg, updater)
    action = updater.artifacts[-1]["parts"][0].root.data

    assert action["name"] in {"respond", "get_user_details", "get_reservation_details"}
    if action["name"] == RESPOND_ACTION_NAME:
        assert "compensation eligibility" not in action["arguments"]["content"].lower()
        assert "flight date" not in action["arguments"]["content"].lower()


@pytest.mark.asyncio
async def test_live_first_turn_remove_passenger_example_does_not_jump_to_compensation():
    agent = Agent()
    updater = DummyUpdater()
    prompt = make_wrapped_first_prompt(
        "Hello! I need to remove a passenger named Sophia from my upcoming round trip flights from LAS to DEN, departing on May 19 and returning on May 20. Can you assist me with that?"
    )
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(text=prompt))],
        message_id=uuid4().hex,
        context_id=None,
    )

    await agent.run(msg, updater)
    action = updater.artifacts[-1]["parts"][0].root.data

    assert action["name"] in {"respond", "get_user_details", "get_reservation_details"}
    if action["name"] == RESPOND_ACTION_NAME:
        assert "compensation eligibility" not in action["arguments"]["content"].lower()
        assert "flight date" not in action["arguments"]["content"].lower()


def test_live_llm_insurance_example_must_not_take_destructive_action():
    agent = Agent()
    agent.tools = TOOLS
    agent.tool_names = {tool["function"]["name"] for tool in TOOLS}
    agent.messages.extend(
        [
            {
                "role": "user",
                "content": 'User message: "Hi there! I’d like some assistance with my flight reservation. I believe I added insurance to my upcoming flight, but it’s not showing up online. Could you help me with that? My user ID is sophia_taylor_9065, and my reservation number is PEP4E0."',
            },
            {
                "role": "user",
                "content": 'tool: {"reservation_id":"PEP4E0","user_id":"sophia_taylor_9065","origin":"CLT","destination":"PHX","flight_type":"one_way","cabin":"basic_economy","flights":[{"flight_number":"HAT176","origin":"CLT","destination":"DTW","date":"2024-05-20","price":51},{"flight_number":"HAT097","origin":"DTW","destination":"PHX","date":"2024-05-20","price":77}],"passengers":[{"first_name":"Sophia","last_name":"Taylor","dob":"1999-05-27"}],"payment_history":[{"payment_id":"credit_card_9302073","amount":128}],"created_at":"2024-05-05T05:10:43","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}',
            },
            {
                "role": "user",
                "content": "I appreciate your help, but I really need to clarify my insurance status for my flight instead. Can you confirm if the insurance is added to my reservation with the number PEP4E0?",
            },
        ]
    )

    action = agent._guard_action(agent._normalize_action(agent._call_llm()))

    assert action["name"] not in {"cancel_reservation", "update_reservation_flights"}
