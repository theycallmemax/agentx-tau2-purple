from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, Part, Role, TextPart

from executor import Executor


def build_test_app():
    skill = AgentSkill(
        id="tau2_task_fulfillment",
        name="Tau2 Task Fulfillment",
        description="Solves tau2 benchmark customer-service tasks by choosing one JSON action at a time.",
        tags=["agentbeats", "tau2", "benchmark", "customer-service"],
        examples=[],
    )
    agent_card = AgentCard(
        name="agentbeats_tau2_policy_agent",
        description="Policy-first purple agent for tau2 benchmark evaluations on AgentBeats.",
        url="http://testserver/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()


def validate_agent_card(card_data: dict) -> list[str]:
    errors: list[str] = []
    required_fields = frozenset(
        {
            "name",
            "description",
            "url",
            "version",
            "capabilities",
            "defaultInputModes",
            "defaultOutputModes",
            "skills",
        }
    )

    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    if "url" in card_data and not (
        card_data["url"].startswith("http://")
        or card_data["url"].startswith("https://")
    ):
        errors.append("Field 'url' must be an absolute URL.")

    return errors


@pytest.mark.asyncio
async def test_agent_card():
    app = build_test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    errors = validate_agent_card(response.json())
    assert not errors, "\n".join(errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(monkeypatch, streaming):
    monkeypatch.setattr(
        "agent.litellm.completion",
        lambda **kwargs: type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "MessageObj",
                                (),
                                {
                                    "tool_calls": [],
                                    "content": '{"name":"respond","arguments":{"content":"hello"}}',
                                },
                            )()
                        },
                    )()
                ]
            },
        )(),
    )

    app = build_test_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://testserver")
        agent_card = await resolver.get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=streaming)).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text="Hello"))],
            message_id=uuid4().hex,
            context_id=None,
        )
        events = [event async for event in client.send_message(msg)]

    assert events, "Agent should respond with at least one event"
