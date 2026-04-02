from typing import Any
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = frozenset(
        [
            "name",
            "description",
            "url",
            "version",
            "capabilities",
            "defaultInputModes",
            "defaultOutputModes",
            "skills",
        ]
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


async def send_text_message(text: str, url: str, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        client = ClientFactory(config).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=None,
        )
        return [event async for event in client.send_message(msg)]


def test_agent_card(agent):
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200
    errors = validate_agent_card(response.json())
    assert not errors, "\n".join(errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    events = await send_text_message("Hello", agent, streaming=streaming)
    assert events, "Agent should respond with at least one event"
