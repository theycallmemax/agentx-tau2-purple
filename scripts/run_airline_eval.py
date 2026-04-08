from __future__ import annotations

import argparse
import asyncio
import json
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


async def send_a2a_text(url: str, text: str) -> list[object]:
    async with httpx.AsyncClient(timeout=300) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        client = ClientFactory(config).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=None,
        )
        return [event async for event in client.send_message(msg)]


def build_request(
    white_agent_url: str,
    domain: str,
    task_ids: list[str] | None,
    max_steps: int,
    user_llm: str,
    temperature: float,
) -> str:
    env_config = {
        "domain": domain,
        "task_ids": task_ids,
        "max_steps": max_steps,
        "user_llm": user_llm,
        "user_llm_args": {"temperature": temperature},
    }
    return f"""
Your task is to instantiate tau-bench to test the agent located at:
<white_agent_url>
{white_agent_url}
</white_agent_url>
You should use the following env configuration:
<env_config>
{json.dumps(env_config, indent=2)}
</env_config>
    """.strip()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local airline eval through the green agent.")
    parser.add_argument("--green-url", default="http://127.0.0.1:9001")
    parser.add_argument("--white-url", default="http://127.0.0.1:9009")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--user-llm", default="openai/gpt-4o")
    parser.add_argument("--user-temperature", type=float, default=0.0)
    args = parser.parse_args()

    request_text = build_request(
        white_agent_url=args.white_url,
        domain=args.domain,
        task_ids=args.task_ids,
        max_steps=args.max_steps,
        user_llm=args.user_llm,
        temperature=args.user_temperature,
    )
    events = await send_a2a_text(args.green_url, request_text)
    for event in events:
        print(event.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
