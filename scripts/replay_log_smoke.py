from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from a2a.types import Message, Part, Role, TextPart

from agent import Agent


class DummyUpdater:
    def __init__(self) -> None:
        self.parts = None

    async def update_status(self, state, message) -> None:
        return None

    async def add_artifact(self, parts, name) -> None:
        self.parts = parts
        self.name = name


def extract_first_user_messages(run_log_path: Path) -> list[str]:
    lines = run_log_path.read_text(errors="ignore").splitlines()
    responses: list[str] = []
    new_task = True
    for line in lines:
        if "user_simulator:_generate_next_message" not in line or "Response:" not in line:
            continue
        msg = line.split("Response:", 1)[1].strip()
        if msg in {"###TRANSFER###", "###STOP###"}:
            new_task = True
            continue
        if new_task:
            responses.append(msg)
            new_task = False
    return responses


async def replay_examples(
    run_log_path: Path, prompt_assets_path: Path, limit: int, model: str
) -> None:
    assets = json.loads(prompt_assets_path.read_text())
    policy = assets["policy"]
    tool_schemas = assets["tool_schemas"]
    examples = extract_first_user_messages(run_log_path)[:limit]

    for i, user_msg in enumerate(examples, 1):
        agent = Agent()
        agent.model = model
        first_prompt = f"""
{policy}
Here's a list of tools you can use (you can use at most one tool at a time):
{json.dumps(tool_schemas, indent=2)}
Please response in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON should contain:
- "name": the tool call function name, or "respond" if you want to respond directly.
- "arguments": the arguments for the tool call, or {{"content": "your message here"}} if you want to respond directly.
You should only use one tool at a time!!
You cannot respond to user and use a tool at the same time!!

Next, I'll provide you with the user message and tool call results.
User message: {json.dumps(user_msg, ensure_ascii=False)}
        """.strip()
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=first_prompt))],
            message_id=uuid4().hex,
            context_id=None,
        )
        updater = DummyUpdater()
        await agent.run(msg, updater)
        action = updater.parts[0].root.data
        print(f"[{i}] USER: {user_msg}")
        print(json.dumps(action, ensure_ascii=False))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay real first-turn benchmark examples from a saved run log.")
    parser.add_argument("--run-log", default="/tmp/run.log")
    parser.add_argument("--prompt-assets", default="/tmp/airline_prompt_assets.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    args = parser.parse_args()

    asyncio.run(
        replay_examples(
            run_log_path=Path(args.run_log),
            prompt_assets_path=Path(args.prompt_assets),
            limit=args.limit,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
