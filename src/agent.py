from __future__ import annotations

import json
import os
import re
from typing import Any

import litellm
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()


RESPOND_ACTION_NAME = "respond"

SYSTEM_PROMPT = """You are a purple agent participating in the tau2 benchmark on AgentBeats.

You will receive benchmark context from the evaluator. The evaluator's messages already include:
- the domain policy
- the available tool schemas
- the conversation history
- tool results

Your job is to choose the single best next action.

Hard rules:
- Return exactly one JSON object with keys "name" and "arguments".
- Use exactly one tool at a time.
- If you should answer the user directly, use the action name "respond".
- Never invent tool results or claim success before a tool result confirms it.
- Prefer reading state before mutating it when the situation is uncertain.
- If policy blocks a request, refuse it clearly using "respond".
- In troubleshooting situations, especially telecom, guide the user one concrete step at a time when a user-side step is required.
- Keep direct user responses concise and operational.
"""

THINK_PROMPT = """Analyze the next best move briefly.

Consider:
1. What is the user's actual goal?
2. What facts are confirmed versus assumed?
3. Is there a policy constraint or safety risk?
4. Is the next move a read action, a write action, or a direct response?

Do not output JSON yet."""

ACT_PROMPT = f"""Now return the single best next action as JSON.

Output format:
{{
  "name": "<tool-name-or-{RESPOND_ACTION_NAME}>",
  "arguments": {{}}
}}

If using "{RESPOND_ACTION_NAME}", arguments must be:
{{
  "content": "<message to the user>"
}}
"""

STATUS_ANALYZING = "Analyzing task state..."
STATUS_ACTING = "Selecting next action..."


def _extract_action_names(text: str) -> set[str]:
    names = {
        name
        for name in re.findall(
            r'"function"\s*:\s*\{.*?"name"\s*:\s*"([^"]+)"',
            text,
            flags=re.DOTALL,
        )
    }
    names.add(RESPOND_ACTION_NAME)
    return names


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class Agent:
    def __init__(self):
        self.model = os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1")
        self.temperature = float(os.getenv("TAU2_AGENT_TEMPERATURE", "0"))
        self.messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.allowed_action_names: set[str] | None = None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        if self.allowed_action_names is None:
            self.allowed_action_names = _extract_action_names(input_text)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(STATUS_ANALYZING),
        )

        self.messages.append({"role": "user", "content": input_text})
        reasoning = self._generate_reasoning()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(STATUS_ACTING),
        )

        action_json = self._generate_action(reasoning)
        assistant_content = json.dumps(action_json, ensure_ascii=False)
        self.messages.append({"role": "assistant", "content": assistant_content})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=action_json))],
            name="Action",
        )

    def _generate_reasoning(self) -> str:
        think_messages = self.messages + [{"role": "user", "content": THINK_PROMPT}]
        try:
            completion = litellm.completion(
                model=self.model,
                messages=think_messages,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content or ""
        except Exception:
            return ""

    def _generate_action(self, reasoning: str) -> dict[str, Any]:
        act_messages = self.messages + [
            {"role": "assistant", "content": f"Internal analysis:\n{reasoning}"},
            {"role": "user", "content": ACT_PROMPT},
        ]

        raw_content = ""
        try:
            completion = litellm.completion(
                model=self.model,
                messages=act_messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            raw_content = completion.choices[0].message.content or ""
        except Exception:
            pass

        action = _extract_json_object(raw_content)
        if action is None:
            return self._fallback_action(
                "I ran into an internal formatting issue. Please repeat the last request."
            )
        return self._normalize_action(action)

    def _normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        name = action.get("name")
        arguments = action.get("arguments")

        if not isinstance(name, str) or not name:
            return self._fallback_action(
                "I couldn't determine the next safe action. Could you clarify your request?"
            )

        if not isinstance(arguments, dict):
            arguments = {}

        if (
            self.allowed_action_names is not None
            and name not in self.allowed_action_names
        ):
            return self._fallback_action(
                f"I selected an unavailable action ({name}). Please restate what you need."
            )

        if name == RESPOND_ACTION_NAME:
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                return self._fallback_action(
                    "I need a bit more information before I can help."
                )
            return {"name": name, "arguments": {"content": content.strip()}}

        return {"name": name, "arguments": arguments}

    def _fallback_action(self, content: str) -> dict[str, Any]:
        return {
            "name": RESPOND_ACTION_NAME,
            "arguments": {"content": content},
        }
