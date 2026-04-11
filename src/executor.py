import json
import os
from datetime import datetime
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidRequestError,
    Task,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agent import Agent


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.debug_logging_enabled = (
            os.getenv("TAU2_AGENT_DEBUG_LOGGING", "1").strip().lower() not in {"0", "false", "no"}
        )
        self.debug_log_path = Path(
            os.getenv(
                "TAU2_AGENT_DEBUG_LOG_PATH",
                str(Path(__file__).resolve().parents[1] / "analysis" / "debug" / "agent_trace.jsonl"),
            )
        )

    def _trace(self, event: str, **fields) -> None:
        if not self.debug_logging_enabled:
            return
        payload = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event}
        payload.update(fields)
        try:
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.debug_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        self._trace("executor_execute", task_id=task.id, context_id=context_id)
        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            # The benchmark may rotate task_id on every turn while keeping the
            # conversation-scoped context_id stable. Reuse the agent strictly by
            # context_id so multi-turn memory survives across message/send calls.
            agent = self.agents.get(context_id)
            if not agent:
                agent = Agent()
                agent.session_state["_context_id"] = context_id
                self.agents[context_id] = agent
                self._trace("executor_agent_created", task_id=task.id, context_id=context_id)
            else:
                agent.session_state["_context_id"] = context_id
                self._trace(
                    "executor_agent_reused",
                    task_id=task.id,
                    context_id=context_id,
                    turn_count=agent.turn_count,
                )

            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as exc:
            import traceback as _tb; _tb.print_exc()
            print(f"Task failed with agent error: {exc}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {exc}",
                    context_id=context_id,
                    task_id=task.id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
