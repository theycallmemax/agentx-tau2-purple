import argparse
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the tau2 purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1"),
        help="LLM model to use through LiteLLM",
    )
    args = parser.parse_args()

    os.environ.setdefault("TAU2_AGENT_LLM", args.agent_llm)

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
        url=args.card_url or f"http://{args.host}:{args.port}/",
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

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
