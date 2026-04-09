import sys
from pathlib import Path

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL (default: http://localhost:9009)",
    )


@pytest.fixture(scope="session")
def agent(request):
    url = request.config.getoption("--agent-url")

    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        if response.status_code != 200:
            pytest.exit(
                f"Agent at {url} returned status {response.status_code}",
                returncode=1,
            )
    except Exception as exc:
        pytest.exit(f"Could not connect to agent at {url}: {exc}", returncode=1)

    return url
