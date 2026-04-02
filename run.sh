#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec uv run src/server.py "$@"
fi

exec uv run src/server.py --host "${HOST:-0.0.0.0}" --port "${AGENT_PORT:-9009}"
