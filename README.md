# AgentBeats Tau2 Purple Agent

Улучшенный purple-agent для `τ²-Bench` в `AgentX / AgentBeats`.

Этот репозиторий основан на двух официальных бейзлайнах:
- [`agent-template`](https://github.com/RDI-Foundation/agent-template)
- [`agentbeats-tutorial/scenarios/tau2/agent`](https://github.com/RDI-Foundation/agentbeats-tutorial/tree/main/scenarios/tau2/agent)

Здесь сохранен тот же A2A scaffold, который ожидает AgentBeats, но baseline-логика заменена на более аккуратный `policy-first` agent:
- отдельный шаг анализа перед действием;
- более жесткое требование к JSON action output;
- валидация имени действия;
- safer default behavior при ошибках модели;
- акцент на policy compliance, read-before-write и user-side remediation для `telecom`.

## Структура

```text
src/
├─ agent.py
├─ executor.py
└─ server.py
tests/
├─ conftest.py
└─ test_agent.py
Dockerfile
run.sh
```

## Как это работает

`tau2` evaluator из AgentBeats tutorial шлет purple-agent'у текст, в котором уже есть:
- policy домена;
- JSON-схемы доступных tools;
- история диалога;
- результаты tool calls.

В ответ этот агент возвращает A2A artifact с JSON формата:

```json
{
  "name": "tool_name_or_respond",
  "arguments": {}
}
```

Если нужно ответить пользователю напрямую, агент использует:

```json
{
  "name": "respond",
  "arguments": {
    "content": "..."
  }
}
```

## Локальный запуск

```bash
uv sync
export OPENAI_API_KEY=...
uv run src/server.py --host 127.0.0.1 --port 9019
```

Проверка A2A card:

```bash
curl http://127.0.0.1:9019/.well-known/agent-card.json
```

## Локальная проверка с tutorial evaluator

Самый удобный smoke test — запустить официальный green-agent из `agentbeats-tutorial` и направить его на этот purple-agent.

Сценарий `tau2` в tutorial использует этот wire format:
- green-agent оркестрирует `tau2` benchmark;
- purple-agent выдает по одному JSON action на ход.

## Docker

```bash
docker build --platform linux/amd64 -t ghcr.io/<your-user>/<repo>:latest .
docker run -p 9009:9009 --env OPENAI_API_KEY=$OPENAI_API_KEY ghcr.io/<your-user>/<repo>:latest --host 0.0.0.0 --port 9009
```

## Что нужно для сабмита в AgentX / AgentBeats

Минимально:
- публичный GitHub repo;
- README;
- A2A-compatible purple agent;
- публично доступный endpoint или опубликованный Docker image, который можно поднять;
- валидный `.well-known/agent-card.json`.

Практически:
1. запушить этот репозиторий в GitHub;
2. опубликовать Docker image в `ghcr.io`;
3. зарегистрировать агента в AgentBeats;
4. запустить оценки на `τ²-Bench`.

## Что улучшать дальше

- domain-specific prompting для `airline`, `retail`, `telecom`;
- более строгий action ranker поверх одного candidate generation;
- lightweight memory compression, чтобы не раздувать контекст;
- отдельная telecom-стратегия для guided troubleshooting.
