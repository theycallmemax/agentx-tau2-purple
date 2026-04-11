# AgentBeats Tau2 Purple Agent

Улучшенный purple-agent для `τ²-Bench` в `AgentX / AgentBeats`.

Этот репозиторий основан на двух официальных бейзлайнах:
- [`agent-template`](https://github.com/RDI-Foundation/agent-template)
- [`agentbeats-tutorial/scenarios/tau2/agent`](https://github.com/RDI-Foundation/agentbeats-tutorial/tree/main/scenarios/tau2/agent)

Здесь сохранен тот же A2A scaffold, который ожидает AgentBeats, но baseline-логика заменена на более аккуратный `policy-first` agent на базе `langgraph`:
- явный turn graph вместо ручного orchestration loop;
- отдельный шаг анализа перед действием;
- более жесткое требование к JSON action output;
- валидация имени действия;
- safer default behavior при ошибках модели;
- акцент на policy compliance, read-before-write и user-side remediation для `telecom`.

## Структура

```text
src/
├─ agent.py
├─ guard.py
├─ runtime.py
├─ session_state.py
├─ executor.py
└─ server.py
tests/
├─ conftest.py
├─ test_agent.py
└─ test_examples.py
scripts/
└─ run_airline_eval.py
Dockerfile
run.sh
```

## Как это работает

`tau2` evaluator из AgentBeats tutorial шлет purple-agent'у текст, в котором уже есть:
- policy домена;
- JSON-схемы доступных tools;
- история диалога;
- результаты tool calls.

Внутри одного хода агент прогоняет сообщение через `langgraph` pipeline:
1. `transfer_hold` — обязательный второй шаг после `transfer_to_human_agents`;
2. `ingest_input` — обновление session state и runtime facts;
3. `llm_decide` — planner + model call;
4. `finalize_action` — guard rails, post-processing и запись action в историю.

В ответ агент возвращает A2A artifact с JSON формата:

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

Если только что добавил `langgraph` в окружение, сначала обнови зависимости:

```bash
uv sync
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

## Как проверять качество

Есть два быстрых контура проверки.

### 1. Regression tests на примерах

Это unit-style сценарии без реального LLM и без живого benchmark server. Они проверяют:
- что агент выбирает read action перед risky write action;
- что агент чинит невалидный JSON retry-ом;
- что агент не вызывает несуществующую tool;
- что аргументы нормализуются перед отправкой.

Часть prompt-regression проверок опирается на открытый `airline policy` из локальной копии `tau2-bench`, чтобы не потерять критичные правила вроде:
- не предлагать компенсацию proactively;
- верифицировать membership / insurance / flight status;
- не просить reservation ids и airport codes, если это можно разрешить через инструменты и контекст.

Запуск:

```bash
uv run pytest tests/test_examples.py -v
```

### 2. Локальный airline eval через green-agent

Если у тебя уже запущен green-agent из `tau2-bench/src/experiments/agentify_tau_bench`, можно отправить benchmark request прямо из этого репозитория.

Поднять purple-agent:

```bash
uv run src/server.py --host 127.0.0.1 --port 9009
```

Поднять green-agent в другом терминале:

```bash
cd /Users/maksimpiskaev/Проекты/agent_hw/tau2-bench/src/experiments/agentify_tau_bench
uv run tau-bench-agent green
```

Запустить локальный eval запрос:

```bash
uv run python scripts/run_airline_eval.py --green-url http://127.0.0.1:9001 --white-url http://127.0.0.1:9009 --domain airline --task-ids 1 2 3
```

Если `--task-ids` не указывать, green-agent будет использовать все задачи домена.

Практически полезный цикл такой:
1. Гоняешь `tests/test_examples.py`, чтобы не сломать базовую логику.
2. Гоняешь маленький airline eval на 3-10 задач.
3. Разбираешь провальные траектории и улучшаешь policy/prompt/action gating.

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
