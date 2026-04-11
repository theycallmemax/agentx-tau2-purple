# CLAUDE.md — tau2-purple-starter

## Цель

Решить все 50 airline задач τ²-Bench на 100%.  
Задачи: `tau2-bench/data/tau2/domains/airline/tasks.json`

## Что это

Airline/telecom customer service agent для τ²-Bench (AgentBeats/AgentX).  
Реализует A2A (Agent-to-Agent) протокол: принимает JSON `{name, arguments}` ходы от green-агента и возвращает один DataPart artifact на каждый ход.

## Структура проекта

```
src/
├─ agent.py          # главный orchestrator — LangGraph turn graph
├─ guard.py          # action validation, policy gating, safe reroute
├─ reservation.py    # business logic: pricing, eligibility, inventory
├─ session_state.py  # state tracking — что уже загружено, intent, route
├─ extractors.py     # regex entity extraction (IDs, airport codes, dates)
├─ intent.py         # intent classifier (cancel/modify/status/baggage…)
├─ policy_gate.py    # domain policy rules
├─ planner.py        # optional planning step before LLM call
├─ playbooks.py      # domain-specific prompt blocks
├─ policy_sanitizer.py # убирает из tool descriptions запрещённые паттерны
├─ parsing.py        # JSON parsing, tool schema extraction
├─ runtime.py        # history compression, subtask tracking
├─ executor.py       # action execution layer
└─ server.py         # FastAPI A2A server
scripts/
└─ run_airline_batch.py  # запуск batch eval через green-agent
tests/
├─ test_agent.py
├─ test_examples.py
└─ test_failed_task_regressions.py  # регрессии на конкретные провалы
```

## Архитектура (LangGraph turn graph)

Каждый ход проходит через 4 ноды:

```
transfer_hold → ingest_input → llm_decide → finalize_action
```

1. **transfer_hold** — если предыдущий ход был `transfer_to_human_agents`, вернуть hold message, не вызывать LLM.
2. **ingest_input** — обновить `session_state` из user text и tool payload.
3. **llm_decide** — вызвать LLM с промптом + инструментами, распарсить JSON action.
4. **finalize_action** — прогнать через `guard_action`, записать в историю.

## Ключевые уроки из отладки

### 1. Route-based reservation matching

**Проблема**: LLM самостоятельно выбирает неверный reservation_id (первый в списке),  
не используя маршрут из запроса пользователя.

**Решение (src/reservation.py)**:
- `find_reservation_by_route(state, origin, destination)` итерирует `reservation_inventory`
  и матчит по `entry.origin/destination` ИЛИ по `flights[*].origin/destination` (для connecting flights)
- Это покрывает случай PHL → LGA, где реальный itinerary PHL → CLT → LGA

**Решение (src/guard.py)**:
- Pre-LLM route resolution: если известен маршрут, но нет reservation_id в state,
  агент сам пробегает по reservation_inventory до нахождения match
- Не доверяет произвольному reservation_id от LLM, если маршрут задан, но state не подтвердил

**Решение (src/extractors.py)**:
- `CITY_AIRPORT_PATTERNS` маппит города в IATA коды (Philadelphia → PHL, LaGuardia → LGA и т.д.)
- Важно: "New York" намеренно НЕ в маппинге — требует уточнения (JFK vs LGA vs EWR)

### 2. session_state не должен обновляться по tool payload

**Проблема**: `update_state_from_tool_payload` обновлял `intent`, `task_type` и другие поля  
по содержимому ответа инструмента, из-за чего `cancel` превращался в `baggage` или `status`.

**Правило**: `session_state` обновляется только по реальным user репликам,  
не по `tool:` payload и не по policy envelope.

### 3. A2A artifact format

Evaluator ждёт `DataPart` artifact, а не `TextPart` со строкой JSON.  
Если agent возвращает `TextPart`, evaluator падает с:  
`Expected a data artifact payload, got parts=[Part(root=TextPart(...))]`

Правильный return в `src/agent.py`:
```python
await task_updater.add_artifact(
    parts=[Part(root=DataPart(data=action_dict))],
    name="action",
)
```

### 4. Airline cancel policy (task 1 — canonical example)

Условия для task 1 (`raj_sanchez_7340`, reservation `Q69X3R`, маршрут PHL → LGA):
- `created_at = 2024-05-14 09:52:38`, benchmark now = `2024-05-15 15:00:00` → >24h
- `cabin = economy`, `insurance = no`, `reason = change_of_plan`
- Вывод: отмена **запрещена**, refund не положен

Правильный flow:
1. `get_user_details("raj_sanchez_7340")`
2. Нормализовать маршрут → PHL, LGA
3. Итерировать reservations до нахождения Q69X3R по маршруту
4. `get_reservation_details("Q69X3R")`
5. Проверить `cancel_eligibility` → False
6. Ответить отказом, не вызывать `cancel_reservation`

## Как запускать

```bash
# поднять агента
uv run src/server.py --host 127.0.0.1 --port 19125

# прогнать задачу 1
set -a && source .env && set +a
/path/to/tau2-bench/.venv/bin/python scripts/run_airline_batch.py \
  --white-url http://127.0.0.1:19125 \
  --domain airline \
  --task-ids 1 \
  --max-steps 30 \
  --user-llm openai/gpt-4o | tee /tmp/task1.log
```

## Локальный прогон green → purple

Ниже рабочая последовательность, при которой локальный eval стабильно стартует и пишет понятные артефакты.

### 1. Что должно быть готово

- Репозиторий purple: `/Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter`
- Репозиторий benchmark: `/Users/maksimpiskaev/Проекты/agent_hw/tau2-bench`
- В `tau2-purple-starter/.env` должен быть валидный ключ для провайдера LLM
- В `tau2-purple-starter` должны быть установлены зависимости:

```bash
cd /Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter
uv sync
```

- В `tau2-bench/src/experiments/agentify_tau_bench/.venv` должен существовать python с зависимостями benchmark

### 2. Purple agent

Поднимать purple лучше из `tau2-purple-starter`, не из benchmark-репозитория.

```bash
cd /Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter
set -a && source .env && set +a
uv run src/server.py --host 127.0.0.1 --port 19123 --agent-llm openai/gpt-4o
```

Замечания:
- `19123` — удобный локальный порт для white/purple
- если порт занят, сначала убить старый процесс
- `--agent-llm` можно менять, но для воспроизводимых локальных airline прогонов обычно использовался `openai/gpt-4o`

### 3. Green-side local batch

Запускать batch нужно из purple-репозитория, но python брать из benchmark virtualenv.

```bash
cd /Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter
set -a && source .env && set +a
/Users/maksimpiskaev/Проекты/agent_hw/tau2-bench/src/experiments/agentify_tau_bench/.venv/bin/python \
  scripts/run_airline_batch.py \
  --white-url http://127.0.0.1:19123 \
  --domain airline \
  --task-ids 1 2 3 \
  --max-steps 30 \
  --user-llm openai/gpt-4o | tee /tmp/tau2_airline_batch.log
```

Для одной задачи:

```bash
cd /Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter
set -a && source .env && set +a
/Users/maksimpiskaev/Проекты/agent_hw/tau2-bench/src/experiments/agentify_tau_bench/.venv/bin/python \
  scripts/run_airline_batch.py \
  --white-url http://127.0.0.1:19123 \
  --domain airline \
  --task-ids 1 \
  --max-steps 30 \
  --user-llm openai/gpt-4o | tee /tmp/tau2_airline_task1.log
```

### 4. Где смотреть результаты

Каждый запуск пишет артефакты в:

```bash
/Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter/analysis/local_runs/<run_id>/
```

Самое полезное:
- `summary.json` — итог по batch
- `results.jsonl` — короткая сводка по задачам
- `trajectories/task_XX.json` — полная траектория конкретной задачи
- `batch_internal.log` — внутренний лог orchestrator/user simulator

Дополнительный trace purple-агента:

```bash
/Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter/analysis/debug/agent_trace.jsonl
```

Он включён по умолчанию через:
- `TAU2_AGENT_DEBUG_LOGGING=1`
- `TAU2_AGENT_DEBUG_LOG_PATH=analysis/debug/agent_trace.jsonl`

### 5. Что значит green и purple в этом проекте

- `green` — локальный runner/user-simulator из `tau2-bench`, который оркестрирует benchmark
- `purple` — этот агент, который получает policy + tools и возвращает ровно один action на ход
- `white-url` в local eval фактически должен указывать на поднятый purple A2A server

### 6. Типичные причины, почему локальный прогон "не работает"

- Purple поднят без `source .env`, и LLM вызовы падают по auth
- Purple поднят на одном порту, а `--white-url` смотрит на другой
- Используется не benchmark virtualenv, а системный python
- Остался старый процесс `src/server.py`, и вы смотрите не в ту версию кода
- В логах видно `Expected a data artifact payload`:
  purple должен возвращать `DataPart`, а не `TextPart`
- В логах видно `Simulation terminated prematurely`:
  сначала открой `trajectories/task_XX.json`, затем `analysis/debug/agent_trace.jsonl`

### 7. Рекомендуемый цикл отладки

1. Поднять purple на `127.0.0.1:19123`
2. Прогнать одну задачу `--task-ids <id>`
3. Открыть `trajectories/task_XX.json`
4. Открыть `analysis/debug/agent_trace.jsonl`
5. Починить forced path / state / guard
6. Прогнать сначала одиночную задачу, потом группу задач

### 8. Purple vs tau2-purple-agent

В этом workspace есть два purple-репозитория:
- `tau2-purple-starter` — основной рабочий агент с state machine, guard, reservation logic
- `tau2-purple-agent` — более простой prompt-driven baseline

Для локальной отладки airline задач по умолчанию использовать нужно именно `tau2-purple-starter`.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `TAU2_AGENT_LLM` | `openai/gpt-5.2` | Модель агента |
| `TAU2_AGENT_TEMPERATURE` | `0` | Temperature LLM |
| `TAU2_AGENT_MAX_CONTEXT_MESSAGES` | `120` | Обрезка истории |
| `TAU2_AGENT_PLAN_MAX_TURNS` | `8` | Макс. ходов планировщика |
| `TAU2_AGENT_DEBUG_LOGGING` | `1` | Запись trace в JSONL |
| `TAU2_AGENT_DEBUG_LOG_PATH` | `analysis/debug/agent_trace.jsonl` | Путь trace файла |

## Тесты

```bash
# все тесты
uv run pytest tests/ -q

# только регрессии на конкретные провалы
uv run pytest tests/test_failed_task_regressions.py -q
```

Тестов: ~55. Регрессии покрывают:
- Task 1: PHL → LGA cancel denial
- Route matching с connecting flights
- session_state не портится tool payload
- JSON parsing, action normalization, guard reroute

## Известные ограничения

- `BENCHMARK_NOW = 2024-05-15T15:00:00` захардкожен в `src/reservation.py` — это намеренно для воспроизводимости benchmark
- `find_reservation_by_route` матчит первую найденную бронь — если у пользователя несколько с одинаковым маршрутом, нужен tie-breaking по дате
- tau2-purple-agent (соседний репо) использует только prompt-based matching без алгоритмического слоя, поэтому хрупче

## Группировка задач по типу

### Группа A: Cancel eligibility — отказ в отмене (политика)
Группа A больше не является чисто `deny-only`: в актуальном `tau2-bench/data/tau2/domains/airline/tasks.json` внутри неё есть и mixed-intent, и aggregate-cancel кейсы. **Tasks: 0, 1, 26, 28, 35, 39, 41, 43, 45, 47, 48, 49**

| Task | Суть |
|------|------|
| 0 | Не отменяет: «не нужна страховка, прошлый агент сказал» |
| 1 | Не отменяет: «другой агент одобрил» + маршрут PHL→LGA |
| 26 | Не отменяет/возвращает деньги при несоблюдении условий |
| 28 | Не отменяет, пользователь требует gift card / certificate |
| 35 | Сначала не отменяет под давлением, затем продолжает диалог и бронирует новый JFK→SFO на May 24, economy, второй по дешевизне |
| 39 | Aggregate-cancel: по `user_id` обходит все брони и отменяет только eligible upcoming reservations, даже если refund не положен |
| 41 | Aggregate-check: по `user_id` обходит все брони с 1 пассажиром и не отменяет ни одну, если eligible нет |
| 43 | Итерирует дубликаты на один день, проверяет нужную бронь и не отменяет неeligible варианты |
| 45 | Не отменяет (семейная авария) |
| 47 | Insurance покрывает только health/weather, не всё |
| 48 | Бронь была >24ч назад, пользователь врёт «10 часов» |
| 49 | Пользователь врёт о наличии страховки |

Практически это три подпаттерна:
- `0, 1, 26, 28, 45, 47, 48, 49`: deny-only. Обычно `get_user_details/get_reservation_details → cancel_eligibility=False → respond(отказ)`. Никаких write-calls.
- `35`: mixed-intent. Сначала policy denial на cancellation, затем переключение в booking flow и `book_reservation` на новый рейс.
- `39, 41, 43`: aggregate / iterative lookup. Агент не должен застревать на просьбе повторить `user_id`; нужно итерировать `get_reservation_details` по найденным reservation ids и принимать решение по каждой броне.

Важно: локальные заметки ниже должны сверяться не с этой markdown-секцией, а с `evaluation_criteria` из `tau2-bench/data/tau2/domains/airline/tasks.json`. Для `35` и `39` именно `tasks.json` является источником истины.

### Группа B: Status/Compensation flow (задержки, отмены рейсов)
**Tasks: 2, 4, 5, 27, 38**

| Task | Суть |
|------|------|
| 2 | Задержка + смена темы на новое бронирование; пользователь не помнит reservation |
| 4 | Пользователь врёт о business cabin и cancelled flight — отказать в компенсации |
| 5 | Задержка HAT045, пользователь утверждает Gold (на самом деле Regular) |
| 27 | Правильная компенсация за реально задержанный рейс |
| 38 | Агент проверяет все детали прежде чем предложить компенсацию |

Паттерн: `get_flight_status → итерация reservations → find matching → policy check → compensation/denial`.

### Группа C: Информационные запросы (без write-actions)
**Tasks: 3, 6, 14, 46**

| Task | Суть |
|------|------|
| 3 | Проверить membership (Gold vs Silver) + число бесплатных багажей |
| 6 | Добавить страховку — невозможно по политике |
| 14 | Найти дешёвые рейсы + суммы gift card / certificate балансов |
| 46 | Вернуть деньги за страховку — невозможно |

Паттерн: `get_user_details → get_reservation_details → respond(факт)`. Никаких write-calls.

### Группа D: Поиск брони по маршруту / flight_number (route disambiguation)
**Tasks: 1, 2, 4, 5, 8, 38**

Пользователь не знает reservation ID — агент итерирует все брони до match.
Паттерн: `get_user_details → loop(get_reservation_details) → route/flight match → action`.

### Группа E: Многозадачные разговоры (cancel + modify + info в одном диалоге)
**Tasks: 7, 9, 17, 22, 37, 44**

| Task | Суть |
|------|------|
| 7 | Cancel двух броней + upgrade basic_economy→business перед отменой |
| 9 | Cancel двух + modify одной на nonstop |
| 17 | 3 изменения одновременно |
| 22 | Multiple actions |
| 37 | Две отмены не разрешены + 1 upgrade |
| 44 | Условия по длительности + cancel/upgrade |

Паттерн: `subtask_queue[cancel, cancel, modify]` — агент обрабатывает все по порядку, не закрывает разговор пока queue не пуст.

### Группа F: Изменение рейса (cabin / route / date)
**Tasks: 10, 11, 12, 13, 15, 16, 18, 19, 29, 30, 31, 32, 33, 34, 36, 40**

| Task | Суть |
|------|------|
| 10 | Нельзя поменять cabin только для части рейсов в брони |
| 11 | Нельзя изменить количество пассажиров |
| 12 | Нельзя поменять cabin только для одного пассажира |
| 13 | Нельзя изменить origin/destination |
| 15 | Найти дешёвый Economy рейс на следующий день (несколько аэропортов) |
| 16 | То же, без нескольких аэропортов |
| 18 | Даунгрейд нескольких броней + расчёт суммарной экономии |
| 19 | Basic economy нельзя изменить → cancel с insurance |
| 29 | Сложная смена (arrival <7am + дешевейший Economy) |
| 30 | Nonstop вместо one-stop + удалить багаж (нельзя) |
| 31 | Смена рейса, если стоит <$100 |
| 32 | basic_economy→economy→ смена рейса |
| 33 | Смена дат + business + страховка закрывает разницу |
| 34 | Много изменений, в итоге слишком дорого → cancel |
| 36 | Смена даты (давление «сложная ситуация») |
| 40 | Смена имени пассажира (нельзя) |

Паттерн: `get_reservation_details → search flights → calculate → confirm → update_reservation_flights`.

### Группа G: Новое бронирование
**Tasks: 8, 20, 21, 23, 24, 25**

| Task | Суть |
|------|------|
| 8 | Забронировать тот же рейс ORD→PHL с дополнительным пассажиром |
| 20 | Бронь с ограничением по времени вылета + оплата |
| 21 | Самый короткий return рейс + добавить baggage |
| 23 | Бронь + one-certificate-per-reservation |
| 24 | Открытый поиск + ограничения оплаты |
| 25 | Бронь только с одним пассажиром + certificate |

### Группа H: Baggage
**Tasks: 21, 30**

| Task | Суть |
|------|------|
| 21 | Добавить 1 baggage |
| 30 | Удалить baggage (нельзя) |

### Ключевые наблюдения для кода

| Паттерн | Затронутые группы | Где чинить |
|---------|-------------------|------------|
| Route/flight disambiguation (итерация броней) | D, B | agent.py pre-LLM subflow |
| Subtask queue не пустеет (преждевременное закрытие) | E | runtime.py termination_controller |
| Social pressure (пользователь врёт/давит) | A, C | guard.py policy gate |
| Cancel+upgrade в одном flow | E, A | guard.py cancel allowlist |
| Compensation без write-action | B | guard.py status_compensation |
| Search contamination (search в cancel/status) | B, D | guard.py search block |

Группы A и F покрывают ~60% всех задач. Группа D — сквозной паттерн, влияет на B, E, G.

## Что делать при новом провале задачи

1. Сохрани лог: `tee /tmp/task_N.log`
2. Найди первый `get_reservation_details(WRONG_ID)` в логе
3. Проверь, какой маршрут пользователь дал и что вернул `find_reservation_by_route`
4. Добавь регрессионный тест в `tests/test_failed_task_regressions.py`
5. Починь в минимальном месте (guard/reservation/session_state), не трогай промпт если нет крайней нужды
6. Прогони `pytest tests/test_failed_task_regressions.py -q` — должно быть зелёным
7. Перезапусти сервер и повтори задачу end-to-end
