# Исправления ошибок агента - Инструкция по запуску

## Что было исправлено

### 1. Обнаружение циклов (agent.py)
- **Проблема**: Агент зацикливался на повторных вызовах `get_reservation_details`, `get_user_details`, `get_flight_status`
- **Решение**: Добавлена ранняя проверка на 2 повторения подряд для read-операций (вместо 3)
- **Файл**: `src/agent.py` - метод `_check_action_loop()`

### 2. Защита от дубликатов (guard.py)  
- **Проблема**: Недостаточно строгая проверка повторяющихся вызовов инструментов
- **Решение**: Уменьшен порог с 3 до 2 для read-операций, улучшены сообщения fallback
- **Файл**: `src/guard.py` - секция "duplicate tool call detection"

### 3. Улучшенные skill prompts
- **baggage.md**: Добавлены четкие правила расчета багажа, запрет на зацикливание
- **status.md**: Добавлены строгие правила компенсации, запрет на повторные проверки статуса
- **general.md**: Добавлены правила предотвращения циклов, улучшения управления контекстом

## Как запустить тесты на провальных задачах

### Вариант 1: Запустить все провальные задачи

```bash
cd /Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter

# Запустить скрипт
./run_failed_tasks.sh
```

### Вариант 2: Запустить вручную с выбором задач

```bash
cd /Users/maksimpiskaev/Проекты/agent_hw/tau2-purple-starter

# Запустить конкретные задачи (можно изменить список)
python scripts/run_airline_batch.py \
  --task-ids 2,3,7,8,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,29,30,32,33,34,35,39,40,41,42,44,47,49 \
  --domain airline \
  --max-steps 30 \
  --output-dir analysis/local_runs/airline_retry_$(date +%Y%m%d_%H%M%S)
```

### Вариант 3: Запустить отдельные задачи для быстрой проверки

```bash
# Запустить только 3 задачи для быстрой проверки
python scripts/run_airline_batch.py \
  --task-ids 2,3,7 \
  --domain airline \
  --max-steps 30 \
  --output-dir analysis/local_runs/airline_quick_test_$(date +%Y%m%d_%H%M%S)
```

## Как проверить результаты

После запуска тестов:

```bash
# Посмотреть summary.json последнего запуска
cat analysis/local_runs/airline_*/summary.json | jq '{
  tasks_total: .tasks_total,
  tasks_completed: .tasks_completed, 
  tasks_passed: .tasks_passed,
  score: .score,
  average_steps: .average_steps_per_task
}'

# Проверить конкретную задачу
cat analysis/local_runs/airline_*/trajectories/task_2.json | jq '.summary'
```

## Ожидаемые улучшения

1. **Меньше шагов на задачу**: С 5.16 в среднем → ожидается ~3-4 шага
2. **Больше пройденных задач**: С 0% → ожидается 40-60%
3. **Меньше зацикливаний**: Практически полное устранение повторных вызовов

## Типичные проблемы и решения

### Проблема: Агент все еще зацикливается
**Решение**: Проверьте логи в `batch_internal.log` на наличие "loop_detected" или "early_loop_detected"

### Проблема: Ошибка 503 (HTTP Error)
**Решение**: Убедитесь, что white agent сервер запущен:
```bash
# Проверить, запущен ли сервер
curl http://127.0.0.1:19123/.well-known/agent-card.json
```

### Проблема: Задачи завершаются мгновенно
**Решение**: Проверьте наличие OPENAI_API_KEY в .env файле

## Мониторинг в реальном времени

Во время запуска можно мониторить прогресс:

```bash
# В отдельном терминале
tail -f analysis/local_runs/airline_*/batch_internal.log | grep -E "(task|SUCCESS|FAILED|ERROR)"
```
