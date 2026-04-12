#!/usr/bin/env bash
# Script to run tests on previously failed tasks only
# Usage: ./run_failed_tasks.sh

set -euo pipefail

# Failed task IDs from the last run (airline_batch_20260412_014754)
# All 32 tasks failed with 0% success rate
FAILED_TASKS="2,3,7,8,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,29,30,32,33,34,35,39,40,41,42,44,47,49"

echo "========================================="
echo "Running tests on previously failed tasks"
echo "========================================="
echo "Task IDs: $FAILED_TASKS"
echo ""

# Run the batch with only the failed tasks
cd "$(dirname "$0")/.."

python scripts/run_airline_batch.py \
  --task-ids $FAILED_TASKS \
  --domain airline \
  --max-steps 30 \
  --output-dir analysis/local_runs/airline_failed_retry_$(date +%Y%m%d_%H%M%S)

echo ""
echo "========================================="
echo "Test run completed!"
echo "========================================="
