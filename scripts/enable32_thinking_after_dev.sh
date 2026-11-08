#!/usr/bin/env bash
set -euo pipefail
root=/data1/sdnware/tablersi
for i in $(seq 1 240); do
  if [ -f "$root/runs/synthetic-repair-32b/results.json" ]; then break; fi
  sleep 5
done
test -f "$root/runs/synthetic-repair-32b/results.json"
# These PID files belong to the two TableRSI services launched in this task.
kill -TERM "$(cat "$root/logs/vllm32.pid")" "$(cat "$root/logs/vllm32-replica.pid")"
sleep 15
bash "$root/scripts/start_model32_thinking.sh"
bash "$root/scripts/start_model32_replica_thinking.sh"
nohup "$root/.venv/bin/python" "$root/scripts/thinking_proxy.py" 31102 http://127.0.0.1:31002 http://127.0.0.1:31003 > "$root/logs/proxy32-thinking.log" 2>&1 < /dev/null &
bash "$root/scripts/dev32_thinking.sh"
