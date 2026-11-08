#!/usr/bin/env bash
set -euo pipefail
root=/data1/sdnware/tablersi
for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:31102/health >/dev/null; then break; fi
  sleep 5
done
curl -sf http://127.0.0.1:31102/health >/dev/null
"$root/.venv/bin/python" -m tablersi.cli eval --repo "$root/data/TableAgentBench" --data "$root/data/prepared" --out "$root/runs/dev-v3-32b-thinking/fixed" --split dev --limit 12 --variant fixed --workers 8 --local-url http://127.0.0.1:31102/v1 --local-model tablersi-qwen3-32b-thinking
