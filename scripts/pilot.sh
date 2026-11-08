#!/usr/bin/env bash
set -euo pipefail
root=/data1/sdnware/tablersi
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:31001/health >/dev/null; then break; fi
  sleep 5
done
curl -sf http://127.0.0.1:31001/health >/dev/null
for variant in fixed direct retry; do
  "$root/.venv/bin/python" -m tablersi.cli eval --repo "$root/data/TableAgentBench" --data "$root/data/prepared" --out "$root/runs/dev-pilot-v1/$variant" --split dev --limit 3 --variant "$variant" --workers 3
done
