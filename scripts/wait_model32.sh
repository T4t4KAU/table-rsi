#!/usr/bin/env bash
set -euo pipefail
root=/data1/sdnware/tablersi
for i in $(seq 1 240); do
  if grep -q '"completed"' "$root/logs/model_download32.log"; then
    bash "$root/scripts/start_model32.sh"
    exit 0
  fi
  if ! kill -0 1868708 2>/dev/null; then echo 'ModelScope downloader exited without success'; exit 1; fi
  sleep 10
done
echo 'Model download timeout'; exit 1
