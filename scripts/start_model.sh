#!/usr/bin/env bash
set -euo pipefail
root=/data1/sdnware/tablersi
export PATH="/home/sdnware/.venvs/vllm-cache-step/bin:/usr/local/cuda/bin:$PATH"
if curl -sf http://127.0.0.1:31001/v1/models >/dev/null; then
  echo 'Model service already running on 31001.'
  exit 0
fi
nohup env CUDA_VISIBLE_DEVICES=1 /home/sdnware/.venvs/vllm-cache-step/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$root/models/Qwen3-8B" --served-model-name tablersi-qwen3-8b \
  --host 127.0.0.1 --port 31001 --max-model-len 32768 \
  --gpu-memory-utilization 0.82 --max-num-seqs 8 --no-enable-log-requests \
  > "$root/logs/vllm.log" 2>&1 < /dev/null &
echo $! > "$root/logs/vllm.pid"
