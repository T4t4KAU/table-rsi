#!/usr/bin/env bash
set -euo pipefail
root=/data1/sdnware/tablersi
export PATH="/home/sdnware/.venvs/vllm-cache-step/bin:/usr/local/cuda/bin:$PATH"
if curl -sf http://127.0.0.1:31003/v1/models >/dev/null; then exit 0; fi
nohup env CUDA_VISIBLE_DEVICES=5,6 /home/sdnware/.venvs/vllm-cache-step/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$root/models/Qwen3-32B" --served-model-name tablersi-qwen3-32b tablersi-qwen3-32b-thinking --reasoning-parser qwen3 --tensor-parallel-size 2 \
  --host 127.0.0.1 --port 31003 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-seqs 12 --no-enable-log-requests \
  > "$root/logs/vllm32-replica.log" 2>&1 < /dev/null &
echo $! > "$root/logs/vllm32-replica.pid"
