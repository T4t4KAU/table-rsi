#!/usr/bin/env bash
set -euo pipefail

ROOT=${TABLERSI_ROOT:-/data1/sdnware/tablersi}
MODEL_PATH=${TABLERSI_MODEL_PATH:-$ROOT/models/Qwen3.6-35B-A3B-FP8}
VLLM_PYTHON=${TABLERSI_VLLM_PYTHON:-/home/sdnware/.venvs/vllm-cache-step/bin/python}
PORT=${TABLERSI_PORT:-31005}
export CUDA_VISIBLE_DEVICES=${TABLERSI_GPUS:-1,7}

# Official FP8 checkpoint; Ampere uses the vLLM Marlin weight-only path.
# https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B-FP8
# https://docs.vllm.ai/en/latest/features/quantization/fp8/
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name tablersi-qwen36-35b-fp8 \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 2 \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --language-model-only --no-enable-log-requests
