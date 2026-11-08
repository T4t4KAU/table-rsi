# Script index

Run these scripts from the repository root after installing the package.
Python modules under `src/tablersi` are the supported library and CLI surface.

## General tools

| Script | Purpose | Additional requirements |
|---|---|---|
| `import_tables.py` | Build an inference-only store in a new directory | CSV/XLS/XLSX source files |
| `download_model.py` | ModelScope download with `--model` and explicit `--out` | `download` extra; model storage |
| `download_model32.py` | Qwen3-32B default for the same downloader | `download` extra; explicit `--out` |
| `compare_runs.py` | Paired cluster bootstrap over recorded runs | Completed compatible run directories |
| `score_answer_revision.py` | Development-only paired scoring of drafts and final answers | Completed development run; scorer credentials |
| `repair_probe.py` | Synthetic literal-repair/structural-abstention diagnostic | Local model endpoint; explicit `--out` |

## Cooperative study

`run_cooperative_pipeline.py` runs the isolated v1 teaching/search/audit pipeline;
`cooperative_status.py` reads its progress without modifying results. See
`docs/COOPERATIVE_STUDY.md` for partition and promotion rules.

`start_model36_fp8.sh` records the v2 Qwen3.6-35B-A3B-FP8 serving configuration.
Its defaults refer to the experiment server. Set `TABLERSI_ROOT`,
`TABLERSI_MODEL_PATH`, `TABLERSI_VLLM_PYTHON`, `TABLERSI_GPUS`, and `TABLERSI_PORT`
for another machine and ensure those GPUs/port are available. It does not stop
existing services. Download the checkpoint with `download_model.py` first.

## Historical study helpers

The remaining scripts document the earlier development and `study-v4`
experiment. They are not a general deployment installer or teaching pipeline.
Their server paths and GPU assignments are preserved as provenance, not assumed
to exist on a new machine. Some start services or manage owned model-server
processes; inspect and configure them before running.

- `start_model*.sh`, `wait_model*.sh`, `enable32_thinking_after_dev.sh`:
  start/wait for the original vLLM endpoints.
- `local_proxy.py`, `thinking_proxy.py`: loopback inference proxies; the thinking
  proxy sets a 512-token reasoning budget within each call's output allowance.
- `pilot.sh`, `dev*.sh`: preliminary and model-selection runs.
- `run_study.py`: original two-attempt L1/L2 protocol and terminal evaluation.
- `study_status.py`: inspect the recorded study state.
- `calibrate_judge.py`: three synthetic scoring checks, not broad judge validation.
- `model_manifest.py`: checkpoint file hashes and original runtime metadata.
- `release_unused_models.py`: legacy `study-v3` process cleanup; not part of the
  current workflow.
- `build_report.py`: render the original study report; needs the `report` extra
  and the matching recorded artifacts, which are not distributed in this repo.

The public evaluator no longer assumes a machine-specific key-file path. Set
`DEEPSEEK_API_KEY` outside the repository or supply `--key-file` when invoking the
CLI. Original frozen experiment archives remain the authority for exact replay.
