# TableRSI

TableRSI is a research prototype for tabular reasoning with a frozen local LLM,
guarded read-only SQL search, and independently audited policy updates.

**Status:** the legacy solver, evaluation pipeline, bounded L1/L2 updates, and
predeployment/deployment data partitioning are implemented. An experimental
cooperative solver now adds role-projected memory, cloud-teacher policy/skill
induction, joint-bundle selection, and independent promotion audits. SFT/LoRA,
causal role attribution, and contextual L2 remain **planned**.
The recorded preliminary experiment did not demonstrate an RSI improvement:
all candidate policy promotions were rejected, leaving fixed, L1, and L2 with
identical final solver policies and memory. See
[implementation boundaries](IMPLEMENTATION_NOTES.md).

## Install

Python 3.11 or 3.12; the recorded experiments used Python 3.12. Model serving uses
a separate environment and an OpenAI-compatible local endpoint.

```bash
git clone https://github.com/T4t4KAU/table-rsi.git
cd table-rsi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional extras: `.[download]` for ModelScope downloads and `.[report]` for plots.
Model weights, datasets, credentials, logs, and run outputs are not in this repo.

## Ask a table question

Import a directory of CSV/XLS/XLSX files into a new inference-only store:

```bash
python scripts/import_tables.py --input-dir /path/to/tables --out data/my-table
tablersi-ask \
  --store data/my-table \
  --question 'What is the total revenue?' \
  --local-url http://127.0.0.1:31001/v1 \
  --local-model tablersi-qwen3-8b
```

The endpoint must already be running. `tablersi-ask` uses no reference answers
and no cloud scorer. An optional `--state` loads an audited policy/memory snapshot.

Download either checkpoint used in the study through ModelScope:

```bash
python -m pip install -e '.[download]'
python scripts/download_model.py --model Qwen/Qwen3-8B --out models/Qwen3-8B
# For Qwen3-32B, replace the model ID and output directory above.
```

## Prepare and evaluate the benchmark

The implementation uses [TableAgentBench](https://github.com/201983290498/TableAgentBench)
at commit `f70f134f63e4a7b0c4d15b2540953b5e95388586`.

```bash
git clone https://github.com/201983290498/TableAgentBench.git data/TableAgentBench
git -C data/TableAgentBench checkout f70f134f63e4a7b0c4d15b2540953b5e95388586
tablersi prepare --repo data/TableAgentBench --data data/prepared
```

`prepare` groups dialogs by table content. It writes solver inputs to
`public.json` and evaluator references to `oracle.json`. Evaluation requires a
local model endpoint and a cloud scorer credential, supplied via `--key-file`
or `DEEPSEEK_API_KEY`. The scorer receives questions, predictions, and reference
points; its reference labels are never passed to the solver.

```bash
tablersi eval \
  --repo data/TableAgentBench --data data/prepared \
  --split dev --limit 3 --variant fixed --out runs/dev-smoke \
  --local-url http://127.0.0.1:31001/v1 --local-model tablersi-qwen3-8b \
  --key-file /path/outside/repo/deepseek.key
```

Available variants are `direct`, `retry`, and `fixed`. `tablersi adapt --level 1`
or `--level 2` produces candidate policy states using the original proposal/audit
partitions; evaluation with `--state` runs a committed state. Adaptation does
not train model weights. The new bootstrap partition has different split names
and is not an input to this legacy adaptation loop.

The legacy primary metric is LLM-judged **reference-point coverage**. The new
cooperative study ranks candidates by the fraction of turns covering all points,
then by average point coverage. Neither is human-verified correctness.
All new experiments must keep final
evaluation labels out of training, memory updates, and candidate selection.

## Predeployment teaching and AgentMemory

The new partition keeps 42 table groups for teaching, 12 for predeployment
validation, and all 30 original test groups for deployment evaluation. Prior
terminal exposure is annotated rather than erased by restarting. References
for deployment are stored separately from agent inputs.

- [Partition commands and experiment protocol](docs/REPRODUCIBILITY.md)
- [AgentMemory design and implementation roadmap](docs/AGENT_MEMORY.md)
- [Current implementation and evidence limits](IMPLEMENTATION_NOTES.md)

The new [cooperative experiment](docs/COOPERATIVE_STUDY.md) generates abstract
policies and cooperative skills from designated training episodes. It keeps
model weights frozen; this is prompt/memory teaching, not weight distillation.
New experimental results must be reported separately from the legacy study.

## Repository layout

```text
src/tablersi/             Solver, data, SQL guards, state, scoring, and CLI
tests/                   Offline tests; no model, GPU, or API key required
scripts/                 Import, download, diagnostic, report, and study helpers
docs/                    Reproducibility notes and planned AgentMemory design
.github/workflows/       Python lint, formatting, tests, and package build
```

The [script index](scripts/README.md) distinguishes general tools from historical
server-specific orchestration. Historical scripts contain fixed paths, GPU
assignments, and ports and require adaptation before use on another machine.

## Development

```bash
ruff check src scripts tests
ruff format --check --output-format concise src scripts tests
python -m pytest tests -q
```

To apply formatting, run `ruff check --fix src scripts tests` followed by
`ruff format src scripts tests`. Python uses four-space indentation and an
88-column target; prompt/template literals can exceed it to preserve their exact
contents. CI checks Python 3.11 and 3.12.

The formatted development source differs from the archived source used in the
preliminary study. Original experiment hashes refer to those frozen archives.
Dataset and model licenses remain with their respective upstream projects.
