# Reproducibility and experiment protocols

A frozen-weight tabular reasoning prototype with guarded read-only SQL search,
versioned cross-task policy updates (L1), and audited improvement-operator
reweighting (L2/RSI). Local solvers: Qwen3-8B and Qwen3-32B downloaded from ModelScope.
The recorded primary study selects Qwen3-32B with a 512-token thinking cap
from development results; task-time weights remain frozen.

## Reproduce

Python 3.12. Install with `pip install -e ".[dev]"` and run `pytest tests`.
Keep API credentials outside this repository. The evaluator accepts `--key-file`.
The local OpenAI-compatible endpoint is supplied through `--local-url`.

## Python code style

Use four spaces for indentation and Ruff for formatting and import ordering.
The formatting target is 88 columns; long prompt and template literals may exceed
it to preserve their exact contents. Install the pinned tools with the `dev` extra.

```bash
ruff check --fix src scripts tests
ruff format src scripts tests
ruff check src scripts tests
ruff format --check --output-format concise src scripts tests
pytest tests
```

This development copy has been reformatted. The frozen source archives and code
hashes accompanying the recorded study refer to the original evaluated source.

## Experiment commands

```
python -m tablersi.cli prepare --repo /path/to/TableAgentBench --data /path/to/prepared
python -m tablersi.cli eval --repo /path/to/TableAgentBench --data /path/to/prepared --out /path/to/run --split dev --limit 3 --variant fixed
python -m tablersi.cli adapt --repo /path/to/TableAgentBench --data /path/to/prepared --out /path/to/l2 --level 2 --attempts 2 --candidate-cap 3 --proposal-limit 5 --audit-limit 6
python -m tablersi.cli eval --repo /path/to/TableAgentBench --data /path/to/prepared --out /path/to/test-l2 --split test --state /path/to/l2/state.json
```

Variants: `direct` (table context to answer), `retry` (one SQL candidate, execution-error retry),
`fixed` (complementary SQL queries and bounded probes/replanning). L1 and L2 are
`fixed` evaluated with their committed state snapshots.

## Data and evaluation

### Deployment split for the next study

Create an immutable partition from the existing prepared data and the previous
study protocol:

```bash
python -m tablersi.partition --source /path/to/prepared --previous-protocol experiments/study_v4/protocol.json --out /path/to/bootstrap-v1
```

The previous proposal/audit groups become `predeploy_train` (42 groups, 170
dialogs, 737 turns); previous development groups become `predeploy_validation`
(12 groups, 65 dialogs, 271 turns). All original test groups remain in
`postdeploy_test` (30 groups, 144 dialogs, 587 turns). The previous terminal
evaluation touched 12 of these groups. The manifest distinguishes these from
the remaining 18 groups (40 dialogs, 181 turns); prior results are not erased
by restarting. Report both the full deployment set and these exposure subsets.

Only training data may supply teacher demonstrations, SFT samples, and initial
skills. Validation selects predeployment configurations; final labels must not
enter teaching, prompts, memory, or online policy updates. Freeze model/adapter,
initial policy/memory, code, and the evaluation protocol before deployment.
This command partitions data; it does not perform teaching or training.

Deployment inputs contain only `public.json`, table stores, and a manifest.
References are written separately to `scorer_only/oracle.json`. For evaluation:

```bash
python -m tablersi.cli eval --repo /path/to/TableAgentBench --data /path/to/bootstrap-v1/postdeploy_test --oracle-file /path/to/bootstrap-v1/scorer_only/oracle.json --split postdeploy_test --variant fixed --out /path/to/new-run
```

`--oracle-file` belongs to the scorer, not `Agent.solve`. Physical file
separation is not an OS access-control boundary. Keep the scorer separate from
the deployed agent. All samples, including missing inputs, remain assigned;
training exclusions must be recorded rather than silently dropping test cases.
Cluster, store, and exact-sheet hashes are checked across partitions; this does
not prove that semantic templates never overlap. The old study is unchanged.

Pinned TableAgentBench commit: `f70f134f63e4a7b0c4d15b2540953b5e95388586`.
The published samples and validation files share table directories. We create
content-grouped development, proposal, two fresh audit, and test partitions;
this is a new experimental split, not the benchmark's original leaderboard.
`public.json` contains questions and source references. `oracle.json` is used
only by the evaluator. Predicted dialog history never contains reference answers.
Missing source files and missing questions remain in the evaluation denominator.

The benchmark's official point-coverage rubric is applied by DeepSeek Flash.
This is an LLM-judged coverage estimate, not exact-match accuracy or verified
semantic correctness. Numeric tolerance and qualitative coverage follow the
pinned official prompt. Scoring errors and missing token usage block promotion.
Development/audit scoring uses a cloud API; solver inference is local. This
experiment does not establish cloud-free autonomous adaptation.

## State and permissions

The local checkpoint is frozen. An attempt proposes bounded edits to query count,
replanning rounds, context size, local/global preference, or an abstract memory
rule. Candidate selection sees proposal tasks only. A selected finalist is
compared against the incumbent on a fresh paired audit batch. The current gate
uses a cluster bootstrap one-sided lower bound and a worst-cluster regression
check. This small-sample, per-attempt gate is approximate, not an anytime-valid
statistical guarantee. Only accepted candidates update solver policy/memory.
L2 additionally reweights the selected improvement operator from the completed
audit gain, including negative gain after rejection, for the next attempt.
L1 keeps this operator distribution fixed. Initial random proposals are shared.

SQL parses into a dependency representation and executes in a separate read-only
SQLite process with a timeout and row limit. Local repair may change one WHERE
literal, with signature, scope, numeric-type and AST-shape guards. Structural
changes require full replanning. E0 execution evidence cannot authorize success
labels. E2 reference-scored development evidence is required for state promotion.

## Current scope and limitations

This implementation executes complete SQL statements; it does not yet materialize
and reuse individual relational-DAG node results. Independent candidate queries
run concurrently. Local patches are restricted to literal changes, not arbitrary
functions. Cross-task memory currently selects predefined abstract checks;
it does not mine new verified executable repair skills. No teacher SFT/LoRA has
been performed. Those parts of the research plan remain future implementation.

Raw Excel rows, including internal blanks and multi-row headings, are preserved.
Cached formula values may be absent. Context retrieval is bounded and may omit
relevant cells. Guard acceptance and SQL success do not prove semantic correctness.
Model/judge versions, code and prompt hashes, data partitions, token ledgers,
queries, responses, and state decisions are recorded for each run. Actual token
costs vary even when all variants use the same upper resource limits.

## Server deployment used in this study

Runtime: vLLM 0.32.0, Transformers 5.19.0, Torch 2.13.0+cu130;
ModelScope 1.40.2 was used only for downloading. Model-serving environments
are separate from the lightweight evaluator environment. Scripts under
`scripts/` record this server's paths and ports; adjust them for another host.
All endpoints bind to 127.0.0.1. The optional thinking endpoints cap reasoning
at 512 tokens **within**, not in addition to, each call's existing maximum.
The primary study runner uses first-turn replay (`--replay-turn-limit 1`)
and full multi-turn terminal evaluation. See the JSON protocol for all
pre-evaluation amendments. For the report, install the `report` extra.

## Ask a new question without evaluation

```
python -m tablersi.ask --store /path/to/prepared/stores/GROUP_ID --question 'Your question' --local-url http://127.0.0.1:31102/v1 --local-model tablersi-qwen3-32b-thinking --state /path/to/committed/state.json
```

Omit `--state` for the fixed controller. Use the selected model/mode endpoint
recorded by the experiment. This command does not load `oracle.json`, reference
answers, the benchmark scorer, or the DeepSeek credential. `--output` optionally
saves the answer, query trace, and resource ledger. It accepts the question on
stdin when `--question` is omitted.

## Import your own tables

Place CSV/XLS/XLSX files in one directory, then create a new store:

```
python scripts/import_tables.py --input-dir /path/to/tables --out /path/to/new-store
```

Pass the resulting directory to `tablersi.ask --store`. This utility does not
load benchmark questions or labels. It refuses to overwrite an existing store.
Review `ingest_errors.json` if a file cannot be read. Formula cells use workbook
cached values, which can be missing; original cell coordinates are retained.
