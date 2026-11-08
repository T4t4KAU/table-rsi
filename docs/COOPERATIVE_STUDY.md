# Cooperative TableRSI experiment

This experiment tests whether a **joint role-policy and memory bundle** improves
a fixed local system. Each candidate may change several roles simultaneously.
Single-role improvements are not prerequisites. No new accuracy result is implied
by the presence of these modules.

## Implemented mechanism

1. `SemanticPlanner` identifies requirements, table choices, bindings, and missing evidence.
2. `QueryBuilder` receives focused original cells and the requirement contract.
3. The read-only executor records SQL, rows, failures, and evidence IDs.
4. `EvidenceVerifier` checks every requirement even if SQL executed successfully;
   it can request up to two rounds of supplementary or corrected queries.
5. `AnswerComposer` receives the contract, checks, raw observations, and evidence.

The four generative roles share frozen Qwen3-32B weights. The executor is
deterministic. A task retrieves at most three cooperative skills once and projects
their role-specific instructions throughout that task. Skill provenance remains
in the stored bundle and is not passed to the solver. A whole-bundle hash identifies
compatible role views. Retrieval does not establish that a skill was applied or
caused an improvement; trace records distinguish retrieval from observations.

Cell coordinates and blanks are preserved. Focused retrieval selects at most six
tables while retaining the full catalog. Evidence compaction removes rows from
complete records and marks omission; it never cuts the tail of a JSON string.
This is artifact provenance, not a complete causal or node-cached relational DAG.

## Prespecified data and comparisons

| Purpose | Table groups | Dialogs | Turns |
|---|---:|---:|---:|
| Engineering development | 12 | 12 | 24 |
| Teacher episodes | 17 | 17 | 34 |
| Candidate search | 13 | 13 | 26 |
| Independent promotion audit | 12 | 53 | 227 |
| Full postdeployment evaluation | 30 | 144 | 587 |

IDs and turn limits are fixed in `experiments/cooperative_v1/protocol.json` before
the new runs. Teacher, search, and audit groups subdivide the 42 predeployment
training groups. Development remains separate. The audit is independent of this
iteration's teacher and search examples; it is not a claim that no investigator
has ever seen these groups during earlier work. Prior terminal exposure remains
annotated for the deployment set.

The primary selection metric is the rate of turns covering **all** reference
scoring points; average scoring-point coverage is secondary. Both use the model
judge, not human correctness. Candidate ranking first uses full-point coverage,
then average coverage, then output tokens. Three cloud-generated proposals target
binding/evidence consistency, answer completeness, and cooperative memory.
Unsuccessful candidates remain recorded; no per-role positive-gain requirement
filters them out.

Only one finalist reaches audit. Promotion requires a positive full-point mean
gain, a positive one-sided 95% paired cluster-bootstrap lower bound for average
coverage, no cluster losing more than 0.20 average coverage, and no failed judge
calls or unknown local usage. This is an approximate single-attempt gate, not an
anytime guarantee. A rejected candidate does not update the released bundle.

The baseline and candidates use the same checkpoint, 1,024-token reasoning cap
inside a 4,096-token per-call limit, seven-call maximum, and fixed SQL/context
limits. Realized costs still differ and are reported. Improvements over the old
512-thinking-token solver cannot be attributed to RSI; only the matched baseline
versus candidate comparison tests the new update mechanism.

## Running

```bash
python -m tablersi.cooperative_study init \
  --data data/bootstrap-v1 --protocol runs/cooperative/protocol.json

python -m tablersi.cooperative_study eval \
  --data data/bootstrap-v1 --protocol runs/cooperative/protocol.json \
  --stage teaching --out runs/cooperative/teaching \
  --repo data/TableAgentBench --key-file /secure/path/deepseek.key

python -m tablersi.cooperative_study teach \
  --data data/bootstrap-v1 --protocol runs/cooperative/protocol.json \
  --episodes runs/cooperative/teaching --out runs/cooperative/teacher \
  --repo data/TableAgentBench --key-file /secure/path/deepseek.key

python -m tablersi.cooperative_study search \
  --data data/bootstrap-v1 --protocol runs/cooperative/protocol.json \
  --candidates runs/cooperative/teacher/candidates --out runs/cooperative/adaptation \
  --repo data/TableAgentBench --key-file /secure/path/deepseek.key
```

Use `--local-url` for a local endpoint that forwards `thinking_token_budget` and
`chat_template_kwargs` unchanged. The historical 512-token thinking proxy is not
appropriate for this experiment. The generic study CLI defaults to the existing
pass-through replica proxy, port 31100. Source hashes, questions, bundle, and model
settings prevent incompatible resume. Raw teaching records, prompts, and model
usage stay in private run directories; only abstract reviewed bundles may be
released publicly.

The pipeline helper expects an isolated root with `runtime/src`,
`runtime/cooperative_v1/protocol.json`, and `data/`. It runs teaching, induction,
search, and audit, writing a resumable status file. It deliberately stops before
terminal evaluation so the release, failures, and provenance can be inspected.
Deployment evaluation must freeze the selected release first and report both
prior-exposed and previously-unexposed groups. It must not feed deployment
references back into the teacher or select the best terminal attempt.

For inference without a teacher or scorer, load the released bundle:

```bash
tablersi-ask --store data/my-table --question 'Compare the costs.' \
  --bundle runs/cooperative/adaptation/release.json \
  --local-url http://127.0.0.1:31100/v1 --local-model tablersi-qwen3-32b
```

This entry point rejects candidate/rejected bundles. The released cooperative
default uses 1,024 reasoning tokens; a new deployment configuration should record
any explicit override.

## Remaining boundaries

- No weight training, SFT/LoRA, or cloud calls inside the deployed solver.
- No executed counterfactual role replay or causal responsibility attribution.
- No learned contextual L2 improver; the three proposal directions are fixed.
- Skill applicability is text guidance plus keyword retrieval, not a formal
  precondition prover. A number/identifier filter catches obvious memorization
  but does not prove that teacher text is free of benchmark-specific information.
- The scorer and teacher can use the same cloud model; independent examples
  reduce direct leakage but do not eliminate correlated judge preferences.

## Reconciled working memory experiment (v2)

`--solver-version v2` enables explicit evidence withdrawal and binding revisions,
dialog-level entity continuity, and a review of the generated answer. The frozen
protocol must declare the same solver version, model, seed, and thinking budget
as the evaluator command. The default remains v1 for existing studies.

Development failures motivated this change: a verifier could find the correct
entity while the answer still used an earlier query for a different entity.
`WorkingMemory` therefore records which existing evidence supports a revised
binding and removes explicitly contested evidence from subsequent role inputs.
The original SQL result and the correction remain in the trace. Corrections are
model proposals, not semantic ground truth. Previous task contracts help resolve
follow-up references but cannot establish factual values.

After producing an answer, v2 uses remaining calls to check its actual claims and,
when concrete corrections or supplementary queries exist, regenerate the answer.
The seven-call limit remains fixed. This can still introduce regressions; the
development-only `scripts/score_answer_revision.py` separately scores the initial
draft and final answer, reusing judgments when their text is identical.

For larger thinking budgets, the per-call generation cap is
`max(4096, 2 * thinking_budget)`. Thus a 4096-token thinking configuration has an
8192-token per-call cap. Matched variants must use the same budget. Engineering
changes, a stronger checkpoint, and learned joint bundles require distinct
comparisons; none alone establishes recursive improvement.
