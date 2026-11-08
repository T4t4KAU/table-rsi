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

The four generative roles share the frozen checkpoint declared in each protocol
(initially Qwen3-32B; subsequent development uses Qwen3.6). The executor is
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

The subsequent v3 engineering variant also permits evidence-linked corrections
to the requirement's formula, metric, entity, time scope, and other declared
semantic fields. It cannot change permissions or budgets. This addresses a
failure in which a verifier enforces an incorrect planning formula and rejects
a better query. The v4 variant applies literal, evidence-linked edits to answer
drafts rather than regenerating the entire answer. Edits must have unique,
nonoverlapping anchors and available evidence; untouched text is preserved.
These checks establish structural validity, not semantic correctness. Each
variant needs its own frozen run and matched comparison.

Completed Qwen3.6-35B development results are recorded in
`experiments/cooperative_v2/development_results.json`: v1/v2/v3/v4 all-point
coverage rates were 8/24, 7/24, 10/24, and 7/24 respectively. These are selected
development runs with empty learned memory, not deployment or RSI results.
No semantic `requirement_revision` event actually occurred in the v3 run;
its improvement cannot be attributed specifically to that implemented operation.
The trace summaries record observed edits separately from supported capabilities.

The same first planner prompt and nominal request seed also produced different
text in all 12 v3-v4 dialog pairs. Online serving was not text-deterministic in
these runs; its exact source has not been isolated. Cluster bootstrap intervals
do not account for this generation variability or development selection. Small
gains need repeated comparisons before mechanism attribution.

## Semantic memory routing experiment (v5)

A label-free check of the 26 search questions found a concrete retrieval failure:
the three instructional bundles retrieved skills on 0/10 English turns each;
one bundle retrieved none on any of the 26 turns. Its global guidance still
entered prompts, but its conditional skills did not. Retrieval keywords often
described internal method terminology rather than ordinary user requests.

The v5 runtime therefore gives the existing planner a bounded index of skill
applicability, exclusions, and planning hints. The planner proposes at most
three known skill IDs linked to its requirement IDs; downstream roles receive
the selected bundle's corresponding full role views. Provenance is excluded.
Malformed selections fall back to keyword retrieval; an explicit empty selection
is respected. No extra model call is added, although prompt tokens increase.
The index is abridged, and model-proposed applicability is not a formal guarantee.

With empty memory, v5 keeps the v3 prompts and control flow. It still receives a
fresh matched baseline for the feedback-teaching search. Initial v3 instructional
results and the v5 feedback/routing experiment remain distinct; their comparison
alone does not isolate feedback from routing. Only same-runtime empty-memory
versus taught-memory comparisons test the learned bundle's contribution. All
searches stop before a separately committed, single-finalist independent audit.

## Expanded teacher-group curriculum

`python -m tablersi.curriculum` uses all dialogs within the original 17
teacher-owned groups: 71 dialogs and 317 turns. Search and audit groups remain
excluded. This is a broader predeployment teaching configuration than the
original 34-episode pilot, and its effect must be measured separately.

The teacher sees the deployed role interfaces and produces three joint bundles.
Its source diagnoses can be reused only when exact input hashes and source IDs
match. Raw proposals are retained. An explicit compilation step removes
unavailable rubric-related retrieval keywords without altering action text;
instructions that depend on a hidden rubric are rejected. This complements,
but does not prove, abstraction or absence of memorization. Candidate policies
are not released until the same-runtime search and independent audit pass.

The curriculum can optionally consume `--episodes` from actual local student
runs. It checks dialog ownership, cluster identity, and question identity before
passing observable actions, answers, and feedback to the teacher. It never
accepts search/audit/deployment trajectories as teaching examples. With
`--bundle`, the teacher also sees the incumbent abstract rules and must produce
a complete replacement; parent version and input hashes retain the lineage.
Without student episodes this is instructional skill induction, not a measured
feedback-driven recursive improvement.

Remote reasoning is explicit, for example `--judge-model deepseek-v4-pro
--teacher-effort high --teacher-max-tokens 24000`. This configures only the
curriculum teacher. Evaluation continues to use its separately frozen scorer.
Malformed teaching JSON is retained and retried up to three times; completed
exact-input diagnoses can be reused after an interrupted run. Raw hidden
reasoning is not stored. Prior failed proposals remain part of the experiment
record rather than being silently replaced.

The inference entry point accepts the matching cooperative runtime and budget:

```bash
tablersi-ask --store data/my-table --question 'Compare the costs.' \
  --bundle runs/cooperative/adaptation/release.json \
  --solver-version v3 --thinking-budget 4096 \
  --local-url http://127.0.0.1:31105/v1 \
  --local-model tablersi-qwen36-35b-fp8
```

Use the exact version/model/budget approved in the release experiment. Selecting
an arbitrary combination is not equivalent to deploying that audited release.
