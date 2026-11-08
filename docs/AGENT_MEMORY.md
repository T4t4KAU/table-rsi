# AgentMemory design

**Status: partially implemented in the experimental cooperative runtime.**
Legacy `State.memory` still holds predefined checks. `memory.Bundle` now provides
immutable per-task snapshots, joint skills with role-specific views, keyword
retrieval, provenance excluded from solver prompts, and whole-bundle versioning.
The teacher can propose new abstract skills; search and independent audit gate
promotion. Candidate skills are not individually verified repair programs.
The richer evidence calibration, lifecycle, and contextual L2 described below
remain research targets; implementation is not evidence of measured improvement.

## State and evidence

| Layer | Contents | Lifetime and reader |
|---|---|---|
| Working memory | Current requirements, column bindings, plan, observations, unresolved questions | Task/dialog; role-specific projections |
| Experience records | Context features, attempted action, failure type, outcome scope, evidence provenance, cost | Local history; candidate skill induction |
| Verified skills | Applicability, exclusions, bounded action template, validators, supporting and contradicting episodes | Versioned cross-task store; solver retrieval |
| Policy and meta history | Context/action utility, uncertainty, cost, promotion decisions, rollback points | L1 controller and L2 improver |

An experience is `verified_success`, `verified_failure`, or `unknown` **for a
named property and scope**. E0 execution feedback establishes executability or
cost, not answer correctness. E1 validates only its stated property under its
preconditions. E2 known synthetic answers, authorized development references,
or confirmed corrections can support task-scoped semantic labels. A cloud
teacher's explanation alone is not a correctness oracle.

## Skill schema

A skill needs a stable ID/version, relational operator, failure type, typed
context features, applicability predicates, exclusions, bounded action template,
required evidence, validator IDs, provenance, independent supporting/contradicting
group IDs, cost statistics, and status (`candidate`, `active`, `retired`).
Store evidence grades separately; do not combine property checks and semantic
successes into one success-rate counter. Statistics should expose sample counts
and uncertainty, including uncertainty from repeated tasks in the same table group.

Example: an exact-year string-filter skill probes the current column's values
and proposes a single WHERE-literal normalization only after the column binding
and exact-year intent are established. Date intervals, uncertain bindings, and
structural changes exclude this skill. The patch still passes the fixed guard.
Column or join repairs must request replanning unless a separately implemented
and validated action permits the structural change.

Long-term skills use abstract slots, not stored benchmark answers, task IDs as
retrieval shortcuts, or unverified values copied from earlier model answers.
Identifiers may be retained in local provenance for auditing, not injected as
predictive features. Raw traces and reusable skills remain separate stores.

## Lifecycle

1. Record a task trace and the available scoped evidence.
2. Propose an abstract skill or refine an existing skill's applicability.
3. Check types, permissions, duplication, conflicts, and relevant counterexamples.
4. Compare against the committed system on independent trusted replay tasks.
5. Atomically commit a version only after sufficient evidence; otherwise retain
   a candidate, narrow its scope, or reject it. Keep rollback checkpoints.

Each task/search wave reads an immutable memory snapshot. Consolidation happens
after the wave so that parallel branches cannot treat one another's unverified
outputs as established knowledge. Skills cannot modify SQL permissions, guards,
reference access, or the frozen model/adapter.

## Retrieval and recursive improvement

Filter first by operator, types, failure context, evidence requirements, and
preconditions. Rank eligible skills by independently observed utility, cost, and
uncertainty. Pass a small, token-budgeted subset to the relevant role. Log retrieved,
applied, rejected, and successful skills separately to identify actual use.

L1 changes skill content, retrieval/ranking, routing, or search allocation.
L2 changes the contextual choice of bounded improvement operators such as
specializing a rule, merging compatible rules, retiring a harmful rule, or
reallocating search. Credit comes from later independent replay benefit and
cost, not skill count or a positive but highly uncertain mean score alone.

## Teaching and deployment

Cloud teaching uses only the training partition. Verified trajectories can
initialize role-conditioned SFT samples and candidate skills. A separate
validation partition selects the release. Freeze model/adapter hashes and the
initial policy/memory snapshot before evaluating deployment tasks.

Deployment references belong only to the scorer. They must not update memory,
retrieval, prompts, or model weights. Online updates require separate known-answer
synthetic tasks, applicable property checks, confirmed corrections, or reserved
trusted replay data. Without adequate evidence, preserve the committed policy.

## Experiments required

- Separate predeployment contributions: no teaching, SFT only, initial memory
  only, and SFT plus initial memory with otherwise aligned conditions.
- From the same released model and initial memory, compare frozen memory, L1,
  and L2 with matched compute limits and explicit realized costs.
- For online adaptation, answer and record a task before updating; assess later
  tasks in a predetermined stream. Never rescore earlier tasks as if the updated
  system had produced their original predictions.
- Report held-out performance, accepted updates, skill application rates,
  cross-group transfer, regressions, token costs, and uncertainty. Separate the
  full deployment set from groups exposed in earlier terminal experiments.

Implementation order: evidence/provenance records, typed skill store and retrieval,
independent replay/promotion, demonstrated L1 transfer, then contextual L2.
