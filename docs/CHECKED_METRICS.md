# Checked metrics and contrastive teaching (v7)

The v6 diagnosis found successful SQL followed by incorrect semantic revision:
cumulative-level growth was confused with single-period growth, and a missing
preceding cumulative value was replaced by the first displayed cumulative value.
v7 adds deterministic raw-cell calculations and reserves the final answer review
under the same seven-call ceiling. It is an experimental controller, not a
promoted release or a claim of improved accuracy.

## Calculation contract

QueryBuilder and EvidenceVerifier return both `queries` and `calculations`.
Each calculation specifies a requirement ID, metric name, unit, operation and
raw coordinates. The runtime generates a read-only query for those exact cells;
the model cannot supply values or executable Python. One calculation consumes
one database call, shared with SQL under the existing twelve-call ceiling.

Supported operations are `series`, `difference` (second minus first),
`relative_change` (second relative to first), `ratio` (first over second), `sum`
and `mean`. A series declares `level` or `cumulative`, increasing integer period
indices and, for cumulative values, accumulation cycles. It returns original
levels, adjacent observed-level changes and endpoint changes. For cumulative
series it additionally derives single-period values only from adjacent observed
periods in the same declared cycle, and growth between those derived values.

The first displayed cumulative value is never silently used as a single-period
value. Missing predecessors, gaps and resets leave dependent quantities null.
Observed-level changes across a gap remain valid as changes between those
observations; they must not be described as one-period growth. The endpoint
comparison covers the supplied series, whose extent must match the question.

Numeric parsing accepts ordinary decimal/scientific notation, properly grouped
thousands, percentages and parenthesized negatives. Unparseable text is unknown,
not zero. Percentage strings become fractions; output relative changes are also
fractions. Relative growth requires a positive base; ratios require a nonzero
denominator. Currency symbols and embedded unit words are not stripped. Larger
aggregations and operations outside this contract continue to use SQL.
Mixing explicit percentage strings and unmarked numbers is rejected; the result
declares its normalized value unit. Typed metadata and source coordinates remain
available to downstream roles, with explicit omission markers if prompt budgets
require compaction. Raw SQL strings in a request list are normalized to objects
without changing SQL text or bypassing the read-only executor.

These guarantees are about coordinates, parsing and arithmetic. Metric names,
units, entities, calendar indices, accumulation cycles and applicability remain
model proposals. The runtime does not prove that the selected cells answer the
question, that declared units agree, or that labels denote the claimed dates.
The bounded executor also does not infer an implicit zero before a cycle starts.

## Revision and call allocation

An unsupported opinion cannot withdraw a successful `checked_metric` record.
The verifier must cite separate, successful, active counterevidence to question
its binding or applicability. Counterevidence relevance still requires model
judgment; the guard is not a semantic correctness proof. All rejected requests
remain visible in working memory. Raw SQL evidence retains its existing repair
path. Failed metric requests can be withdrawn normally.

Both empty-memory and learned-memory v7 runs use the same sequence: routing
plan, refined plan, query construction, one pre-answer verification, composition,
final verification, and optional correction. New tools add no model calls.
This changes the controller jointly with the executor; a v6/v7 comparison alone
does not isolate their individual causal effects.

Verifier/composer projections preserve requested outputs but omit unverified
planning formulas and bindings. The original proposal remains in the trace.
Pre-answer composition and verification do not repeat earlier model checks as
factual evidence. Explicit runtime notes explain missing arithmetic operands.
This reduces an observed anchoring path; it does not enforce factual correctness
of every sentence or prove that an LLM respects all unknown-value constraints.

## Contrastive teaching

`cooperative_study teach --episodes BASE --contrast-episodes NEW --solver-version
v7 ...` pairs only the tasks in the frozen teaching partition. Identity, cluster,
question sequence and all coverage scores must match/validate before a pair is
sent to the teacher. Candidate and baseline trajectories include preserved
successes, repairs, damage and unresolved failures; neither side is assumed to
be superior. The teacher sees the fixed v7 tool contract and produces joint role
rules with observable applicability, exclusions, preservation conditions and
checks. No held-out labels enter teaching.

This is same-task contrastive teaching inspired by ExpeL, not a full reproduction
of ExpeL or GEPA. The existing three joint candidate operators remain; full
Pareto candidate evolution and merging are not implemented. Search scores select
candidates and never become the teacher's task references. The independent
audit and deployment partitions remain sealed until a finalist is committed.

## Validation status

Regression tests reproduce the diagnosed cumulative error through real raw-cell
execution and check gaps, cycle resets, missing values, strict numeric parsing,
operation direction, unsupported withdrawals, final-review budget, teaching
pair identity, and scorer-only labels. Operational probes are distinct from
scored evaluation. The first teaching probe did not call the new tool; its
traces are retained, and the output protocol was unified before any new scored
evaluation. A second probe exposed typed metadata lost by the old SQL-only
projection and string requests silently skipped by the executor; both paths
now have regression coverage. A third probe still copied an incorrect planning
formula despite a correct tool result; formula-free review/composition and
explicit missing-precondition notes address that path. The fourth probe no
longer imputed the missing second-quarter single-period value in the diagnosed
case, but still overstated the scope of a maximum comparison. It is not a fully
verified factual answer. The full suite has 113 passing tests. End-to-end gains
must be reported separately from these checks.
