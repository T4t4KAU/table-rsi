# Recorded preliminary study

These are metadata and aggregate metrics from the completed `study-v4` run,
not a new experiment with the current formatted development source.

- `protocol.json`: partitions, model-selection rule, budgets, amendments, and
  evaluation boundaries. Also supplies prior terminal exposure to the new
  bootstrap partitioner.
- `results.csv`: aggregate scores and resource accounting. No raw benchmark
  answers, model traces, credentials, or checkpoints are included.

The primary experiment used 12 table groups and 47 turns. Reference-point
coverage was 55.12% for direct answering, 60.02% for SQL with error retries,
and 59.40% for the fixed TableRSI controller. L1 and L2 committed no policy or
memory updates; their terminal scores alias the fixed controller's predictions.
They are not independent repeated measurements. The data do not demonstrate
an RSI benefit or a statistically established improvement over the baselines.

The full experiment artifacts and frozen runtime source are maintained outside
this source repository. Code/prompt hashes in those archives identify the
evaluated implementation. Historical helpers are indexed in
[`scripts/README.md`](../../scripts/README.md).
