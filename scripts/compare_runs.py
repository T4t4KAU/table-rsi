"""Cluster-resampled confidence intervals for completed, paired evaluation runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return {p.stem: json.loads(p.read_text()) for p in Path(path).glob("tab*.json")}


def statistic(records, indices, full_points=False):
    scores = [
        t["score"].get("coverage_ratio", 0.0)
        for i in indices
        for t in records[i]["turns"]
    ]
    return float(np.mean([s >= 1 for s in scores] if full_points else scores))


def compare(paths, replicates=4000, seed=20261108):
    runs = [read(p) for p in paths]
    ids = sorted(runs[0])
    assert all(sorted(r) == ids for r in runs), (
        "runs must contain exactly the same dialogs"
    )
    for run in runs[1:]:
        for i in ids:
            if run[i]["cluster"] != runs[0][i]["cluster"] or [
                t["question"] for t in run[i]["turns"]
            ] != [t["question"] for t in runs[0][i]["turns"]]:
                raise ValueError(
                    "Runs must contain identical paired questions and clusters"
                )
    groups = sorted({runs[0][i]["cluster"] for i in ids})
    by = {g: [i for i in ids if runs[0][i]["cluster"] == g] for g in groups}
    rng = np.random.default_rng(seed)
    samples = [
        [i for g in rng.choice(groups, len(groups), replace=True) for i in by[g]]
        for _ in range(replicates)
    ]
    base = np.array([statistic(runs[0], s) for s in samples])
    base_full = np.array([statistic(runs[0], s, True) for s in samples])
    result = []
    for path, run in zip(paths, runs):
        values = np.array([statistic(run, s) for s in samples])
        delta = values - base
        full = np.array([statistic(run, s, True) for s in samples])
        result.append(
            {
                "path": str(path),
                "dialogs": len(ids),
                "clusters": len(groups),
                "turn_macro_coverage": statistic(run, ids),
                "cluster_bootstrap_95_ci": np.quantile(values, [0.025, 0.975]).tolist(),
                "gain_vs_first": statistic(run, ids) - statistic(runs[0], ids),
                "paired_gain_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
                "fully_covered_turn_rate": statistic(run, ids, True),
                "full_point_bootstrap_95_ci": np.quantile(
                    full, [0.025, 0.975]
                ).tolist(),
                "full_point_gain_vs_first": statistic(run, ids, True)
                - statistic(runs[0], ids, True),
                "full_point_paired_gain_95_ci": np.quantile(
                    full - base_full, [0.025, 0.975]
                ).tolist(),
            }
        )
    return {
        "method": "paired cluster bootstrap percentile; descriptive, no multiple-comparison correction",
        "replicates": replicates,
        "seed": seed,
        "runs": result,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    Path(a.out).write_text(json.dumps(compare(a.paths), indent=2))
