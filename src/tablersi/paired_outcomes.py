"""Report preservation and repair separately, using existing paired judge scores.

Inspired by https://aclanthology.org/2025.acl-long.203/ . Here the binary outcome
is full reference-point coverage, not independently verified correctness. A
between-policy comparison is descriptive, not a causal self-correction estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _index(episodes):
    indexed = {}
    for episode in episodes:
        for index, turn in enumerate(episode["turns"]):
            key = (episode["id"], index)
            if key in indexed:
                raise ValueError(f"Duplicate turn: {key}")
            value = turn.get("score", {}).get("coverage_ratio")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Missing or invalid coverage: {key}")
            indexed[key] = (
                episode["cluster"],
                turn["question"],
                value,
            )
    return indexed


def paired_outcomes(baseline, candidate):
    """Keep every paired turn; never silently drop failures or unmatched turns."""
    left, right = _index(baseline), _index(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Nonempty, exactly paired turn sets required")
    counts = {
        "full_to_full": 0,
        "full_to_not_full": 0,
        "not_full_to_full": 0,
        "not_full_to_not_full": 0,
    }
    changes, coverage_gain = [], 0.0
    for key in sorted(left):
        if left[key][:2] != right[key][:2]:
            raise ValueError(f"Question or cluster mismatch: {key}")
        old, new = left[key][2], right[key][2]
        counts[
            ("full" if old == 1 else "not_full")
            + "_to_"
            + ("full" if new == 1 else "not_full")
        ] += 1
        coverage_gain += new - old
        if (old == 1) != (new == 1):
            changes.append(
                {"id": key[0], "turn": key[1], "baseline": old, "candidate": new}
            )
    full = counts["full_to_full"] + counts["full_to_not_full"]
    nonfull = len(left) - full
    return {
        "turns": len(left),
        "counts": counts,
        "preservation_given_full": counts["full_to_full"] / full if full else None,
        "damage_given_full": counts["full_to_not_full"] / full if full else None,
        "repair_given_not_full": counts["not_full_to_full"] / nonfull
        if nonfull
        else None,
        "net_full_point_gain": (counts["not_full_to_full"] - counts["full_to_not_full"])
        / len(left),
        "mean_coverage_gain": coverage_gain / len(left),
        "full_point_changes": changes,
        "scope": "Paired model-judged reference coverage; not human factual accuracy, a causal effect, or a promotion decision.",
    }


def load_episodes(folder):
    return [json.loads(p.read_text()) for p in sorted(Path(folder).glob("tab*.json"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = paired_outcomes(
        load_episodes(args.baseline), load_episodes(args.candidate)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
