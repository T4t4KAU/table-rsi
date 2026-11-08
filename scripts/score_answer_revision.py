"""Score pre-review drafts to measure answer-review gains and regressions.

This is development-only analysis. It never changes solver records or supplies
scores to the solver. Identical draft/final answers share the existing judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tablersi.cli import write
from tablersi.evaluate import Judge
from tablersi.model import Client


def drafts(record):
    for index, turn in enumerate(record["turns"]):
        outputs = [
            step["response"]
            for step in turn["trace"]
            if step.get("role") == "AnswerComposer" and step.get("response")
        ]
        yield index, turn, outputs[0] if outputs else turn["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--judge-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted(args.run.glob("tab*.json"))]
    if sorted(r["id"] for r in records) != sorted(config["ids"]):
        raise ValueError("Wait for the complete run before revision analysis")
    if any(r["split"] != "predeploy_validation" for r in records):
        raise ValueError("Answer-revision diagnosis is restricted to development")
    provenance = {
        "run_hash": config["hash"],
        "oracle_hash": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "judge_model": args.judge_model,
        "scope": "paired draft/final development diagnosis; same completed solver trajectories",
    }
    if (args.out / "config.json").exists():
        if json.loads((args.out / "config.json").read_text()) != provenance:
            raise ValueError("Cannot mix incompatible revision analyses")
    write(args.out / "config.json", provenance)
    oracle = json.loads(args.oracle.read_text())
    judge = Judge(
        Client(args.judge_url, args.judge_model, args.key_file, local=False), args.repo
    )

    def one(record):
        dest = args.out / (record["id"] + ".json")
        if dest.exists():
            return json.loads(dest.read_text())
        turns = []
        for index, turn, draft in drafts(record):
            changed = draft != turn["answer"]
            before = (
                judge.score(
                    turn["question"], draft, oracle[record["id"]][index]["score_points"]
                )
                if changed
                else turn["score"]
            )
            turns.append(
                {
                    "question": turn["question"],
                    "draft": draft,
                    "changed": changed,
                    "before": before,
                    "after": turn["score"],
                }
            )
        result = {"id": record["id"], "cluster": record["cluster"], "turns": turns}
        write(dest, result)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, records))
    turns = [t for r in results for t in r["turns"]]
    before = [t["before"].get("coverage_ratio", 0) for t in turns]
    after = [t["after"].get("coverage_ratio", 0) for t in turns]
    summary = {
        "turns": len(turns),
        "changed": sum(t["changed"] for t in turns),
        "judge_failures": sum(
            "coverage_ratio" not in t[side]
            for t in turns
            for side in ("before", "after")
        ),
        "coverage_before": sum(before) / len(turns),
        "coverage_after": sum(after) / len(turns),
        "full_point_before": sum(x >= 1 for x in before) / len(turns),
        "full_point_after": sum(x >= 1 for x in after) / len(turns),
        "improved_turns": sum(b > a for a, b in zip(before, after)),
        "regressed_turns": sum(b < a for a, b in zip(before, after)),
        "note": "Model-judged scores; separate scoring calls can vary. Diagnostic, not an independent randomized trial.",
    }
    write(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
