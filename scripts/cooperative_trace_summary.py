"""Summarize observed cooperative actions without claiming causal skill effects."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize_trace(path):
    config = json.loads((path / "config.json").read_text())
    records = [
        json.loads((path / (name + ".json")).read_text())
        for name in config["ids"]
        if (path / (name + ".json")).exists()
    ]
    turns = [turn for record in records for turn in record["turns"]]
    roles, retrieved, projected, revisions, patch_outcomes = (
        Counter() for _ in range(5)
    )
    errors = Counter()
    routing_modes = Counter()
    turns_with_memory = 0
    for turn in turns:
        ids = turn.get("retrieved_skill_ids", [])
        turns_with_memory += bool(ids)
        retrieved.update(ids)
        for event in turn.get("trace", []):
            role = event.get("role", "unknown")
            if "usage" in event:
                roles[role] += 1
                projected.update(
                    f"{role}:{ident}" for ident in event.get("memory_ids", [])
                )
            if "error" in event:
                errors[role + ":request"] += 1
            if "parse_error" in event:
                errors[role + ":parse"] += 1
            if role == "Executor":
                execution = event.get("execution", {})
                errors["sql_failure"] += not execution.get("ok", False)
            if role == "MemoryReconciler":
                revisions.update(item["action"] for item in event.get("events", []))
            if role == "AnswerPatch":
                outcome = (
                    "applied"
                    if event.get("applied")
                    else event.get("reason", "unchanged")
                )
                patch_outcomes[outcome] += 1
            if role == "MemoryRouter":
                routing_modes[event.get("mode", "unknown")] += 1
    return {
        "scope": "Observed events only. Retrieval/projection does not establish application, correctness, or causal benefit.",
        "complete": len(records) == len(config["ids"]),
        "dialogs": len(records),
        "turns": len(turns),
        "turns_with_retrieved_skills": turns_with_memory,
        "retrieved_turn_counts": dict(retrieved),
        "projected_role_call_counts": dict(projected),
        "role_calls": dict(roles),
        "working_memory_events": dict(revisions),
        "answer_patch_outcomes": dict(patch_outcomes),
        "memory_routing_modes": dict(routing_modes),
        "errors": dict(errors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize_trace(args.run), ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
