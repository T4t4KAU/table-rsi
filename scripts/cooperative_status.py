"""Inspect cooperative experiment progress without reading answers or references."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    status_path = args.root / "pipeline_status.json"
    result = {
        "pipeline": json.loads(status_path.read_text())
        if status_path.exists()
        else None,
        "evaluations": {},
    }
    for path in sorted(args.root.rglob("config.json")):
        config = json.loads(path.read_text())
        if not isinstance(config.get("ids"), list):
            continue
        summary_path = path.parent / "summary.json"
        result["evaluations"][str(path.parent.relative_to(args.root))] = {
            "expected_dialogs": len(config["ids"]),
            "completed_dialogs": sum(
                (path.parent / (i + ".json")).exists() for i in config["ids"]
            ),
            "complete": summary_path.exists(),
            "summary": json.loads(summary_path.read_text())
            if summary_path.exists()
            else None,
        }
    result["teacher_diagnoses"] = len(
        list((args.root / "teacher/diagnoses").glob("*.json"))
    )
    result["teacher_candidates"] = len(
        list((args.root / "teacher/candidates").glob("*.json"))
    )
    decision = args.root / "adaptation/decision.json"
    result["decision"] = json.loads(decision.read_text()) if decision.exists() else None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
