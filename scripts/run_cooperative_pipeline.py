"""Run one predeployment joint-policy iteration with restartable stage outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--local-url", default="http://127.0.0.1:31100/v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--development-summary", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_file = root / "pipeline_status.json"
    common = [
        "--data",
        str(root / "data"),
        "--protocol",
        str(root / "runtime/cooperative_v1/protocol.json"),
        "--repo",
        str(args.repo.resolve()),
        "--key-file",
        str(args.key_file.resolve()),
        "--local-url",
        args.local_url,
        "--workers",
        str(args.workers),
    ]
    stages = [
        (
            "teaching_episodes",
            ["eval", "--stage", "teaching", "--out", str(root / "teaching_episodes")],
        ),
        (
            "teacher_induction",
            [
                "teach",
                "--episodes",
                str(root / "teaching_episodes"),
                "--out",
                str(root / "teacher"),
            ],
        ),
        (
            "joint_search_and_audit",
            [
                "search",
                "--candidates",
                str(root / "teacher/candidates"),
                "--out",
                str(root / "adaptation"),
            ],
        ),
    ]
    environment = {**os.environ, "PYTHONPATH": str(root / "runtime/src")}
    status = {"pid": os.getpid(), "started_unix": time.time(), "stages": []}

    def save():
        temp = status_file.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_file)

    (root / "logs").mkdir(exist_ok=True)
    if args.development_summary:
        status["active_stage"] = "waiting_for_development"
        save()
        deadline = time.monotonic() + 1800
        while not args.development_summary.exists():
            if time.monotonic() >= deadline:
                status["state"] = "failed"
                status["reason"] = (
                    "Development summary did not arrive within 30 minutes"
                )
                save()
                raise SystemExit(1)
            time.sleep(10)
        development = json.loads(args.development_summary.read_text())
        if development["judge_failures"] or development["unknown_model_usage_calls"]:
            status["state"] = "failed"
            status["reason"] = (
                "Development has judge failures or unknown usage; inspect first"
            )
            save()
            raise SystemExit(1)
    for name, tail in stages:
        command = [
            sys.executable,
            "-u",
            "-m",
            "tablersi.cooperative_study",
            *tail,
            *common,
        ]
        status["active_stage"] = name
        save()
        with (root / "logs" / (name + ".log")).open("a") as log:
            result = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status["stages"].append(
            {
                "name": name,
                "returncode": result.returncode,
                "finished_unix": time.time(),
            }
        )
        if result.returncode:
            status["state"] = "failed"
            save()
            raise SystemExit(result.returncode)
        save()
    status["state"] = "complete"
    status["active_stage"] = None
    status["note"] = (
        "Predeployment only. Inspect release and audit before terminal evaluation."
    )
    save()


if __name__ == "__main__":
    main()
