"""Frozen-protocol development selection, paired L1/L2 adaptation, terminal test."""

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/data1/sdnware/tablersi")
PY = str(ROOT / ".venv/bin/python")
OUT = ROOT / "runs/study-v4"
OUT.mkdir(parents=True, exist_ok=True)


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def status(stage, **kw):
    record = {"stage": stage, "time": time.time(), **kw}
    write(OUT / "status.json", record)
    print(json.dumps(record), flush=True)


def launch(args, log):
    f = open(log, "w")
    p = subprocess.Popen(
        [PY, "-m", "tablersi.cli", *args], stdout=f, stderr=subprocess.STDOUT
    )
    f.close()
    return p


def wait_file(path, limit_s=3600):
    end = time.time() + limit_s
    while time.time() < end:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(5)
    raise TimeoutError(str(path))


def main():
    status("waiting_for_development")
    reports = {
        m: wait_file(ROOT / f"runs/dev-v3-{m}/fixed/summary.json")
        for m in ("8b", "32b", "8b-thinking", "32b-thinking")
    }
    eligible = {
        m: r
        for m, r in reports.items()
        if r["judge_failures"] == 0 and r["unknown_model_usage_calls"] == 0
    }
    if not eligible:
        raise RuntimeError(
            "Neither development run is complete and fully scored; resolve infrastructure before freezing test"
        )
    selected = max(
        eligible,
        key=lambda m: (
            eligible[m]["turn_macro_coverage"],
            -eligible[m]["local_output_tokens"],
        ),
    )
    url = {
        "8b": "http://127.0.0.1:31001/v1",
        "32b": "http://127.0.0.1:31100/v1",
        "8b-thinking": "http://127.0.0.1:31101/v1",
        "32b-thinking": "http://127.0.0.1:31102/v1",
    }[selected]
    model = "tablersi-qwen3-" + selected
    common = [
        "--repo",
        str(ROOT / "data/TableAgentBench"),
        "--data",
        str(ROOT / "data/prepared"),
        "--local-url",
        url,
        "--local-model",
        model,
        "--workers",
        "4",
    ]
    write(
        OUT / "model_selection.json",
        {
            "criterion": "development turn-macro coverage; lower output tokens breaks ties",
            "selected": selected,
            "reports": reports,
            "test_consulted": False,
        },
    )
    subprocess.run(
        [
            "tar",
            "-czf",
            str(OUT / "frozen_source.tar.gz"),
            "--exclude=__pycache__",
            "--exclude=*.egg-info",
            "-C",
            str(ROOT / "src"),
            ".",
        ],
        check=True,
    )
    status("adapting", selected=selected)
    adapt = [
        "adapt",
        *common,
        "--attempts",
        "2",
        "--candidate-cap",
        "4",
        "--proposal-limit",
        "4",
        "--audit-limit",
        "6",
        "--replay-turn-limit",
        "1",
    ]
    l1 = launch([*adapt, "--level", "1", "--out", str(OUT / "L1")], OUT / "L1.log")
    decision = OUT / "L1/attempt0/decision.json"
    while not decision.exists():
        if l1.poll() is not None:
            raise RuntimeError("L1 stopped before first decision")
        time.sleep(5)
    # Initial policies, proposals, cases, random seeds and scoring observations are identical.
    # Reusing this common first attempt eliminates spurious differences due to judge noise.
    (OUT / "L2").mkdir(exist_ok=True)
    shutil.copytree(OUT / "L1/attempt0", OUT / "L2/attempt0", dirs_exist_ok=True)
    write(
        OUT / "common_first_attempt.json",
        {
            "source": "L1/attempt0",
            "reused_by": "L2/attempt0",
            "reason": "identical initial solver and operator policy; common random-number control",
            "logical_update_cost_charged_to_both_levels": True,
            "physical_api_cost_charged_once": True,
        },
    )
    l2 = launch([*adapt, "--level", "2", "--out", str(OUT / "L2")], OUT / "L2.log")
    if l1.wait() != 0 or l2.wait() != 0:
        raise RuntimeError("Adaptation process failed; test remains sealed")
    status("terminal_evaluation", selected=selected)
    states = {
        label: json.loads((OUT / label / "state.json").read_text())
        for label in ("L1", "L2")
    }
    default = {
        "policy": {
            "initial_queries": 2,
            "replan_rounds": 1,
            "local_first": True,
            "context_chars": 14000,
            "diversity": 0.15,
            "verify": True,
        },
        "memory": [],
    }

    def key(s):
        return json.dumps({k: s[k] for k in ("policy", "memory")}, sort_keys=True)

    source = {key(default): "fixed"}
    aliases = {"direct": "direct", "retry": "retry", "fixed": "fixed"}
    jobs = [
        ("direct", "direct", None),
        ("retry", "retry", None),
        ("fixed", "fixed", None),
    ]
    for label, state in states.items():
        if key(state) in source:
            aliases[label] = source[key(state)]
        else:
            source[key(state)] = label
            aliases[label] = label
            jobs.append((label, "fixed", OUT / label / "state.json"))
    write(
        OUT / "test_equivalence.json",
        {
            "aliases": aliases,
            "reason": "identical policy and memory imply identical frozen solver; meta-state is not a task-time input",
            "states": states,
        },
    )

    def run(job):
        label, variant, state = job
        args = [
            "eval",
            *common,
            "--split",
            "test",
            "--limit",
            "12",
            "--variant",
            variant,
            "--out",
            str(OUT / "test" / label),
        ]
        if state:
            args += ["--state", str(state)]
        p = launch(args, OUT / f"test-{label}.log")
        if p.wait() != 0:
            raise RuntimeError("test failed: " + label)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, jobs))
    summaries = {
        label: json.loads((OUT / "test" / origin / "summary.json").read_text())
        for label, origin in aliases.items()
    }
    write(
        OUT / "results.json",
        {
            "selected_model": model,
            "aliases": aliases,
            "summary": summaries,
            "preliminary": True,
            "dialogs_per_variant": 12,
            "full_benchmark": False,
        },
    )
    status("complete", selected=selected, summary=summaries)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("failed", error=repr(e))
        raise
