from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import Agent
from .data import digest, prepare
from .evaluate import Judge, summarize
from .model import Client
from .state import (
    Evidence,
    Policy,
    State,
    admissible,
    audit_gate,
    commit_attempt,
    propose,
)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def clients(a):
    local = Client(
        a.local_url,
        a.local_model,
        local=True,
        thinking_budget=getattr(a, "thinking_budget", 0),
    )
    judge = Client(a.judge_url, a.judge_model, a.key_file, local=False)
    return local, Judge(judge, a.repo)


def select(data, split, limit=0):
    # Sample before seeing outcomes; one per cluster first, then remaining dialogs.
    tasks = [t for t in data if t["split"] == split]
    tasks.sort(key=lambda t: digest(["selection-v1", t["id"]]))
    seen = set()
    first = []
    rest = []
    for t in tasks:
        if t["cluster"] not in seen:
            first.append(t)
            seen.add(t["cluster"])
        else:
            rest.append(t)
    tasks = first + rest
    return tasks[:limit] if limit else tasks


def evaluate(a, state, variant, tasks, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Load scorer-only labels here; Agent.solve never receives them.
    oracle_path = Path(getattr(a, "oracle_file", None) or Path(a.data) / "oracle.json")
    oracle = json.loads(oracle_path.read_text())
    local, judge = clients(a)
    provenance = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    provenance["data_manifest"] = hashlib.sha256(
        (Path(a.data) / "manifest.json").read_bytes()
    ).hexdigest()
    provenance["scorer_oracle"] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    provenance["official_prompt"] = hashlib.sha256(
        (Path(a.repo) / "src/prompts/AgentEvalPrompt.py").read_bytes()
    ).hexdigest()
    config = {
        "variant": variant,
        "state": state.dump(),
        "ids": [t["id"] for t in tasks],
        "questions_hash": digest([(t["id"], t["questions"]) for t in tasks]),
        "model": a.local_model,
        "judge": a.judge_model,
        "seed": a.seed,
        "thinking_budget": getattr(a, "thinking_budget", 0),
        "provenance": provenance,
    }
    confighash = digest(config)
    if (out / "config.json").exists() and json.loads(
        (out / "config.json").read_text()
    ).get("hash") != confighash:
        raise ValueError("refusing to mix incompatible resumable runs")
    write(out / "config.json", {**config, "hash": confighash})

    def one(task):
        file = out / (task["id"] + ".json")
        if file.exists():
            return json.loads(file.read_text())
        if variant in {
            "cooperative",
            "cooperative_v2",
            "cooperative_v3",
            "cooperative_v4",
            "cooperative_v5",
            "cooperative_v6",
            "cooperative_v7",
        }:
            from .cooperative import CooperativeAgent

            agent = CooperativeAgent(
                local,
                state,
                reconciled=variant == "cooperative_v2",
                replanned=variant == "cooperative_v3",
                patched=variant == "cooperative_v4",
                routed=variant == "cooperative_v5",
                aligned=variant == "cooperative_v6",
                checked=variant == "cooperative_v7",
            )
        else:
            agent = Agent(local, state, variant=variant)
        history = []
        turns = []
        for i, q in enumerate(task["questions"]):
            result = agent.solve(
                Path(a.data) / "stores" / task["group"], q, history, seed=a.seed + i
            )
            # History contains only solver answers, never judgments or gold points.
            history.extend(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": result["answer"]},
                ]
            )
            points = oracle[task["id"]][i]["score_points"]
            result["gold_count"] = len(points)
            try:
                result["score"] = (
                    judge.score(q, result["answer"], points)
                    if q
                    else {
                        "total_metrics": len(points),
                        "covered_metrics": 0,
                        "coverage_ratio": 0.0,
                        "reasoning": "missing user question; counted as failed",
                    }
                )
            except Exception as ex:
                result["score"] = {"error": str(ex)[:500]}
            turns.append(result)
        d = {
            "id": task["id"],
            "cluster": task["cluster"],
            "split": task["split"],
            "group": task["group"],
            "previous_terminal_group": task.get("previous_terminal_group", False),
            "previous_terminal_dialog": task.get("previous_terminal_dialog", False),
            "turns": turns,
        }
        write(file, d)
        print(
            json.dumps(
                {
                    "completed": task["id"],
                    "variant": variant,
                    "turns": len(turns),
                    "coverage": sum(t["score"].get("coverage_ratio", 0) for t in turns)
                    / len(turns),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return d

    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(one, t) for t in tasks]):
            records.append(f.result())
    records.sort(key=lambda d: d["id"])
    report = summarize(records)
    write(out / "summary.json", report)
    print(
        json.dumps({"summary": report, "out": str(out)}, ensure_ascii=False), flush=True
    )
    return records, report


def adapt(a):
    data = json.loads((Path(a.data) / "public.json").read_text())
    state = State.load(a.state) if a.state else State()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proposal = select(data, "proposal", a.proposal_limit)
    if a.replay_turn_limit:
        proposal = [
            {**t, "questions": t["questions"][: a.replay_turn_limit]} for t in proposal
        ]
    for attempt in range(a.attempts):
        audit = select(data, f"audit{attempt}", a.audit_limit)
        if a.replay_turn_limit:
            audit = [
                {**t, "questions": t["questions"][: a.replay_turn_limit]} for t in audit
            ]
        if not audit:
            break
        inc_rec, inc_summary = evaluate(
            a, state, "fixed", proposal, out / f"attempt{attempt}/proposal-incumbent"
        )
        best = (
            inc_summary["turn_macro_coverage"],
            -inc_summary["local_input_tokens"],
            -inc_summary["local_output_tokens"],
            -inc_summary["database_calls"],
        )
        finalist = None
        candidates = propose(state, a.level, attempt, cap=a.candidate_cap, seed=a.seed)
        write(
            out / f"attempt{attempt}/proposal-order.json",
            {
                "level": a.level,
                "attempt": attempt,
                "meta_before": state.meta,
                "operators": [op for op, z in candidates],
            },
        )
        for op, z in candidates:
            rec, summary = evaluate(
                a, z, "fixed", proposal, out / f"attempt{attempt}/proposal-{op}"
            )
            rank = (
                summary["turn_macro_coverage"],
                -summary["local_input_tokens"],
                -summary["local_output_tokens"],
                -summary["database_calls"],
            )
            if rank > best:
                best = rank
                finalist = (op, z)
        if finalist is None:
            write(
                out / f"attempt{attempt}/decision.json",
                {"promotion": False, "reason": "no proposal candidate beats incumbent"},
            )
            continue
        op, z = finalist
        inc, _ = evaluate(
            a, state, "fixed", audit, out / f"attempt{attempt}/audit-incumbent"
        )
        cand, _ = evaluate(
            a, z, "fixed", audit, out / f"attempt{attempt}/audit-candidate"
        )
        if [r["id"] for r in inc] != [r["id"] for r in cand]:
            raise ValueError("unpaired audit")
        current = [
            sum(t["score"].get("coverage_ratio", 0) for t in r["turns"])
            / len(r["turns"])
            for r in inc
        ]
        candidate = [
            sum(t["score"].get("coverage_ratio", 0) for t in r["turns"])
            / len(r["turns"])
            for r in cand
        ]
        gate = audit_gate(
            current,
            candidate,
            [r["cluster"] for r in inc],
            seed=a.seed + attempt,
            min_clusters=4,
        )
        evidence = Evidence(
            "E2",
            "authorized_development",
            f"audit{attempt}",
            "reference-point coverage",
            gate.get("gain"),
        )
        gate["evidence_admitted"] = admissible(evidence, "success", f"audit{attempt}")
        if not gate["evidence_admitted"]:
            gate.update(accept=False, gain=None, reason="evidence authority rejected")
        # Missing judge outcomes and incomplete model usage prohibit promotion.
        if any(
            "coverage_ratio" not in t["score"] or t["budget"]["unknown_usage"]
            for r in inc + cand
            for t in r["turns"]
        ):
            gate.update(accept=False, reason="unknown scoring or usage", gain=None)
        state = commit_attempt(state, z, op, f"audit{attempt}", gate, a.level)
        write(
            out / f"attempt{attempt}/decision.json",
            {"operator": op, "gate": gate, "state": state.dump()},
        )
        state.save(out / "state.json")
    state.save(out / "state.json")
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "eval", "adapt"])
    p.add_argument("--repo", required=True)
    p.add_argument("--data", required=True)
    p.add_argument(
        "--oracle-file",
        help="Reference file for the evaluator only; never passed to the solver.",
    )
    p.add_argument("--out", default="runs/default")
    p.add_argument("--local-url", default="http://127.0.0.1:31001/v1")
    p.add_argument("--local-model", default="tablersi-qwen3-8b")
    p.add_argument("--judge-url", default="https://api.deepseek.com")
    p.add_argument("--judge-model", default="deepseek-flash")
    p.add_argument(
        "--key-file",
        help="Cloud scorer credential file; otherwise use DEEPSEEK_API_KEY.",
    )
    p.add_argument("--split", default="dev")
    p.add_argument("--variant", choices=["direct", "retry", "fixed"], default="fixed")
    p.add_argument("--state")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--level", type=int, choices=[1, 2], default=1)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--candidate-cap", type=int, default=3)
    p.add_argument("--proposal-limit", type=int, default=5)
    p.add_argument("--audit-limit", type=int, default=6)
    p.add_argument(
        "--replay-turn-limit",
        type=int,
        default=0,
        help="Prefix turns per replay dialog; zero uses all turns. Evaluation always uses complete selected dialogs.",
    )
    a = p.parse_args()
    if a.command == "prepare":
        report = prepare(Path(a.repo), Path(a.data))
        print(json.dumps({k: v for k, v in report.items() if k != "stores"}, indent=2))
        return
    if a.command == "adapt":
        adapt(a)
        return
    data = json.loads((Path(a.data) / "public.json").read_text())
    state = State.load(a.state) if a.state else State()
    if a.variant == "retry":
        state.policy = Policy(
            initial_queries=1, replan_rounds=1, local_first=False, verify=False
        )
    tasks = select(data, a.split, a.limit)
    evaluate(a, state, a.variant, tasks, a.out)


if __name__ == "__main__":
    main()
