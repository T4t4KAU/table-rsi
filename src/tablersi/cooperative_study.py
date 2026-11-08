"""Predeployment teaching, joint-bundle search, and single-use independent audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .cli import evaluate, select, write
from .memory import ROLES, Bundle, fingerprint, validate_teacher_abstraction
from .model import Client, json_object
from .observations import Observations
from .state import audit_gate

OPERATORS = {
    "binding_evidence": "联合修改列绑定、查询构造和证据复核，解决语义正确但执行结果不支持结论的问题。",
    "complete_answer": "联合修改需求分解、证据覆盖和答案组织，提升整题答全能力；不牺牲数字正确性。",
    "cooperative_memory": "归纳跨角色协作技能，允许多个角色共同改变才产生收益；兼顾表头、缺失数据、推断和计算。",
}


def read(path):
    return json.loads(Path(path).read_text())


def first_per_cluster(tasks, split):
    count = len({t["cluster"] for t in tasks if t["split"] == split})
    return select(tasks, split, count)


def initialize(root, output):
    if output.exists():
        raise FileExistsError("Protocol already exists; do not overwrite")
    train = read(root / "predeploy_train/public.json")
    valid = read(root / "predeploy_validation/public.json")
    deployment = read(root / "postdeploy_test/public.json")
    remapped = [{**t, "split": t["original_split"]} for t in train]
    sets = {
        "teaching": first_per_cluster(remapped, "proposal"),
        "search": first_per_cluster(remapped, "audit0"),
        "audit": [t for t in remapped if t["split"] == "audit1"],
        "development": first_per_cluster(valid, "predeploy_validation"),
        "deployment": deployment,
    }
    owners = {}
    for name, tasks in sets.items():
        for t in tasks:
            if owners.setdefault(t["cluster"], name) != name:
                raise ValueError("Content cluster crosses study partitions")
    protocol = {
        "version": "cooperative-v1",
        "partitions": {name: [t["id"] for t in tasks] for name, tasks in sets.items()},
        "turn_limits": {
            "teaching": 2,
            "search": 2,
            "development": 2,
            "audit": 0,
            "deployment": 0,
        },
        "cluster_counts": {
            name: len({t["cluster"] for t in tasks}) for name, tasks in sets.items()
        },
        "manifest_hash": fingerprint(read(root / "manifest.json")),
        "primary_metric": "fully_covered_turn_rate (model-judged, not human correctness)",
        "secondary_metric": "turn_macro_coverage",
        "target": {"rate": 0.65, "metric_pending_user_choice": True},
        "candidate_operators": list(OPERATORS),
        "promotion": "One finalist; fresh paired audit. Full-point mean must improve; coverage one-sided 95% cluster bootstrap lower bound > 0; worst cluster coverage regression <= 0.2; no judge failures or unknown model usage.",
        "scope": "One predeployment L1 joint-bundle iteration; not an implemented L2 or online-learning claim.",
        "test_policy": "Freeze before full deployment evaluation. No tuning or promotion using deployment labels. Report prior-exposed and previously-unexposed groups separately.",
        "runtime": {"model": "tablersi-qwen3-32b", "thinking_budget": 1024, "seed": 42},
    }
    write(output, protocol)
    return protocol


def tasks_for(root, protocol, stage):
    name = (
        "postdeploy_test"
        if stage == "deployment"
        else "predeploy_validation"
        if stage == "development"
        else "predeploy_train"
    )
    data = root / name
    rows = {t["id"]: t for t in read(data / "public.json")}
    tasks = [copy.deepcopy(rows[i]) for i in protocol["partitions"][stage]]
    limit = protocol["turn_limits"][stage]
    if limit:
        for task in tasks:
            task["questions"] = task["questions"][:limit]
    return data, tasks


def evaluate_stage(args, stage, bundle, out):
    protocol = read(args.protocol)
    root = Path(args.data)
    if fingerprint(read(root / "manifest.json")) != protocol["manifest_hash"]:
        raise ValueError("Frozen partition manifest changed")
    data, tasks = tasks_for(root, protocol, stage)
    if args.thinking_budget != protocol["runtime"]["thinking_budget"]:
        raise ValueError("Thinking budget differs from frozen protocol")
    if (
        args.local_model != protocol["runtime"]["model"]
        or args.seed != protocol["runtime"]["seed"]
    ):
        raise ValueError("Model or seed differs from frozen protocol")
    if args.solver_version != protocol["runtime"].get("solver_version", "v1"):
        raise ValueError("Solver version differs from frozen protocol")
    oracle = (
        root / "scorer_only/oracle.json"
        if stage == "deployment"
        else data / "oracle.json"
    )
    config = SimpleNamespace(**{**vars(args), "data": data, "oracle_file": oracle})
    variant = (
        "cooperative"
        if args.solver_version == "v1"
        else "cooperative_" + args.solver_version
    )
    records, summary = evaluate(config, bundle, variant, tasks, out)
    if stage == "deployment":
        from .evaluate import summarize

        write(
            Path(out) / "exposure_subsets.json",
            {
                "previous_terminal_groups": summarize(
                    [r for r in records if r["previous_terminal_group"]]
                ),
                "not_in_previous_terminal_groups": summarize(
                    [r for r in records if not r["previous_terminal_group"]]
                ),
            },
        )
    return records, summary


def teach(args):
    """Only designated training episodes and their references enter the teacher."""
    out = Path(args.out)
    protocol = read(args.protocol)
    data, tasks = tasks_for(Path(args.data), protocol, "teaching")
    oracle = read(data / "oracle.json")
    client = Client(args.judge_url, args.judge_model, args.key_file, local=False)
    parent = Bundle.load(args.bundle) if args.bundle else Bundle()
    source = Path(args.episodes)
    config = {
        "protocol_hash": fingerprint(protocol),
        "parent": parent.hash,
        "teacher_model": args.judge_model,
        "episode_hashes": {
            t["id"]: hashlib.sha256(
                (source / (t["id"] + ".json")).read_bytes()
            ).hexdigest()
            for t in tasks
        },
    }
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("Cannot mix incompatible teaching runs")
    write(out / "config.json", config)

    def diagnose(task):
        dest = out / "diagnoses" / (task["id"] + ".json")
        if dest.exists():
            return read(dest)
        episode = read(source / (task["id"] + ".json"))
        if episode["id"] != task["id"] or episode["cluster"] != task["cluster"]:
            raise ValueError("Episode provenance mismatch")
        cells = ""
        if task["supported"]:
            cells = Observations(data / "stores" / task["group"]).render(
                " ".join(q or "" for q in task["questions"]), max_chars=22000
            )
        evidence = [
            {
                "question": t["question"],
                "answer": t["answer"],
                "reference": oracle[task["id"]][i]["score_points"],
                "score": {
                    k: v
                    for k, v in t["score"].items()
                    if k not in {"attempts", "usage"}
                },
                "trace": [
                    {
                        k: v
                        for k, v in step.items()
                        if k in {"role", "response", "execution", "error"}
                    }
                    for step in t["trace"]
                ],
            }
            for i, t in enumerate(episode["turns"])
        ]
        prompt = (
            "你是部署前的教学分析器。以下是授权训练题的参考评分点、学生轨迹和原始单元格。参考评分不是绝对可靠；分清数据不足、评分类噪声与学生实际错误。"
            "找出必须协同修改的角色及其输入输出契约，允许多角色共同修改后才出现收益。只归纳跨任务规则，不背题。"
            '输出JSON：{"failure_patterns":[],"joint_changes":[{"roles":[],"precondition":"","actions":{},"check":""}],"uncertainties":[]}。'
            "不要在归纳中包含具体公司、项目、任务编号或答案数值。\n原始观察："
            + cells
            + "\n训练经历："
            + json.dumps(evidence, ensure_ascii=False)
        )
        response = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0,
            json_mode=True,
        )
        diagnosis = json_object(response["content"])
        record = {
            "task": task["id"],
            "cluster": task["cluster"],
            "input_hash": fingerprint(prompt),
            "diagnosis": diagnosis,
            "response": response,
        }
        write(dest, record)
        print(json.dumps({"diagnosed": task["id"]}), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=min(args.workers, 6)) as pool:
        diagnoses = list(pool.map(diagnose, tasks))
    for operator, intent in OPERATORS.items():
        dest = out / "candidates" / (operator + ".json")
        if dest.exists():
            continue
        prompt = (
            "你正在优化冻结本地模型的协同表格推理策略。联合更新包是候选单位，不要求各个角色单独改善。固定角色和SQL安全边界不可修改。"
            "此次搜索方向："
            + intent
            + "\n已知角色："
            + json.dumps(ROLES)
            + "\n输出JSON仅包含role_guidance和skills。role_guidance是角色名到简短抽象规则的映射（至少两个角色，每角色不超过八百中文字符）。"
            "skills至多八项，每项包含id、keywords（少量一般业务/操作词用于检索）、applicability、exclusions、role_views（至少两个角色的配套动作）。"
            "不能存储训练题答案、具体实体/文件/表号/列号、具体数字或年份；用抽象占位语义表达。不能要求读取评分点或访问云端。"
            "规则须基于下面经验，可被新任务原始证据推翻；不得声称已经验证成功。\n当前策略："
            + json.dumps({"role_guidance": parent.role_guidance}, ensure_ascii=False)
            + "\n训练经验："
            + json.dumps([d["diagnosis"] for d in diagnoses], ensure_ascii=False)
        )
        for attempt in range(3):
            response = client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=6000,
                temperature=0.3,
                json_mode=True,
            )
            write(
                out / "generations" / f"{operator}-{attempt}.json",
                {"input_hash": fingerprint(prompt), "response": response},
            )
            try:
                value = json_object(response["content"])
                bundle = Bundle.from_dict(
                    {
                        "version": parent.version + 1,
                        "parent": parent.hash,
                        "status": "candidate",
                        "role_guidance": value["role_guidance"],
                        "skills": value.get("skills", []),
                        "provenance": {
                            "operator": operator,
                            "config_hash": fingerprint(config),
                            "teaching_clusters": sorted({t["cluster"] for t in tasks}),
                        },
                    }
                )
                if len(bundle.role_guidance) < 2:
                    raise ValueError("Joint proposal needs multiple roles")
                validate_teacher_abstraction(bundle)
                write(dest, bundle.dump())
                break
            except (KeyError, ValueError, TypeError) as exc:
                prompt += "\n上次结构校验失败，请重新生成完整JSON：" + str(exc)
        else:
            raise ValueError(f"No valid candidate for {operator}")
        print(json.dumps({"candidate": operator, "hash": bundle.hash}), flush=True)


def rank(summary):
    if summary["judge_failures"] or summary["unknown_model_usage_calls"]:
        return (-1, -1, 0)
    return (
        summary["fully_covered_turn_rate"],
        summary["turn_macro_coverage"],
        -summary["local_output_tokens"],
    )


def search_and_audit(args):
    out = Path(args.out)
    parent = Bundle.load(args.bundle) if args.bundle else Bundle()
    candidates = [
        (p.stem, Bundle.load(p)) for p in sorted(Path(args.candidates).glob("*.json"))
    ]
    if {name for name, _ in candidates} != set(OPERATORS):
        raise ValueError("Expected the three prespecified cooperative proposals")
    if any(b.parent != parent.hash for _, b in candidates):
        raise ValueError("Candidate is not derived from this incumbent")
    _, base_summary = evaluate_stage(args, "search", parent, out / "search/baseline")
    board = [{"name": "baseline", "summary": base_summary, "hash": parent.hash}]
    for name, bundle in candidates:
        _, summary = evaluate_stage(args, "search", bundle, out / "search" / name)
        board.append({"name": name, "summary": summary, "hash": bundle.hash})
        write(out / "search_board.json", board)
    best = max(board, key=lambda x: rank(x["summary"]))
    if best["name"] == "baseline":
        write(
            out / "decision.json",
            {
                "promotion": False,
                "reason": "No joint candidate improves search ranking",
            },
        )
        write(out / "release.json", parent.dump())
        return
    finalist = dict(candidates)[best["name"]]
    commitment = {
        "name": best["name"],
        "hash": finalist.hash,
        "parent": parent.hash,
        "protocol_hash": fingerprint(read(args.protocol)),
    }
    if (out / "audit_commitment.json").exists() and read(
        out / "audit_commitment.json"
    ) != commitment:
        raise ValueError("Audit already committed to a different finalist")
    write(out / "audit_commitment.json", commitment)
    incumbent, old = evaluate_stage(args, "audit", parent, out / "audit/baseline")
    candidate, new = evaluate_stage(args, "audit", finalist, out / "audit/candidate")
    if [r["id"] for r in incumbent] != [r["id"] for r in candidate]:
        raise ValueError("Unpaired audit records")
    current, proposed, clusters = [], [], []
    for left, right in zip(incumbent, candidate):
        if len(left["turns"]) != len(right["turns"]):
            raise ValueError("Unpaired audit turns")
        for a, b in zip(left["turns"], right["turns"]):
            current.append(a["score"].get("coverage_ratio", 0))
            proposed.append(b["score"].get("coverage_ratio", 0))
            clusters.append(left["cluster"])
    gate = audit_gate(current, proposed, clusters, min_clusters=12, seed=args.seed)
    full_gain = new["fully_covered_turn_rate"] - old["fully_covered_turn_rate"]
    clean = not any(
        s["judge_failures"] or s["unknown_model_usage_calls"] for s in (old, new)
    )
    accepted = bool(gate["accept"] and full_gain > 0 and clean)
    decision = {
        "promotion": accepted,
        "coverage_gate": gate,
        "full_point_gain": full_gain,
        "clean_usage_and_judging": clean,
        "finalist": commitment,
    }
    write(out / "decision.json", decision)
    release = copy.deepcopy(finalist if accepted else parent)
    if accepted:
        release.status = "promoted"
        release.provenance["audit_decision_hash"] = fingerprint(decision)
    write(out / "release.json", release.dump())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "eval", "teach", "search"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--stage",
        choices=["development", "teaching", "search", "audit", "deployment"],
        default="development",
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--local-url", default="http://127.0.0.1:31100/v1")
    parser.add_argument("--local-model", default="tablersi-qwen3-32b")
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument(
        "--solver-version", choices=["v1", "v2", "v3", "v4"], default="v1"
    )
    parser.add_argument("--judge-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-model", default="deepseek-flash")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "init":
        print(
            json.dumps(
                initialize(args.data, args.protocol), ensure_ascii=False, indent=2
            )
        )
    elif not args.out or not args.repo:
        parser.error("--out and --repo are required")
    elif args.command == "eval":
        evaluate_stage(
            args,
            args.stage,
            Bundle.load(args.bundle) if args.bundle else Bundle(),
            args.out,
        )
    elif args.command == "teach":
        if not args.episodes:
            parser.error("--episodes is required for teaching")
        teach(args)
    elif args.command == "search":
        if not args.candidates:
            parser.error("--candidates is required for search")
        search_and_audit(args)


if __name__ == "__main__":
    main()
