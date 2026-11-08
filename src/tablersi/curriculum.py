"""Induce joint skills from all dialogs in the already designated teacher groups.

Only predeploy_train labels from teacher-owned clusters are accessible here.
Search/audit/deployment groups never enter the teacher prompt. This is a broader
predeployment curriculum, not evidence of an accepted or beneficial update.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cli import write
from .cooperative import CONTRACTS, RECONCILED_CONTRACTS, REPLANNED_CONTRACTS
from .cooperative_study import OPERATORS, read
from .memory import ROLES, Bundle, fingerprint, validate_teacher_abstraction
from .model import Client, json_object
from .observations import Observations

RUBRIC_TERMS = re.compile(
    r"参考点|评分点|评分标准|标准答案|真值|reference points|ground.?truth", re.I
)


def compile_teacher_keywords(proposed):
    """Remove unreachable rubric retrieval keys, preserving all action text."""
    result = copy.deepcopy(proposed)
    removed = []
    skills = result.get("skills", [])
    if not isinstance(skills, list):
        return result, removed
    for skill in skills:
        if not isinstance(skill, dict) or not isinstance(skill.get("keywords"), list):
            continue
        retained = []
        for keyword in skill["keywords"]:
            if isinstance(keyword, str) and RUBRIC_TERMS.search(keyword):
                removed.append({"skill_id": skill.get("id"), "keyword": keyword})
            else:
                retained.append(keyword)
        skill["keywords"] = retained
    return result, removed


def validate_deployment_rules(bundle):
    """Deployment receives evidence, never the teacher's reference rubric."""
    validate_teacher_abstraction(bundle)
    views = {
        "role_guidance": bundle.role_guidance,
        "skills": [
            {
                "keywords": s.keywords,
                "applicability": s.applicability,
                "exclusions": s.exclusions,
                "role_views": s.role_views,
            }
            for s in bundle.skills
        ],
    }
    match = RUBRIC_TERMS.search(json.dumps(views, ensure_ascii=False))
    if match:
        raise ValueError(
            "Deployment rules cannot depend on unavailable reference rubrics: "
            f"{match.group()!r}. Translate the rule to checking actual SQL "
            "evidence and user requirements; rephrase negated rubric vocabulary "
            "as an evidence boundary without assuming access to teaching labels."
        )
    return bundle


def teacher_groups(tasks, protocol):
    by_id = {t["id"]: t for t in tasks}
    owners = {by_id[i]["cluster"] for i in protocol["partitions"]["teaching"]}
    excluded = {
        by_id[i]["cluster"]
        for part in ("search", "audit")
        for i in protocol["partitions"][part]
    }
    if owners & excluded:
        raise ValueError("Teacher groups overlap search/audit groups")
    grouped = {cluster: [] for cluster in sorted(owners)}
    for task in tasks:
        if task["cluster"] in owners:
            if task["split"] != "predeploy_train":
                raise ValueError("Only predeployment training dialogs are eligible")
            grouped[task["cluster"]].append(task)
    return grouped


def teaching_episodes(path, groups):
    """Admit student traces only from the explicitly owned teaching dialogs."""
    if path is None:
        return {}
    tasks = {t["id"]: t for group in groups.values() for t in group}
    episodes = {}
    for file in sorted(path.glob("tab*.json")):
        episode = read(file)
        task = tasks.get(episode.get("id"))
        if task is None or episode.get("cluster") != task["cluster"]:
            raise ValueError("Student episode is outside teacher-owned groups")
        turns = episode.get("turns", [])
        if len(turns) > len(task["questions"]) or any(
            turn.get("question") != task["questions"][i] for i, turn in enumerate(turns)
        ):
            raise ValueError("Student episode questions do not match teaching data")
        if task["id"] in episodes:
            raise ValueError("Duplicate student teaching episode")
        episodes[task["id"]] = episode
    if not episodes:
        raise ValueError("No student teaching episodes found")
    return episodes


def episode_feedback(episodes, tasks):
    """Project observable actions and outcomes, without hidden model reasoning."""
    return [
        {
            "question": turn["question"],
            "student_answer": turn["answer"],
            "score_feedback": turn.get("score"),
            "working_memory": turn.get("working_memory"),
            "retrieved_skill_ids": turn.get("retrieved_skill_ids", []),
            "observable_actions": [
                {
                    "role": event.get("role"),
                    "response": event.get("response", "")[:5000],
                    "execution": event.get("execution"),
                    "events": event.get("events"),
                    "error": event.get("error", event.get("parse_error")),
                }
                for event in turn.get("trace", [])
            ],
        }
        for task in tasks
        for turn in episodes.get(task["id"], {}).get("turns", [])
    ]


def diagnose_with_retries(client, prompt, response_dir, max_tokens):
    """Persist failed generations and retry malformed or truncated teaching JSON."""
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(3):
        response = client.chat(
            messages, max_tokens=max_tokens, temperature=0, json_mode=True
        )
        write(response_dir / f"{attempt}.json", response)
        try:
            diagnosis = json_object(response["content"])
            if not isinstance(diagnosis, dict) or not isinstance(
                diagnosis.get("transferable_skills"), list
            ):
                raise ValueError("Expected an object with transferable_skills list")
            if response.get("finish_reason") == "length":
                raise ValueError("Teaching response reached its token limit")
            return diagnosis, response
        except (ValueError, TypeError) as exc:
            if attempt == 2:
                raise ValueError("Teaching JSON failed after three attempts") from exc
            # An empty reasoning-only completion is not a valid assistant turn.
            if response["content"]:
                messages.append({"role": "assistant", "content": response["content"]})
            messages.append(
                {
                    "role": "user",
                    "content": "请简洁完成所要求的JSON对象；无需详述分析过程。上次输出错误："
                    + str(exc),
                }
            )
    raise AssertionError("Unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--judge-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-model", default="deepseek-flash")
    parser.add_argument(
        "--teacher-effort", choices=["none", "low", "high", "max"], default="none"
    )
    parser.add_argument("--teacher-max-tokens", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reuse-diagnoses", type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol)
    if fingerprint(read(args.data / "manifest.json")) != protocol["manifest_hash"]:
        raise ValueError("Partition manifest changed")
    data = args.data / "predeploy_train"
    groups = teacher_groups(read(data / "public.json"), protocol)
    episodes = teaching_episodes(args.episodes, groups)
    oracle = read(data / "oracle.json")
    parent = Bundle.load(args.bundle) if args.bundle else Bundle()
    config = {
        "scope": "all dialogs and turns within original teacher-owned clusters",
        "protocol_hash": fingerprint(protocol),
        "manifest_hash": protocol["manifest_hash"],
        "teacher_model": args.judge_model,
        "teacher_effort": args.teacher_effort,
        "token_limits": {
            "diagnosis": args.teacher_max_tokens or 5000,
            "proposal": args.teacher_max_tokens or 10000,
        },
        "parent": parent.hash,
        "student_episode_hashes": {
            key: fingerprint(value) for key, value in episodes.items()
        },
        "teacher_code_hash": fingerprint(Path(__file__).read_text()),
        "reused_diagnosis_config": fingerprint(
            read(args.reuse_diagnoses / "config.json")
        )
        if args.reuse_diagnoses
        else None,
        "role_contract_hash": fingerprint(
            [CONTRACTS, RECONCILED_CONTRACTS, REPLANNED_CONTRACTS]
        ),
        "groups": {key: [t["id"] for t in value] for key, value in groups.items()},
        "dialogs": sum(len(value) for value in groups.values()),
        "turns": sum(len(t["questions"]) for value in groups.values() for t in value),
    }
    if (args.out / "config.json").exists() and read(args.out / "config.json") != config:
        raise ValueError("Cannot mix incompatible curricula")
    write(args.out / "config.json", config)
    client = Client(
        args.judge_url,
        args.judge_model,
        args.key_file,
        local=False,
        remote_effort=None if args.teacher_effort == "none" else args.teacher_effort,
        request_timeout=600 if args.teacher_effort != "none" else 180,
    )

    def diagnose(entry):
        cluster, tasks = entry
        dest = args.out / "diagnoses" / (fingerprint(cluster)[:20] + ".json")
        if dest.exists():
            return read(dest)
        examples = [
            {
                "question": question,
                "reference_points": oracle[t["id"]][i]["score_points"],
            }
            for t in tasks
            for i, question in enumerate(t["questions"])
            if question
        ]
        stores = sorted({t["group"] for t in tasks if t["supported"]})
        context = " ".join(example["question"] for example in examples)
        cells = [
            Observations(data / "stores" / group).render(
                context, max_chars=max(6000, 60000 // max(1, len(stores)))
            )
            for group in stores
        ]
        prompt = (
            "你是部署前的表格分析课程教师。以下所有样本均来自同一授权教学组。参考点可能有算术/口径错误或要求题目未明确说出的内容；必须对照原始数据，不把错误参考当事实。"
            "从本组实际业务任务中提炼二至五个具体、可迁移的分析技能。保留业务含义、输入字段的语义角色、数学公式与聚合顺序。不要把‘推导指标体系’‘核对口径’‘跨角色协作’这类泛泛原则作为技能名，也不要复述本提示的研究目标。"
            "每个技能必须给出一个用语义占位符表达的操作示例：什么输入条件触发、规划何种指标、如何计算、怎样核验、怎样解释。不能只提出禁止事项。可以是一个业务指标组合或一个结构性查询/纠错模式；允许多个角色一起修改才产生收益。"
            "抽象规则中不得出现具体任务、公司、品牌、年份、表号、列号、答案数值。数值或SQL示例必须用语义占位符。"
            '输出JSON：{"transferable_skills":[{"intent_family":"","trigger_concepts":[],"role_actions":{},"required_evidence":"","counterexamples":[],"completion_check":""}],"reference_uncertainties":[]}。\n原始观察：'
            + json.dumps(cells, ensure_ascii=False)
            + "\n授权教学题与参考："
            + json.dumps(examples, ensure_ascii=False)
        )
        feedback = episode_feedback(episodes, tasks)
        if feedback:
            prompt += (
                "\n本地学生在本教学组的实际运行记录："
                + json.dumps(feedback, ensure_ascii=False)
                + "\n结合学生的具体失败与成功，提出跨角色配套修订；区分可观察的错误传播与未经干预验证的因果猜测。保留正确行为，避免同类新错误。评分反馈也可能有误，须与原始证据核对。"
            )
        cached_path = (
            args.reuse_diagnoses / "diagnoses" / dest.name
            if args.reuse_diagnoses
            else None
        )
        if cached_path and cached_path.exists():
            cached = read(cached_path)
            if cached["input_hash"] != fingerprint(prompt) or cached["source_ids"] != [
                t["id"] for t in tasks
            ]:
                raise ValueError("Reused diagnosis has different inputs")
            write(dest, cached)
            return cached
        diagnosis, response = diagnose_with_retries(
            client,
            prompt,
            args.out / "diagnosis_attempts" / dest.stem,
            args.teacher_max_tokens or 5000,
        )
        result = {
            "cluster": cluster,
            "source_ids": [t["id"] for t in tasks],
            "input_hash": fingerprint(prompt),
            "diagnosis": diagnosis,
            "response": response,
        }
        write(dest, result)
        print(
            json.dumps({"diagnosed_cluster": cluster, "dialogs": len(tasks)}),
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        diagnoses = list(pool.map(diagnose, groups.items()))
    for operator, intent in OPERATORS.items():
        dest = args.out / "candidates" / (operator + ".json")
        if dest.exists():
            continue
        prompt = (
            "为冻结本地表格分析模型归纳一个跨角色协同更新包。更新方向："
            + intent
            + "。角色为"
            + json.dumps(ROLES)
            + "。将教学组经验压缩成固定角色契约之外的增量技能。不能复制固定契约或堆砌通用谨慎声明；不要把‘检查单位/核对字段/明确口径’包装为新的技能。优先保留来自具体教学业务的指标组合、语义公式、聚合顺序、反例及跨角色配套动作。"
            "规则可被新任务的原始证据推翻；检查和修订原计划本身，不能只强制下游服从。无需每个角色单独变好。"
            "输出JSON仅包含role_guidance和skills。role_guidance至少涉及两个角色，每角色至多二百中文字符，只写增量策略。skills至多十项，每项包含id、keywords、applicability、exclusions、role_views（至少两个已知角色）、provenance（列表，记录支撑该技能的diagnosis_index及概括理由）。"
            "keywords应使用用户会在任务中提到的业务/分析概念，避免只有内部技术词而无法检索。每个role_view至多八百字符。"
            "不存具体实体、文件、任务编号、列号、答案数值或年份。公式使用语义变量；不得调用云端或修改SQL权限。"
            "部署学生只有用户问题、原始表、SQL结果和工作记忆，永远没有参考点、评分标准、真值或标准答案。教师看到的参考仅用于归纳，输出规则中不可出现这些概念；将其转译成对原始证据与用户需求的复核。"
            "严格保持下面固定角色接口与JSON格式。可以补充策略，但不能改成不存在的角色或新的输出协议。累计水平的相对变化与单期环比是两种合法但不同的口径，不能一概禁止累计值之间比较；差分需要完整的前期累计值。"
            "每项技能必须有具体的业务或操作触发、条件动作和语义占位示例，避免重复泛泛的角色协作/字段核对原则。检索词应包含用户常用业务词，如费用、销售、风险、增长，而非只有内部术语。通用规则保持简短，特定方法写入适用条件明确的技能。"
            "业务解释可以基于可得数据提出明确标注假设的代理分析，不能冒充已观测事实或官方指标。\n教学经验："
            + json.dumps(
                [
                    {"diagnosis_index": i, "skills": d["diagnosis"]}
                    for i, d in enumerate(diagnoses)
                ],
                ensure_ascii=False,
            )
            + "\n固定部署角色接口："
            + json.dumps(
                {
                    role: CONTRACTS[role]
                    + RECONCILED_CONTRACTS[role]
                    + REPLANNED_CONTRACTS[role]
                    for role in ROLES
                },
                ensure_ascii=False,
            )
        )
        if parent.role_guidance or parent.skills:
            prompt += (
                "\n当前已冻结的父版本（只在有教学证据支持时修订，输出完整替代包）："
                + json.dumps(
                    {
                        "role_guidance": parent.role_guidance,
                        "skills": [
                            {k: v for k, v in skill.items() if k != "provenance"}
                            for skill in parent.dump()["skills"]
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            response = client.chat(
                messages,
                max_tokens=args.teacher_max_tokens or 10000,
                temperature=0,
                json_mode=True,
            )
            write(args.out / "proposals" / f"{operator}-{attempt}.json", response)
            try:
                proposed, removed = compile_teacher_keywords(
                    json_object(response["content"])
                )
                candidate = Bundle.from_dict(
                    {
                        "version": parent.version + 1,
                        "parent": parent.hash,
                        "role_guidance": proposed["role_guidance"],
                        "skills": proposed["skills"],
                        "status": "candidate",
                        "provenance": {
                            "interface_compilation": {
                                "removed_unavailable_retrieval_keywords": removed,
                                "action_text_changed": False,
                            },
                            "curriculum_hash": fingerprint(config),
                            "source_ids": [
                                t["id"] for tasks in groups.values() for t in tasks
                            ],
                            "teacher_model": args.judge_model,
                            "scope": "teacher-group-only curriculum; unverified until paired search and audit",
                        },
                    }
                )
                validate_deployment_rules(candidate)
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 2:
                    raise
                messages.extend(
                    [
                        {"role": "assistant", "content": response["content"]},
                        {
                            "role": "user",
                            "content": "修正格式或抽象性错误，保留可迁移规则："
                            + str(exc),
                        },
                    ]
                )
            else:
                write(dest, candidate.dump())
                print(
                    json.dumps({"candidate": operator, "hash": candidate.hash}),
                    flush=True,
                )
                break


if __name__ == "__main__":
    main()
