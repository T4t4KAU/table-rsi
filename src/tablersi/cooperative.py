"""A frozen local solver with joint role policies and evidence-linked working memory."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .agent import SYSTEM
from .answer_patch import apply_answer_patch
from .memory import Bundle, fingerprint
from .model import Budget, json_object
from .observations import Observations, evidence_text
from .sql import execute, signature
from .working_memory import WorkingMemory

CONTRACTS = {
    "SemanticPlanner": """分析当前问题和前文指代，制定计划而不回答。表格可能有跨行表头；列绑定不确定时不要猜。选出相关表，分别列出问题要求的每个输出，包括比较、原因、建议等非数值需求。不把历史答案中的数字当成表格事实。返回JSON：{"tables":["t0"],"requirements":[{"id":"r1","need":"用户需求","kind":"lookup/calculation/comparison/explanation","binding":"原表列和表头行依据，未知则写unknown","formula":"计算公式或none","evidence_needed":"需要查询什么"}]}。""",
    "QueryBuilder": """依据需求契约和原始单元格生成至多四条互补查询。检查表头行、单位、合计与明细、年份列、分母；若计划绑定错误要纠正。不要把说明文字强制CAST成零；数字含逗号或百分号时先明确清洗与单位。SQL必须返回真正支持需求的记录、比较或计算；复杂计算分成可核查的简单查询。必要时先查询表头和相关原始行。原始列均为TEXT，排序时注意数值转换。对无法由数据计算的指标不发明定义。返回JSON：{"binding_checks":[],"queries":[{"needs":["r1"],"purpose":"目的和单位","sql":"SELECT ..."}]}。无需SQL的概念问题可返回空queries。""",
    "EvidenceVerifier": """独立核对每项需求与实际证据。SQL执行成功只说明可执行，不说明语义正确。检查列偏移、空值、范围内最大/最小、合计重复、比较方向、单位、统计口径、分母、缺失数据和前文指代。用当前表格纠正计划，不以之前模型的主张作为证据。所有新数值都应由查询得到。返回JSON：{"checks":[{"need":"r1","status":"supported/missing/conflict/unanswerable","evidence_ids":["e1"],"issue":"问题或none"}],"queries":[{"needs":["r1"],"purpose":"必要补充或纠错","sql":"SELECT ..."}]}。即使SQL无错误也要复核；至多四条新查询，证据足够则queries为空。""",
    "AnswerComposer": """逐项完整回答用户当前问题，不输出内部JSON。需求契约和复核意见只是模型建议，以原始单元格和实际查询结果为准。数值结论引用表/行或查询证据，交代单位和比较方向。区分事实、合理解释、建议和无法计算的内容。对原因、风险、建议等需求，给出与观察事实对应的有限推断，不因它不是SQL输出就完全遗漏。不把查不到当成零或不存在，不把总额与单项混淆，不把历史回答直接当事实。发现证据冲突时明确指出，不编造答案。""",
}

RECONCILED_CONTRACTS = {
    "SemanticPlanner": "追问的分析对象默认继承最近用户明确指定的对象，不能因表格里另一个指标也有同名字段就换对象。每条需求另列entity、metric、time_scope；若有多个可能对象则分别核查，不把历史答案的对象误当用户对象。趋势比较需规划起止值、绝对变化、相对变化及阶段特征；比率要明确口径和分母，不能用相似的比率代替。",
    "QueryBuilder": "查询必须保留所选指标/实体的原始文字列以及原始行号，不能只返回无身份信息的一串数字。明确在SQL中计算需求涉及的差额、变化率和比例，不将需要计算的数字留给答案角色心算。",
    "EvidenceVerifier": "若已发现证据来自错误对象或错误列，必须明确撤回该证据：在返回JSON中添加withdraw_evidence:[{id:证据ID,reason:原因}]。证据纠错后还须更新原来的列绑定：添加binding_corrections:[{need_id:需求ID,binding:完整正确绑定,evidence_ids:[已存在的新证据ID]}]。不能引用尚未执行的查询结果ID；补查后在下一轮再提交绑定修订。若输入有draft_answer，逐项检查答案中的每个数字、比较方向、对象指代、所有需求是否答全；对无证据的新计算补查。添加revise_answer布尔值：仅在发现具体错误/遗漏时为true，并在checks写出如何修复；没有证据不能为了显得不同而改写。",
    "AnswerComposer": "使用current_contract中的最新绑定；被明确撤回的证据不得继续作为答案依据。与实体不匹配的旧计划必须放弃。每个计算数值应来自未撤回的实际SQL结果，不自行增加未经计算的增长率或均值。趋势按同一时期给出绝对和相对变化、阶段特征，比例解释要与其分子分母一致。若修订草稿，只修复明确发现的问题，保留有证据的正确内容，不能用猜测补齐。",
}

REPLANNED_CONTRACTS = {
    "SemanticPlanner": "用户问题和原始表头优先于模型提出的契约。对区间趋势，先规划原始指标的起止变化及相邻变化：区分累计水平的增幅、单期数量环比、增长率的百分点变化，不能互相替代。累计序列的首个展示期不一定是累计起点；只有查到上一期累计值才可差分推导单期量。问题存在两种合理口径时分别标注并规划可计算的结果，不能擅自只保留一种。风险等指标未定义时，拆出表中可得的状态量和流量，给出透明的替代分析而不冒充官方指数；并列的单一指标不足以推出整体风险相同。",
    "QueryBuilder": "核查公式本身是否回答用户所问，不只核查能否执行。区间比较同时查询起止差额、起止相对变化和必要的阶段变化。累计指标的变化是累计水平变化；单期指标必须有相邻累计值才可差分。SQLite的log(x)是常用对数，ln(x)才是自然对数；CAGR可用pow(末值/初值, 1.0/期数)-1，零或负基数需另行处理。原始数值与清洗后数值并列返回，不能把格式异常静默转换为另一个量级。",
    "EvidenceVerifier": "需求契约的公式、指标口径也可能错；不能因查询不符合错误契约就撤回正确证据。重新对照用户问题和原始表头：必要时在JSON添加requirement_revisions:[{need_id:需求ID,updates:{formula:修正公式,metric:修正指标语义,evidence_needed:需要的补充证据},evidence_ids:[已成功执行的原始证据ID],reason:修订理由}]。updates可包含need、kind、binding、formula、evidence_needed、entity、metric、time_scope；只修改错误字段。对累计/单期、库存/流量、原始值/清洗值分别校验，不把第一个展示期的累计量当成单期量。区间首尾变化不可被相邻变化替代。缺少指标定义时允许明确标注假设的替代分析，不能虚构官方公式。",
    "AnswerComposer": "原始用户需求与表头含义优先于模型契约。明确区分累计水平增幅与单期环比，给出区间首尾比较并标明口径；若两种解释均合理，分别提供可由证据支持的数值。没有前期累计值，不把首个展示期冒充单期。风险比较应综合可得的状态和流量；可透明说明替代指标与假设，不能将其称为表中已定义的指数。名称或数字疑似识别错误时保留原文，并用其他身份列交叉核查可能的规范写法；没有依据不得擅自改名。",
}


class CooperativeAgent:
    def __init__(
        self,
        client,
        bundle: Bundle,
        max_calls=7,
        reconciled=False,
        replanned=False,
        patched=False,
    ):
        self.client = client
        self.bundle = copy.deepcopy(bundle).validate()
        self.max_calls = max_calls
        self.reconciled = reconciled or replanned or patched
        self.replanned = replanned or patched
        self.patched = patched
        self.dialog_contract = None

    def solve(self, store, question, history=None, seed=42):
        started = time.monotonic()
        generation_limit = max(4096, 2 * getattr(self.client, "thinking_budget", 0))
        budget = Budget(self.max_calls, 400000, self.max_calls * generation_limit, 12)
        trace, evidence = [], []
        selected = []
        signatures = set()
        working_memory = None
        if not history:
            self.dialog_contract = None

        def finish(answer, status):
            if working_memory:
                self.dialog_contract = copy.deepcopy(working_memory.contract)
            return {
                "question": question,
                "answer": answer,
                "trace": trace,
                "budget": budget.to_dict(),
                "elapsed_s": time.monotonic() - started,
                "state_version": self.bundle.version,
                "bundle_hash": self.bundle.hash,
                "variant": "cooperative_v4"
                if self.patched
                else "cooperative_v3"
                if self.replanned
                else "cooperative_v2"
                if self.reconciled
                else "cooperative",
                "working_memory": working_memory.project() if working_memory else None,
                "retrieved_skill_ids": [s.id for s in selected],
                "evidence_status": status,
                "successful_queries": sum(e.get("ok", False) for e in evidence),
                "failed_queries": sum(not e.get("ok", False) for e in evidence),
            }

        if not question:
            return finish("数据集该轮缺失用户问题，无法回答。", "malformed input")
        if not (Path(store) / "catalog.json").exists():
            return finish(
                "发布的数据集中缺少该任务的表格文件，无法据此计算。",
                "missing source tables",
            )
        observations = Observations(store)
        history = [
            {"role": m["role"], "content": str(m.get("content") or "")[:2200]}
            for m in (history or [])[-6:]
        ]
        query_context = (
            question
            + " "
            + " ".join(m["content"] for m in history if m["role"] == "user")
        )
        selected = self.bundle.retrieve(
            query_context + " " + observations.catalog_text()
        )
        working = {"question": question, "history": history}
        if self.reconciled and self.dialog_contract:
            working["previous_task_contract"] = self.dialog_contract
            working["history_contract_status"] = (
                "Previous model proposal; use for entity continuity, not as factual proof. Current explicit user intent overrides it."
            )

        def call(role, view, cells, final=False, patch_mode=False):
            memory = self.bundle.project(role, selected)
            prompt_evidence = (
                working_memory.active(evidence) if working_memory else evidence
            )
            if working_memory:
                view = {**view, **working_memory.project()}
            prompt = (
                "本轮输入：\n"
                + json.dumps(working, ensure_ascii=False)
                + "\n原始表格观察：\n"
                + cells
                + "\n上游工作产物（均待验证）：\n"
                + json.dumps(view, ensure_ascii=False)
                + "\n执行证据：\n"
                + evidence_text(prompt_evidence, 14000)
                + "\n可适用的跨任务协同规则：\n"
                + json.dumps(memory, ensure_ascii=False)
                + "\n当前角色固定契约：\n"
                + CONTRACTS[role]
            )
            if self.reconciled:
                prompt += "\n协作交接补充契约：\n" + RECONCILED_CONTRACTS[role]
            if self.replanned:
                prompt += "\n可修订语义契约：\n" + REPLANNED_CONTRACTS[role]
            if patch_mode:
                prompt += (
                    "\n本次是内部答案编辑模式，以下JSON协议替代普通自然语言输出。"
                    "只修改复核指出的具体错误、补充有证据的遗漏；保留原文中其他正确事实、分析和结论。"
                    "不要重新生成整篇答案或仅输出SQL。每个替换的old必须逐字匹配草稿中唯一的一段，且各段不能重叠；new是替换后的文字。"
                    "每项编辑附reason和evidence_ids，只能使用当前未撤回的成功证据ID，或确由原始观察支持时使用raw_cells。"
                    '输出JSON：{"edits":[{"old":"原文唯一片段","new":"修正片段","reason":"具体问题","evidence_ids":["e3"]}],'
                    '"additions":[{"text":"补充的遗漏分析","reason":"遗漏需求","evidence_ids":["e4"]}]}。'
                    "没有必要修改时两列表均为空；最多十六项。"
                )
            max_tokens = generation_limit
            if not budget.allowed(SYSTEM + prompt, max_tokens):
                trace.append({"role": role, "skipped": "hard model budget"})
                return None
            budget.used_calls += 1
            record = {
                "role": role,
                "input_hash": fingerprint(prompt),
                "bundle_hash": self.bundle.hash,
                "memory_ids": [s["id"] for s in memory["skills"]],
                "reads_evidence": [e["id"] for e in prompt_evidence],
            }
            try:
                response = self.client.chat(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    seed=seed + budget.used_calls,
                    temperature=0.1,
                    json_mode=not final,
                )
            except Exception as exc:
                budget.unknown_usage += 1
                budget.used_input += len((SYSTEM + prompt).encode())
                budget.used_output += max_tokens
                trace.append({**record, "error": str(exc)[:500]})
                return None
            if response.get("usage"):
                budget.charge(response["usage"])
            else:
                budget.unknown_usage += 1
                budget.used_input += len((SYSTEM + prompt).encode())
                budget.used_output += max_tokens
            trace.append({**record, **response, "response": response["content"]})
            try:
                result = (
                    response["content"] if final else json_object(response["content"])
                )
                if not final and not isinstance(result, dict):
                    raise ValueError("Expected object")
                return result
            except (ValueError, TypeError) as exc:
                trace.append({"role": role, "parse_error": str(exc)})
                return None

        def query(items, author):
            if not isinstance(items, list):
                return 0
            count = 0
            for item in items[:4]:
                if not isinstance(item, dict) or not isinstance(item.get("sql"), str):
                    continue
                sql = item["sql"]
                if (
                    not sql.strip()
                    or len(sql) > 6000
                    or budget.used_db >= budget.db_calls
                ):
                    continue
                try:
                    sig = signature(sql)
                except Exception:
                    sig = sql
                if sig in signatures:
                    continue
                signatures.add(sig)
                budget.used_db += 1
                result = execute(Path(store) / "tables.sqlite", sql, row_limit=60)
                rec = {
                    "id": f"e{len(evidence) + 1}",
                    "sql": sql,
                    "purpose": str(item.get("purpose", ""))[:500],
                    "needs": item.get("needs", []),
                    **result,
                }
                evidence.append(rec)
                trace.append({"role": "Executor", "author": author, "execution": rec})
                count += 1
            return count

        overview = observations.render(query_context, max_chars=18000)
        contract = call("SemanticPlanner", {}, overview) or {}
        if self.reconciled:
            working_memory = WorkingMemory(
                contract, allow_requirement_revisions=self.replanned
            )
        tables = contract.get("tables", [])
        if not isinstance(tables, list):
            tables = []
        tables = [t for t in tables if isinstance(t, str)]
        cells = observations.render(query_context, tables, max_chars=26000)
        plan = call("QueryBuilder", {"contract": contract}, cells) or {}
        query(plan.get("queries", []), "QueryBuilder")
        checks = []
        for _ in range(2):
            review = (
                call(
                    "EvidenceVerifier",
                    {
                        "contract": working_memory.contract
                        if working_memory
                        else contract,
                        "checks": checks,
                    },
                    cells,
                )
                or {}
            )
            checks = review.get("checks", [])
            if working_memory:
                events = working_memory.apply(review, evidence)
                trace.append({"role": "MemoryReconciler", "events": events})
            if not query(review.get("queries", []), "EvidenceVerifier"):
                break
        answer = call(
            "AnswerComposer",
            {
                "contract": working_memory.contract if working_memory else contract,
                "checks": checks,
            },
            cells,
            final=True,
        )
        if self.reconciled and answer and budget.used_calls <= budget.calls - 2:
            review = (
                call(
                    "EvidenceVerifier",
                    {"draft_answer": answer, "checks": checks},
                    cells,
                )
                or {}
            )
            events = working_memory.apply(review, evidence)
            trace.append(
                {"role": "MemoryReconciler", "events": events, "stage": "answer_review"}
            )
            additional = query(review.get("queries", []), "AnswerReview")
            if review.get("revise_answer") is True or events or additional:
                corrected = call(
                    "AnswerComposer",
                    {
                        "draft_answer": answer,
                        "checks": review.get("checks", []),
                        "instruction": "修复上述明确问题；新证据也必须核实对象和单位。",
                    },
                    cells,
                    final=not self.patched,
                    patch_mode=self.patched,
                )
                if self.patched and corrected:
                    answer, patch_result = apply_answer_patch(
                        answer,
                        corrected,
                        [
                            e["id"]
                            for e in working_memory.active(evidence)
                            if e.get("ok")
                        ],
                    )
                    trace.append({"role": "AnswerPatch", **patch_result})
                elif corrected:
                    answer = corrected
        return finish(
            answer or "本轮未能生成完整且有依据的答案。",
            "E0 execution; semantic review is model judgment"
            if any(e.get("ok") for e in evidence)
            else "table observations only",
        )
