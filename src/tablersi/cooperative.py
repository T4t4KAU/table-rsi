"""A frozen local solver with joint role policies and evidence-linked working memory."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .agent import SYSTEM
from .answer_patch import apply_answer_patch
from .memory import Bundle, fingerprint
from .metrics import compile_metric, evaluate_metric
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

METRIC_PROTOCOL = """确定性计算工具协议（替代旧的仅SQL计算要求）：返回JSON必须包含queries和calculations两个列表，可为空。calculations最多四项。每项格式：
{"need_id":"r1","metric":"指标名称与口径","unit":"原始单位","operation":"series","value_kind":"cumulative","points":[{"label":"原始时期名称","period":1,"cycle":"同一累计年度或周期","cell":{"table":"t0","row":2,"column":"c2"}},{"label":"下一时期","period":2,"cycle":"同一累计年度或周期","cell":{"table":"t0","row":2,"column":"c3"}}]}。
cell只能引用原始单元格坐标，不能填写推算值或SQL结果。按实际时期升序排列；period是等间隔时期序号，缺期时保留序号间隔；cycle区分累计重置周期。series的value_kind可为level（普通水平序列）或cumulative（累计序列）。工具同时返回水平差额/增幅、区间首尾变化；累计序列另外返回相邻累计差分的单期值及其环比。缺失前期值时首期单期必为null，绝不填零或首期累计量。相对变化输出小数，百分比展示乘100。
其他operation：difference（第二点减第一点）、relative_change（第二点相对第一点的变化，基期须为正）、ratio（第一点除第二点，分母非零）、sum、mean。它们也使用points中的原始cell，前三级要求恰好两点。单位不可混用；百分号原文自动转为小数，无法严格解析的原文保留为未知而不是零。一个计算只能覆盖同一实体、指标与统一单位；这些选择仍须依据用户需求和原始表头核查。
原始坐标已明确且每个指标不超过二十四个单元格时，累计和趋势计算必须用calculations，不另写SQL差分或增长公式。SQL用于定位原始行、查表头、较大分组聚合或工具不支持的关系查询。不要为同一指标再生成一个带不同口径的SQL计算。需要先定位时本轮查询，下一角色再提交calculations。若需要数值计算但calculations为空，在calculation_policy说明为何工具不适用。不要为调用工具省略用户要求的解释、比较或建议。"""

CHECKED_CONTRACTS = {
    "SemanticPlanner": "明确每项指标是普通水平、累计量还是比率；不把其他口径替换用户需求。后续工具可从原始单元格计算水平变化与累计差分，但不能证明你选对了指标或时期。",
    "QueryBuilder": METRIC_PROTOCOL,
    "EvidenceVerifier": METRIC_PROTOCOL
    + "\nchecked_metric证据的算术由程序执行；重点核查原始单元格绑定、单位、时期及适用性。不能仅因与自然语言规则表述不同而撤回。若要撤回checked_metric，在withdraw_evidence条目增加counterevidence_ids，引用其他已成功执行的证据，说明具体绑定或口径冲突；不能引用尚未执行的结果。不要重复已经完成的检查。若输入有draft_answer，逐项查其需求覆盖与数字；只有明确问题才要求改写。",
    "AnswerComposer": "checked_metric提供有来源的确定性算术，选择指标、单位和时间语义仍须核查。level_relative_change是原始水平的增幅；single_period_relative_change才是差分单期量的环比，二者不能互相替代。null表示该公式前提未满足，不可用别的列或心算补零。百分比显示时把小数乘100。保留原始单元格与口径，回答全部需求；受质疑的绑定要说明，不能因模型重复质疑就推翻已核实的算术。",
}

CHECKED_OUTPUTS = {
    "QueryBuilder": "依据用户需求和原始单元格生成结构化工具请求。逐项核查表头、列绑定、单位、分母、时间范围、明细与合计。只输出JSON，包含binding_checks、queries、calculations；queries每项包含needs、purpose、sql，calculations按下方确定性计算工具协议填写。每类最多四项；没有必要则为空。不要输出未经工具执行的计算答案。",
    "EvidenceVerifier": '核对每项用户需求与实际原始证据，尤其检查对象、时期、指标口径、单位、缺失前提。模型计划和先前复核不是真值。只输出JSON：checks列表（每项need、status、evidence_ids、issue），queries列表（每项必须是{"needs":["r1"],"purpose":"目的","sql":"SELECT ..."}对象），calculations列表，revise_answer布尔值。补充工具请求按下方协议；证据足够则两个列表为空。若输入draft_answer，检查完整回答及错误，不为改写而改写。',
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
        routed=False,
        aligned=False,
        checked=False,
    ):
        self.client = client
        self.bundle = copy.deepcopy(bundle).validate()
        self.max_calls = max_calls
        self.reconciled = (
            reconciled or replanned or patched or routed or aligned or checked
        )
        self.replanned = replanned or patched or routed or aligned or checked
        self.patched = patched
        self.routed = routed or aligned or checked
        self.aligned = aligned or checked
        self.checked = checked
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
                "variant": "cooperative_v7"
                if self.checked
                else "cooperative_v6"
                if self.aligned
                else "cooperative_v5"
                if self.routed
                else "cooperative_v4"
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
        if not self.aligned:
            selected = self.bundle.retrieve(
                query_context + " " + observations.catalog_text()
            )
        working = {"question": question, "history": history}
        if self.reconciled and self.dialog_contract:
            working["previous_task_contract"] = self.dialog_contract
            working["history_contract_status"] = (
                "Previous model proposal; use for entity continuity, not as factual proof. Current explicit user intent overrides it."
            )

        def call(role, view, cells, final=False, patch_mode=False, phase=None):
            memory = self.bundle.project(role, selected)
            if self.aligned and (not selected or phase == "routing_draft"):
                memory = {"guidance": "", "skills": []}
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
                + (
                    CHECKED_OUTPUTS.get(role, CONTRACTS[role])
                    if self.checked
                    else CONTRACTS[role]
                )
            )
            if self.reconciled:
                supplement = RECONCILED_CONTRACTS[role]
                if self.checked:
                    supplement = supplement.replace(
                        "在SQL中计算", "通过calculations或必要的SQL计算"
                    ).replace("实际SQL结果", "实际工具结果")
                prompt += "\n协作交接补充契约：\n" + supplement
            if self.replanned:
                prompt += "\n可修订语义契约：\n" + REPLANNED_CONTRACTS[role]
            if self.checked:
                prompt += "\n原始单元格计算契约：\n" + CHECKED_CONTRACTS[role]
            if (
                self.routed
                and self.bundle.skills
                and role == "SemanticPlanner"
                and phase != "selected_memory_plan"
            ):
                prompt += (
                    "\n从available_joint_skills按当前用户需求的语义选择零至三项适用技能，不依赖字面关键词或语言一致。"
                    "先检查适用条件和排除条件，将所选技能的规划提示落实到对应requirements；不适用则不选。"
                    "每个选择理由须指出当前问题或原始观察中触发该技能特有条件的线索，不可仅因都涉及汇总、比较等通用功能就选；条件尚无依据时不强行套用。"
                    '在计划JSON增加skill_selection:[{"skill_id":"已知技能ID","reason":"适用理由","need_ids":["已有需求ID"]}]。'
                    "选择将作为一个整体传给后续所有角色；不得臆造技能ID。"
                )
            if phase == "selected_memory_plan":
                prompt += (
                    "\n本轮技能选择已经固定。请使用所选技能的完整规划规则核对并完善draft_contract，"
                    "形成交给查询和验证角色的同一份计划。不得重新选择技能；规则不适用时不要强行套用。"
                    "没有所选技能时，仅依据用户问题和原始观察核对草案。"
                    "保留原有需求ID与用户要求，可以补充新需求，但不能因技能内容而删除用户要求、"
                    "缩短时间范围或把原始指标换成另一指标。未知的日期转换、列含义和公式留待实际查询核实，"
                    "不得把未经工具验证的推算写成已确认事实。返回完整的tables和requirements JSON。"
                )
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
            if phase:
                record["phase"] = phase
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
                if self.checked and isinstance(item, str):
                    # Normalize only the schema; retain the exact SQL and use
                    # the same read-only guard, budget and executor as usual.
                    item = {
                        "sql": item,
                        "purpose": "SQL string normalized to request object",
                    }
                    trace.append(
                        {
                            "role": "ToolSchemaNormalization",
                            "author": author,
                            "change": "SQL string -> object; query text unchanged",
                        }
                    )
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

        def calculate(items, author):
            if not self.checked or not isinstance(items, list):
                return 0
            count = 0
            requirements = {
                r.get("id")
                for r in working_memory.contract.get("requirements", [])
                if isinstance(r, dict) and isinstance(r.get("id"), str)
            }
            for spec in items[:4]:
                if budget.used_db >= budget.db_calls:
                    break
                source = None
                try:
                    sql = compile_metric(spec)
                    if spec["need_id"] not in requirements:
                        raise ValueError("Calculation must link to an existing need")
                    sig = "metric:" + fingerprint(spec)
                    if sig in signatures:
                        continue
                    signatures.add(sig)
                    budget.used_db += 1
                    source = execute(Path(store) / "tables.sqlite", sql, row_limit=24)
                    result = evaluate_metric(spec, source)
                except (ValueError, ArithmeticError, TypeError) as exc:
                    result = {"ok": False, "error": str(exc)[:600]}
                rec = {
                    "id": f"e{len(evidence) + 1}",
                    "kind": "checked_metric",
                    "needs": [spec.get("need_id")] if isinstance(spec, dict) else [],
                    **result,
                }
                evidence.append(rec)
                trace.append(
                    {
                        "role": "MetricExecutor",
                        "author": author,
                        "execution": rec,
                        "source_execution": source,
                        "requested_calculation": spec,
                    }
                )
                count += 1
            return count

        overview = observations.render(query_context, max_chars=18000)
        planner_view = (
            {"available_joint_skills": self.bundle.routing_index()}
            if self.routed and self.bundle.skills
            else {}
        )
        routing = {"decisions": []}
        contract = (
            call(
                "SemanticPlanner",
                planner_view,
                overview,
                phase="routing_draft" if self.aligned else None,
            )
            or {}
        )
        if self.routed and self.bundle.skills:
            selected, routing = self.bundle.route(contract, selected)
            trace.append(
                {
                    "role": "MemoryRouter",
                    "selected_skill_ids": [skill.id for skill in selected],
                    "semantic_status": "Model-proposed applicability; not oracle verified",
                    **routing,
                }
            )
        if self.aligned:
            # Fallback routes have a reason but no semantic decisions. Aligned
            # planning starts with no keyword-selected skills, so keep it empty.
            decisions = routing.get("decisions", [])
            # Empty-memory controls also refine once: learning must not receive
            # an unreported extra planning opportunity relative to its control.
            refined = call(
                "SemanticPlanner",
                {
                    "draft_contract": contract,
                    "fixed_skill_selection": decisions,
                },
                overview,
                phase="selected_memory_plan",
            )
            requirements = (refined or {}).get("requirements", [])
            original_requirements = contract.get("requirements", [])
            original_ids = {
                r["id"]
                for r in (
                    original_requirements
                    if isinstance(original_requirements, list)
                    else []
                )
                if isinstance(r, dict) and isinstance(r.get("id"), str)
            }
            refined_ids = (
                [r.get("id") for r in requirements if isinstance(r, dict)]
                if isinstance(requirements, list)
                else []
            )
            valid = (
                bool(refined)
                and isinstance(requirements, list)
                and len(refined_ids) == len(requirements)
                and all(isinstance(i, str) and i for i in refined_ids)
                and len(set(refined_ids)) == len(refined_ids)
                and original_ids <= set(refined_ids)
                and isinstance(refined.get("tables"), list)
                and all(
                    isinstance(t, str) and t in observations.names
                    for t in refined["tables"]
                )
            )
            trace.append(
                {
                    "role": "MemoryPlanning",
                    "status": "accepted" if valid else "rejected",
                    "selected_skill_ids": [skill.id for skill in selected],
                    "reason": "Plan retains original requirement IDs and known tables"
                    if valid
                    else "Invalid refinement; retain draft and disable downstream learned guidance and skill views",
                }
            )
            if valid:
                contract = refined
                contract["skill_selection"] = decisions
            else:
                selected = []
        if self.reconciled:
            working_memory = WorkingMemory(
                contract,
                allow_requirement_revisions=self.replanned,
                guard_checked_metrics=self.checked,
            )
        tables = contract.get("tables", [])
        if not isinstance(tables, list):
            tables = []
        tables = [t for t in tables if isinstance(t, str)]
        cells = observations.render(query_context, tables, max_chars=26000)
        plan = call("QueryBuilder", {"contract": contract}, cells) or {}
        query(plan.get("queries", []), "QueryBuilder")
        calculate(plan.get("calculations", []), "QueryBuilder")
        checks = []
        # v7 reserves composer + final review + repair under the same seven-call
        # ceiling. Both empty and learned-memory arms follow this controller.
        for _ in range(1 if self.checked else 2):
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
            additional = query(review.get("queries", []), "EvidenceVerifier")
            additional += calculate(review.get("calculations", []), "EvidenceVerifier")
            if not additional:
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
            additional += calculate(review.get("calculations", []), "AnswerReview")
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
