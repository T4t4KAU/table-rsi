import json

import pytest

from tablersi.cooperative import CooperativeAgent
from tablersi.data import build_store
from tablersi.memory import Bundle, Skill, validate_teacher_abstraction
from tablersi.observations import Observations, evidence_text


def make_store(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.csv").write_text("item,value\nA,3\nB,5\n")
    store = tmp_path / "store"
    build_store(source, store)
    return store


def test_joint_memory_projects_roles_without_provenance():
    skill = Skill(
        "ratio",
        ["比例"],
        {"SemanticPlanner": "先明确分母", "EvidenceVerifier": "独立检查分母"},
        "比例计算",
        provenance=[{"gold": "PRIVATE_TRAINING_ANSWER"}],
    )
    bundle = Bundle(skills=[skill]).validate()
    selected = bundle.retrieve("计算比例")
    assert [s.id for s in selected] == ["ratio"]
    assert not bundle.retrieve("解释库存趋势")
    assert (
        bundle.project("SemanticPlanner", selected)["skills"][0]["instruction"]
        == "先明确分母"
    )
    assert not bundle.project("AnswerComposer", selected)["skills"]
    assert "PRIVATE_TRAINING_ANSWER" not in json.dumps(
        bundle.project("EvidenceVerifier", selected)
    )


def test_teacher_cannot_store_obvious_benchmark_answers():
    bundle = Bundle(role_guidance={"AnswerComposer": "答案为 3349.5"})
    with pytest.raises(ValueError, match="concrete number"):
        validate_teacher_abstraction(bundle)
    with pytest.raises(ValueError, match="two known roles"):
        Bundle(skills=[Skill("x", ["x"], {"QueryBuilder": "x"}, "x")]).validate()


def test_semantic_repair_runs_after_successful_wrong_query(tmp_path):
    store = make_store(tmp_path)

    class Fake:
        def __init__(self):
            self.messages = []
            self.responses = iter(
                [
                    {"tables": ["t0"], "requirements": [{"id": "r1", "need": "total"}]},
                    {
                        "queries": [
                            {
                                "sql": "SELECT SUM(CAST(c1 AS REAL)) AS total FROM t0 WHERE _row > 1"
                            }
                        ]
                    },
                    {
                        "checks": [
                            {
                                "need": "r1",
                                "status": "conflict",
                                "issue": "wrong column",
                            }
                        ],
                        "queries": [
                            {
                                "sql": "SELECT SUM(CAST(c2 AS REAL)) AS total FROM t0 WHERE _row > 1"
                            }
                        ],
                    },
                    {
                        "checks": [
                            {
                                "need": "r1",
                                "status": "supported",
                                "evidence_ids": ["e2"],
                            }
                        ],
                        "queries": [],
                    },
                    "合计为8，依据e2。",
                ]
            )

        def chat(self, messages, **kwargs):
            self.messages.append(messages)
            value = next(self.responses)
            return {
                "content": value if isinstance(value, str) else json.dumps(value),
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "elapsed_s": 0.1,
            }

    client = Fake()
    result = CooperativeAgent(client, Bundle()).solve(store, "What is the total value?")
    executed = [t["execution"] for t in result["trace"] if t["role"] == "Executor"]
    assert len(executed) == 2 and all(e["ok"] for e in executed)
    assert executed[0]["rows"] == [(0.0,)]
    assert executed[1]["rows"] == [(8.0,)]
    assert result["answer"] == "合计为8，依据e2。"
    assert result["budget"]["used_calls"] == 5
    assert "e2" in client.messages[-1][1]["content"]


def test_evidence_trimming_preserves_json_and_first_result():
    items = [
        {
            "id": f"e{i}",
            "ok": True,
            "sql": "SELECT x",
            "rows": [["x" * 100] for _ in range(30)],
        }
        for i in range(4)
    ]
    rendered = evidence_text(items, max_chars=1200)
    parsed = json.loads(rendered)
    assert len(rendered) <= 1200
    assert [e["id"] for e in parsed] == ["e0", "e1", "e2", "e3"]
    assert any(e.get("prompt_rows_omitted") for e in parsed)
    assert len(items[0]["rows"]) == 30


def test_long_sql_metadata_preserves_results_and_original_trace():
    items = [
        {
            "id": f"e{i}",
            "ok": True,
            "sql": "SELECT " + "x" * 5900,
            "columns": ["total"],
            "rows": [[i + 10]],
            "truncated": False,
        }
        for i in range(4)
    ]
    before = json.dumps(items)
    rendered = evidence_text(items, max_chars=1600)
    records = json.loads(rendered)
    assert len(rendered) <= 1600
    assert [record["id"] for record in records] == [f"e{i}" for i in range(4)]
    assert [record["rows"] for record in records] == [[[i + 10]] for i in range(4)]
    assert all(record["ok"] and not record["truncated"] for record in records)
    assert all("sql" in record["prompt_fields_omitted"] for record in records)
    assert all("sql" not in record for record in records)
    assert json.dumps(items) == before


def test_oversized_column_labels_never_leave_unlabelled_result_values():
    items = [
        {
            "id": "e1",
            "ok": False,
            "error": "large diagnostic " * 500,
            "columns": ["column " * 1000],
            "rows": [[42]],
        }
    ]
    before = json.dumps(items)
    rendered = evidence_text(items, max_chars=500)
    record = json.loads(rendered)[0]
    assert len(rendered) <= 500
    assert record["id"] == "e1" and record["ok"] is False
    assert set(record["prompt_fields_omitted"]) == {"error", "columns", "rows"}
    assert record["rows"] == [] and "columns" not in record
    assert json.dumps(items) == before


def test_evidence_normal_path_is_unchanged():
    items = [{"id": "e1", "ok": True, "rows": [[1], [2], [3]]}]
    expected = [{"id": "e1", "ok": True, "rows": [[1], [2]], "prompt_rows_omitted": 1}]
    budget = len(json.dumps(expected, ensure_ascii=False))
    # Force trimming with a long final row, while retaining the normal metadata.
    items[0]["rows"][-1] = ["x" * 100]
    assert evidence_text(items, max_chars=budget) == json.dumps(
        expected, ensure_ascii=False
    )


def test_focused_observations_keep_raw_coordinates(tmp_path):
    store = make_store(tmp_path)
    text = Observations(store).render("total value", ["t0"], max_chars=4000)
    assert 'c1="item"' in text and 'c2="3"' in text
    assert "row 2:" in text
    assert "shown 3/3" in text


def test_missing_store_never_calls_local_model(tmp_path):
    class Never:
        def chat(self, *args, **kwargs):
            raise AssertionError("Must not call")

    result = CooperativeAgent(Never(), Bundle()).solve(tmp_path / "absent", "total?")
    assert result["budget"]["used_calls"] == 0


def test_reconciled_solver_updates_binding_and_corrects_final_answer(tmp_path):
    store = make_store(tmp_path)

    class Fake:
        def __init__(self):
            self.responses = iter(
                [
                    {
                        "tables": ["t0"],
                        "requirements": [
                            {"id": "r1", "need": "total", "binding": "c1"}
                        ],
                    },
                    {
                        "queries": [
                            {
                                "sql": "SELECT SUM(CAST(c1 AS REAL)) AS total FROM t0 WHERE _row > 1"
                            }
                        ]
                    },
                    {
                        "withdraw_evidence": [{"id": "e1", "reason": "wrong column"}],
                        "queries": [
                            {
                                "sql": "SELECT SUM(CAST(c2 AS REAL)) AS total FROM t0 WHERE _row > 1"
                            }
                        ],
                    },
                    {
                        "binding_corrections": [
                            {"need_id": "r1", "binding": "c2", "evidence_ids": ["e2"]}
                        ],
                        "queries": [],
                    },
                    "合计为80。",
                    {
                        "checks": [
                            {"status": "conflict", "issue": "e2 gives 8, not 80"}
                        ],
                        "queries": [],
                        "revise_answer": True,
                    },
                    "合计为8，依据e2。",
                ]
            )

        def chat(self, messages, **kwargs):
            value = next(self.responses)
            return {
                "content": value if isinstance(value, str) else json.dumps(value),
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "elapsed_s": 0.1,
            }

    result = CooperativeAgent(Fake(), Bundle(), reconciled=True).solve(
        store, "total value?"
    )
    assert result["answer"] == "合计为8，依据e2。"
    assert result["budget"]["used_calls"] == 7
    assert result["budget"]["unknown_usage"] == 0
    assert (
        result["working_memory"]["current_contract"]["requirements"][0]["binding"]
        == "c2"
    )
    answers = [t for t in result["trace"] if t["role"] == "AnswerComposer"]
    assert all(t["reads_evidence"] == ["e2"] for t in answers)
    assert len([t for t in result["trace"] if t["role"] == "Executor"]) == 2


def test_replanned_contract_reaches_answer_composer(tmp_path):
    store = make_store(tmp_path)

    class Fake:
        def __init__(self):
            self.count = 0

        def chat(self, messages, **kwargs):
            self.count += 1
            payloads = [
                {
                    "tables": ["t0"],
                    "requirements": [{"id": "r1", "formula": "count rows"}],
                },
                {
                    "queries": [
                        {
                            "sql": "SELECT SUM(CAST(c2 AS REAL)) AS total FROM t0 WHERE _row > 1"
                        }
                    ]
                },
                {
                    "requirement_revisions": [
                        {
                            "need_id": "r1",
                            "updates": {"formula": "sum values"},
                            "evidence_ids": ["e1"],
                            "reason": "The user requests a sum, not a row count",
                        }
                    ]
                },
                "Total is 8.",
                {"revise_answer": False},
            ]
            if self.count == 4:
                prompt = messages[-1]["content"]
                current = json.loads(
                    prompt.split("上游工作产物（均待验证）：\n")[1].split(
                        "\n执行证据："
                    )[0]
                )
                assert (
                    current["current_contract"]["requirements"][0]["formula"]
                    == "sum values"
                )
            value = payloads[self.count - 1]
            return {
                "content": value if isinstance(value, str) else json.dumps(value),
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "elapsed_s": 0.1,
            }

    result = CooperativeAgent(Fake(), Bundle(), replanned=True).solve(
        store, "Sum values"
    )
    assert result["answer"] == "Total is 8."
    assert result["variant"] == "cooperative_v3"
    assert result["budget"]["unknown_usage"] == 0
    assert (
        result["working_memory"]["correction_history"][0]["action"]
        == "requirement_revision"
    )


def test_evidence_linked_answer_patch_preserves_other_claims(tmp_path):
    store = make_store(tmp_path)

    class Fake:
        def __init__(self):
            self.count = 0

        def chat(self, messages, **kwargs):
            self.count += 1
            responses = [
                {"tables": ["t0"], "requirements": [{"id": "r1", "need": "sum"}]},
                {
                    "queries": [
                        {
                            "sql": "SELECT SUM(CAST(c2 AS REAL)) AS total FROM t0 WHERE _row > 1"
                        }
                    ]
                },
                {"queries": []},
                "Total is 80. Keep the supported explanation.",
                {"revise_answer": True, "checks": [{"issue": "e1 gives 8"}]},
                {
                    "edits": [
                        {
                            "old": "Total is 80.",
                            "new": "Total is 8.",
                            "reason": "Correct total using e1",
                            "evidence_ids": ["e1"],
                        }
                    ]
                },
            ]
            if self.count == 6:
                assert kwargs["json_mode"]
                assert "内部答案编辑模式" in messages[-1]["content"]
            value = responses[self.count - 1]
            return {
                "content": value if isinstance(value, str) else json.dumps(value),
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "elapsed_s": 0.1,
            }

    result = CooperativeAgent(Fake(), Bundle(), patched=True).solve(store, "Total?")
    assert result["answer"] == "Total is 8. Keep the supported explanation."
    assert result["variant"] == "cooperative_v4"
    assert result["budget"]["used_calls"] == 6
    assert result["budget"]["unknown_usage"] == 0
    assert result["trace"][-1]["role"] == "AnswerPatch"
    assert result["trace"][-1]["applied"]


def test_semantic_routing_activates_cross_language_joint_skill_without_extra_call(
    tmp_path,
):
    store = make_store(tmp_path)
    skill = Skill(
        "stock_flow",
        ["库存"],
        {
            "SemanticPlanner": "Compare stock with flow.",
            "QueryBuilder": "ROUTED_QUERY_INSTRUCTION",
            "EvidenceVerifier": "ROUTED_CHECK_INSTRUCTION",
            "AnswerComposer": "ROUTED_ANSWER_INSTRUCTION",
        },
        "Compare inventory risk using stock and outflow.",
        provenance=[{"private": "SECRET_TEACHER_REFERENCE"}],
    )
    bundle = Bundle(skills=[skill])
    assert not bundle.retrieve("Compare inventory risk.")

    class Local:
        def __init__(self):
            self.messages = []
            self.responses = iter(
                [
                    {
                        "tables": ["t0"],
                        "requirements": [{"id": "r1", "need": "risk"}],
                        "skill_selection": [
                            {
                                "skill_id": "stock_flow",
                                "reason": "inventory and flow are relevant",
                                "need_ids": ["r1"],
                            }
                        ],
                    },
                    {"queries": []},
                    {"checks": [], "queries": []},
                    "The available table does not record stock or outflow.",
                    {"checks": [], "queries": [], "revise_answer": False},
                ]
            )

        def chat(self, messages, **kwargs):
            self.messages.append(messages)
            value = next(self.responses)
            return {
                "content": value if isinstance(value, str) else json.dumps(value),
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }

    client = Local()
    result = CooperativeAgent(client, bundle, routed=True).solve(
        store, "Compare inventory risk."
    )
    assert result["budget"]["used_calls"] == 5
    assert result["retrieved_skill_ids"] == ["stock_flow"]
    assert "ROUTED_QUERY_INSTRUCTION" in client.messages[1][1]["content"]
    assert "ROUTED_CHECK_INSTRUCTION" in client.messages[2][1]["content"]
    assert "ROUTED_ANSWER_INSTRUCTION" in client.messages[3][1]["content"]
    assert "SECRET_TEACHER_REFERENCE" not in json.dumps(client.messages)
    assert any(t.get("mode") == "semantic_selection" for t in result["trace"])


def test_semantic_router_rejects_unknown_ids_and_unlinked_requirements():
    skill = Skill("s", ["x"], {"SemanticPlanner": "p", "QueryBuilder": "q"}, "a")
    bundle = Bundle(skills=[skill])
    contract = {
        "requirements": [{"id": "r1"}],
        "skill_selection": [
            {"skill_id": "unknown", "reason": "r", "need_ids": ["r1"]},
            {"skill_id": "s", "reason": "r", "need_ids": ["unknown_need"]},
        ],
    }
    selected, event = bundle.route(contract, [])
    assert selected == [] and event["mode"] == "keyword_fallback"
    contract["skill_selection"] = []
    selected, event = bundle.route(contract, [skill])
    assert selected == [] and event["mode"] == "semantic_selection"


def test_router_empty_memory_preserves_v3_prompts_and_call_budget(tmp_path):
    store = make_store(tmp_path)

    class Local:
        def __init__(self):
            self.messages = []

        def chat(self, messages, **kwargs):
            self.messages.append(messages)
            return {
                "content": "{}" if kwargs.get("json_mode") else "A has value 3.",
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }

    old, new = Local(), Local()
    a = CooperativeAgent(old, Bundle(), replanned=True).solve(store, "Value?")
    b = CooperativeAgent(new, Bundle(), routed=True).solve(store, "Value?")
    assert old.messages == new.messages
    assert a["budget"] == b["budget"]
    assert a["answer"] == b["answer"]
