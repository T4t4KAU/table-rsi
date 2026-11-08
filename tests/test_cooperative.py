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
