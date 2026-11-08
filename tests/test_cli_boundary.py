import json
from types import SimpleNamespace

import pytest

from tablersi import cli
from tablersi.data import build_store
from tablersi.memory import Bundle
from tablersi.state import State


@pytest.mark.parametrize("external_oracle", [False, True])
@pytest.mark.parametrize("variant", ["direct", "cooperative", "cooperative_v2"])
def test_oracle_is_scorer_only_and_history_uses_predictions(
    tmp_path, monkeypatch, external_oracle, variant
):
    data = tmp_path / "data"
    data.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.csv").write_text("name,value\nA,3\n")
    build_store(source, data / "stores/g")
    (data / "manifest.json").write_text("{}")
    (data / "oracle.json").write_text(
        json.dumps(
            {
                "task0": [
                    {"score_points": ["SECRET_ORACLE_1"]},
                    {"score_points": ["SECRET_ORACLE_2"]},
                ]
            }
        )
    )
    repo = tmp_path / "repo"
    prompts = repo / "src/prompts"
    prompts.mkdir(parents=True)
    (prompts / "AgentEvalPrompt.py").write_text('SCORE_POINTS_ACC_JUDGE="fixture"')

    class FakeClient:
        requests = []

        def chat(self, messages, **kw):
            self.requests.append(messages)
            return {
                "content": f"prediction_{len(self.requests)}",
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                "elapsed_s": 0.1,
                "model": "fake",
            }

    class FakeJudge:
        references = []

        def score(self, q, answer, points):
            self.references.append(points)
            return {
                "total_metrics": len(points),
                "covered_metrics": 0,
                "coverage_ratio": 0.0,
            }

    client = FakeClient()
    judge = FakeJudge()
    monkeypatch.setattr(cli, "clients", lambda a: (client, judge))
    a = SimpleNamespace(
        data=data,
        repo=repo,
        local_model="fake",
        judge_model="fake-judge",
        seed=42,
        workers=1,
    )
    if external_oracle:
        a.oracle_file = tmp_path / "scorer-only.json"
        (data / "oracle.json").rename(a.oracle_file)
    task = {
        "id": "task0",
        "cluster": "cluster0",
        "split": "dev",
        "group": "g",
        "questions": ["First question", "Follow-up question"],
    }
    state = Bundle() if variant.startswith("cooperative") else State()
    records, report = cli.evaluate(a, state, variant, [task], tmp_path / "out")
    assert judge.references == [["SECRET_ORACLE_1"], ["SECRET_ORACLE_2"]]
    assert "SECRET_ORACLE" not in json.dumps(client.requests)
    first_turn_calls = records[0]["turns"][0]["budget"]["used_calls"]
    assert records[0]["turns"][0]["answer"].startswith("prediction_")
    assert records[0]["turns"][0]["answer"] in json.dumps(
        client.requests[first_turn_calls]
    )
    assert report["turns"] == 2
