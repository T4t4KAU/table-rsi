import sqlite3

from tablersi.agent import Agent
from tablersi.data import build_store, context
from tablersi.state import State


def test_original_rows_and_headers_preserved(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.csv").write_text("name,value\nA,3\n,\nB,5\n,\n")
    store = tmp_path / "store"
    cat, errors = build_store(source, store)
    assert not errors
    assert cat[0]["rows"] == 4
    conn = sqlite3.connect(store / "tables.sqlite")
    assert conn.execute("select _row,c1,c2 from t0 where _row=4").fetchone() == (
        4,
        "B",
        "5",
    )
    conn.close()


def test_missing_data_abstains_without_model(tmp_path):
    class Never:
        def chat(self, *a, **k):
            raise AssertionError("must not call model")

    result = Agent(Never(), State()).solve(tmp_path / "absent", "What is the total?")
    assert result["budget"]["used_calls"] == 0
    assert result["evidence_status"] == "missing source tables"


def test_invalid_json_usage_not_double_charged(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.csv").write_text("x\n3\n")
    build_store(source, tmp_path / "store")

    class Fake:
        def chat(self, *a, **k):
            return {
                "content": "invalid json",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "elapsed_s": 0.1,
            }

    result = Agent(Fake(), State()).solve(tmp_path / "store", "total x?")
    b = result["budget"]
    assert b["unknown_usage"] == 0
    assert b["used_input"] == 10 * b["used_calls"]
    assert b["used_output"] == 5 * b["used_calls"]


def test_context_lists_every_schema_and_explicit_bindings(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i in range(10):
        (source / f"{i}.csv").write_text("item,value\nA,3\nB,4\n")
    # Different file content avoids intentional exact-sheet deduplication.
    for i in range(10):
        (source / f"{i}.csv").write_text(f"item,value\nA,{i}\nB,4\n")
    build_store(source, tmp_path / "store")
    text = context(tmp_path / "store", "A", 9000)
    assert len(text) <= 9000
    for i in range(10):
        assert f"t{i}: file=" in text
    assert 'c1="item"' in text and 'c2="value"' in text
