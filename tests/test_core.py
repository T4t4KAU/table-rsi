import copy
import sqlite3

import pytest

from tablersi.model import Budget
from tablersi.sql import (
    GuardError,
    compile_dag,
    execute,
    local_literal_patch,
    parse,
    signature,
)
from tablersi.state import (
    Evidence,
    State,
    admissible,
    audit_gate,
    commit_attempt,
    propose,
)


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.execute("create table sales (item text, gross real, returns real)")
    c.executemany(
        "insert into sales values (?,?,?)", [("A", 100, 40), ("B", 90, 0), ("B", 90, 0)]
    )
    c.commit()
    c.close()
    return p


def test_readonly_and_bag_null_semantics(db):
    assert execute(db, "select sum(gross-returns) from sales")["rows"] == [(240.0,)]
    for s in [
        "drop table sales",
        "select 1; delete from sales",
        "PRAGMA writable_schema=ON",
        "SELECT load_extension('/tmp/foo')",
        "ATTACH '/tmp/a' AS a",
    ]:
        assert not execute(db, s)["ok"]
    assert execute(db, "select count(*) from sales")["rows"] == [(3,)]


def test_cte_dag_and_scope(db):
    q = "WITH a AS (SELECT * FROM sales WHERE gross>90) SELECT item FROM a"
    assert execute(db, q)["rows"] == [("A",)]
    d = compile_dag(q)
    assert any(n["op"] == "Filter" for n in d["nodes"])
    assert any(n["op"] == "Scan" and n["inputs"] for n in d["nodes"])


def test_local_patch_authority(db):
    s = "select item from sales where gross>90"
    p, invalid = local_literal_patch(s, signature(s), 0, "90", "95")
    assert invalid
    assert execute(db, p)["rows"] == [("A",)]
    with pytest.raises(GuardError):
        local_literal_patch(s, "wrong", 0, "90", "95")
    with pytest.raises(GuardError):
        local_literal_patch(
            "select * from sales limit 1",
            signature("select * from sales limit 1"),
            0,
            "1",
            "2",
        )
    with pytest.raises(GuardError):
        local_literal_patch(s, signature(s), 0, "90", "0 OR 1=1")


def test_evidence_authority():
    e = Evidence("E0", "executor", "t1", "execution", True)
    assert admissible(e, "executability", "t1")
    assert not admissible(e, "success", "t1")
    e1 = Evidence("E1", "metamorphic", "t1", "permutation", True)
    assert admissible(e1, "property", "t1", "permutation")
    assert not admissible(e1, "success", "t1")
    assert not admissible(e1, "property", "t2", "permutation")
    assert not admissible(
        Evidence("E2", "model_guess", "t1", "task", 1), "success", "t1"
    )
    assert admissible(
        Evidence("E2", "authorized_development", "t1", "task", 1), "success", "t1"
    )


def test_rejection_can_update_improver_but_not_solver():
    s = State()
    z = copy.deepcopy(s)
    z.policy.replan_rounds = 2
    n = commit_attempt(
        s, z, "more_search", "a0", {"accept": False, "gain": -0.2}, level=2
    )
    assert n.policy == s.policy and n.version == 0 and n.meta["more_search"] < 0
    assert not s.meta and not s.spent_audits
    with pytest.raises(ValueError):
        commit_attempt(n, z, "more_search", "a0", {"accept": True, "gain": 1}, 2)
    k = commit_attempt(
        s, z, "more_search", "a0", {"accept": True, "gain": 0.3}, level=1
    )
    assert k.version == 1 and not k.meta


def test_fresh_audit_requires_evidence():
    assert not audit_gate([0], [1], ["one"])["accept"]
    assert audit_gate([0] * 8, [1] * 8, list("aabbccdd"))["accept"]
    assert not audit_gate([1] * 8, [0] * 8, list("aabbccdd"))["accept"]


def test_proposals_do_not_mutate_state():
    s = State()
    old = s.dump()
    p = propose(s, 2)
    assert p and s.dump() == old
    for _, z in p:
        z.policy.validate()


def test_hard_reservations():
    b = Budget(calls=1, input_tokens=20, output_tokens=10)
    assert b.allowed("abc", 5)
    assert not b.allowed("中" * 10, 5)
    b.used_calls = 1
    assert not b.allowed("a", 1)


def test_rsi_shared_initial_randomization_and_feedback_effect():
    s = State()
    assert [o for o, _ in propose(s, 1)] == [o for o, _ in propose(s, 2)]
    s.meta = {"more_search": 5, "less_search": -5}
    orders1 = [[o for o, _ in propose(s, 1, seed=i)] for i in range(20)]
    orders2 = [[o for o, _ in propose(s, 2, seed=i)] for i in range(20)]
    assert orders1 != orders2
    assert sum(x[0] == "more_search" for x in orders2) > sum(
        x[0] == "more_search" for x in orders1
    )


def test_sql_resource_limits(db):
    assert not execute(db, "select randomblob(2000000)")["ok"]
    with pytest.raises(GuardError):
        parse("select " + (" " * 65536) + "1")


def test_local_client_does_not_read_cloud_key_environment(monkeypatch):
    from tablersi.model import Client

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-marker")
    assert Client("http://127.0.0.1:31001/v1", "local-model", local=True).key == ""
    assert (
        Client("https://api.deepseek.com", "judge-model", local=False).key
        == "test-secret-marker"
    )


def test_signatures_preserve_unsupported_function_arguments():
    a = "SELECT INSTR('a-a-a','a',1,2)"
    b = "SELECT INSTR('a-a-a','a',1,3)"
    assert signature(a) != signature(b)
    assert signature("SELECT 1") == signature("SELECT   1")


def test_local_patch_cannot_silently_rewrite_unrelated_function():
    q = "SELECT INSTR(item,'a',1,2) FROM sales WHERE gross>90"
    literals = list(parse(q).find_all(__import__("sqlglot").exp.Literal))
    index = next(i for i, literal in enumerate(literals) if str(literal.this) == "90")
    with pytest.raises(GuardError):
        local_literal_patch(q, signature(q), index, "90", "95")
