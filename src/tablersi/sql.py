"""Fixed SQL permissions, relational dependency DAGs, and isolated read-only execution."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sqlite3
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel


class GuardError(ValueError):
    pass


def parse(sql):
    if len(sql.encode("utf-8")) > 65536:
        raise GuardError("SQL text exceeds fixed byte limit")
    try:
        trees = sqlglot.parse(sql, read="sqlite")
    except Exception as e:
        raise GuardError(f"parse: {e}")
    if len(trees) != 1 or not isinstance(
        trees[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)
    ):
        raise GuardError("one read-only SELECT query required")
    tree = trees[0]
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Command,
        exp.Alter,
        exp.Into,
        exp.Transaction,
    )
    if any(isinstance(n, forbidden) for n in tree.walk()):
        raise GuardError("forbidden statement")
    return tree


def ast_payload(tree):
    # Preserve every argument; exclude only parser source-position metadata.
    return [{k: v for k, v in item.items() if k != "m"} for item in tree.dump()]


def signature(sql):
    return hashlib.sha256(
        json.dumps(
            ast_payload(parse(sql)), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str]
    expression: str
    signature: str


def compile_dag(sql):
    tree = parse(sql)
    nodes = []

    def add(op, e, inputs):
        text = e.sql() if isinstance(e, exp.Expression) else str(e)
        nid = f"n{len(nodes)}"
        sig = hashlib.sha256((op + text + "|".join(inputs)).encode()).hexdigest()
        nodes.append(Node(nid, op, inputs, text, sig))
        return nid

    # Nested scopes form separate subgraphs; dependencies remain explicit.
    def visit(q):
        if isinstance(q, (exp.Union, exp.Intersect, exp.Except)):
            return add(type(q).__name__, q, [visit(q.this), visit(q.expression)])
        sources = []
        f = q.args.get("from_")
        if f:
            t = f.this
            sources.append(
                visit(t.this) if isinstance(t, exp.Subquery) else add("Scan", t, [])
            )
        for j in q.args.get("joins", []):
            t = j.this
            right = visit(t.this) if isinstance(t, exp.Subquery) else add("Scan", t, [])
            sources = [add("Join", j, [*sources, right])]
        tail = sources
        for key, op in [
            ("where", "Filter"),
            ("group", "Aggregate"),
            ("having", "Having"),
        ]:
            if q.args.get(key):
                tail = [add(op, q.args[key], tail)]
        if not q.args.get("group") and any(
            isinstance(n, exp.AggFunc) for e in q.expressions for n in e.walk()
        ):
            tail = [add("Aggregate", "scalar aggregation", tail)]
        tail = [
            add(
                "Project",
                exp.Tuple(expressions=[e.copy() for e in q.expressions]),
                tail,
            )
        ]
        for key, op in [
            ("distinct", "Distinct"),
            ("order", "Sort"),
            ("limit", "Limit"),
            ("offset", "Offset"),
        ]:
            if q.args.get(key):
                tail = [add(op, q.args[key], tail)]
        return tail[0]

    ctes = tree.args.get("with_")
    cte_roots = {}
    if ctes:
        for c in ctes.expressions:
            cte_roots[c.alias_or_name] = visit(c.this)
    root = visit(tree)
    for n in nodes:
        if n.op == "Scan":
            try:
                name = sqlglot.parse_one(
                    n.expression, read="sqlite", into=exp.Table
                ).name
            except Exception:
                name = n.expression
            if name in cte_roots:
                n.inputs.append(cte_roots[name])
    return {
        "nodes": [asdict(n) for n in nodes],
        "root": root,
        "sql_signature": signature(sql),
    }


def descendants(dag, node):
    invalid = {node}
    changed = True
    while changed:
        before = len(invalid)
        invalid.update(n["id"] for n in dag["nodes"] if set(n["inputs"]) & invalid)
        changed = len(invalid) != before
    return invalid


def local_literal_patch(sql, old_signature, index, old_value, new_value):
    if signature(sql) != old_signature:
        raise GuardError("stale patch signature")
    tree = parse(sql)
    lits = list(tree.find_all(exp.Literal))
    if not (0 <= index < len(lits)) or str(lits[index].this) != str(old_value):
        raise GuardError("literal precondition mismatch")
    target = lits[index]
    # Join, source, projection, aggregation, sort and limit edits are not local.
    parent = target.parent
    while parent is not None and not isinstance(
        parent, (exp.Where, exp.Join, exp.Select)
    ):
        parent = parent.parent
    if not isinstance(parent, exp.Where):
        raise GuardError("local patch must affect one WHERE predicate")
    if not target.is_string:
        try:
            if not Decimal(str(new_value)).is_finite():
                raise ValueError()
        except (InvalidOperation, ValueError):
            raise GuardError("numeric literal required")
    replacement = exp.Literal(this=str(new_value), is_string=target.is_string)
    target.replace(replacement)
    try:
        patched = tree.sql(dialect="sqlite", unsupported_level=ErrorLevel.RAISE)
    except Exception as ex:
        raise GuardError("lossless SQLite serialization unavailable: " + str(ex))
    # The sole in-memory mutation above is the authorized literal replacement.
    # Rendering must preserve that exact tree, including unrelated function arity
    # and every other literal, binding, edge and operator.
    if ast_payload(tree) != ast_payload(parse(patched)):
        raise GuardError("serialization changed bindings, structure or other values")
    # One changed Filter node; all downstream caches must be invalidated.
    dag = compile_dag(sql)
    targets = [
        n["id"]
        for n in dag["nodes"]
        if n["op"] == "Filter" and old_value in n["expression"]
    ]
    return patched, sorted(set().union(*(descendants(dag, n) for n in targets)))


_ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_BAD_FUNCTIONS = {"load_extension", "readfile", "writefile", "fts3_tokenizer", "eval"}


def _run(db, sql, row_limit, queue, timeout):
    conn = None
    try:
        conn = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024 * 1024)
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 65536)
        conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 200)
        conn.execute("PRAGMA query_only=ON")
        conn.enable_load_extension(False)

        def auth(action, a, b, c, d):
            if action not in _ALLOWED:
                return sqlite3.SQLITE_DENY
            if (
                action == sqlite3.SQLITE_FUNCTION
                and (b or a or "").lower() in _BAD_FUNCTIONS
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(auth)
        start = time.monotonic()
        conn.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 1000)
        cur = conn.execute(sql)
        rows = cur.fetchmany(row_limit + 1)
        queue.put(
            {
                "ok": True,
                "columns": [x[0] for x in cur.description],
                "rows": rows[:row_limit],
                "truncated": len(rows) > row_limit,
                "elapsed_s": time.monotonic() - start,
            }
        )
    except Exception as e:
        queue.put({"ok": False, "error": str(e)[:1200]})
    finally:
        if conn:
            conn.close()


def execute(db, sql, row_limit=80, timeout=8):
    start = time.monotonic()
    try:
        dag = compile_dag(sql)
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "evidence": "E0",
            "elapsed_s": time.monotonic() - start,
        }
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run, args=(str(db), sql, row_limit, queue, timeout))
    proc.start()
    try:
        result = queue.get(timeout=timeout + 3)
    except Exception:
        result = {"ok": False, "error": "execution timeout"}
    finally:
        proc.join(timeout=0.3)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        queue.close()
    return {
        **result,
        "evidence": "E0",
        "dag": dag,
        "elapsed_s": time.monotonic() - start,
    }
