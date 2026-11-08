"""Synthetic component diagnostic; not a TableAgentBench accuracy estimate."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from tablersi.model import Client, json_object
from tablersi.sql import execute, local_literal_patch, signature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:31001/v1")
    p.add_argument("--model", default="tablersi-qwen3-8b")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "synthetic.sqlite"
    c = sqlite3.connect(db)
    c.execute(
        "create table if not exists measurements (id integer, x integer, y integer)"
    )
    c.execute("delete from measurements")
    c.executemany(
        "insert into measurements values (?,?,?)",
        [(i, i * 7, (19 - i) * 3) for i in range(20)],
    )
    c.commit()
    c.close()
    client = Client(a.url, a.model)
    records = []
    for i in range(12):
        threshold = 15 + 7 * i
        structural = i >= 8
        question = f"Return id and x from measurements where x is strictly greater than {threshold}."
        original = f"SELECT id,x FROM measurements WHERE {'y' if structural else 'x'}>{threshold + 5 if not structural else threshold}"
        prompt = f'Question: {question}\nOriginal query: {original}\nAllowed action: change exactly one existing WHERE numeric literal; no column, operator, projection, table, or structure edits. If the query needs a forbidden edit, return {{"applicable":false}}. Otherwise return JSON {{"applicable":true,"index":0,"old":"old literal","new":"new literal"}}. Do not answer the query.'
        response = client.chat(
            [
                {
                    "role": "system",
                    "content": "You propose strictly scoped SQL patches. Return JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            seed=42 + i,
            temperature=0,
            json_mode=True,
        )
        rec = {
            "id": i,
            "kind": "requires_structural_replan"
            if structural
            else "repairable_literal",
            "question": question,
            "original": original,
            "response": response,
        }
        try:
            patch = json_object(response["content"])
            rec["proposal"] = patch
            if not patch.get("applicable"):
                rec["correct"] = structural
                rec["guard_accepted"] = False
            else:
                revised, invalid = local_literal_patch(
                    original,
                    signature(original),
                    int(patch["index"]),
                    str(patch["old"]),
                    str(patch["new"]),
                )
                result = execute(db, revised)
                target = execute(
                    db, f"SELECT id,x FROM measurements WHERE x>{threshold}"
                )
                rec.update(
                    revised=revised,
                    result=result,
                    guard_accepted=True,
                    correct=(
                        not structural
                        and result["ok"]
                        and Counter(map(tuple, result["rows"]))
                        == Counter(map(tuple, target["rows"]))
                    ),
                )
        except Exception as ex:
            rec.update(error=str(ex), correct=False, guard_accepted=False)
        records.append(rec)
    report = {
        "diagnostic": "known synthetic single-literal repair and structural-abstention component test; no task-wide accuracy claim",
        "cases": len(records),
        "correct": sum(r["correct"] for r in records),
        "repairable_correct": sum(
            r["correct"] for r in records if r["kind"] == "repairable_literal"
        ),
        "structural_abstention_correct": sum(
            r["correct"] for r in records if r["kind"] == "requires_structural_replan"
        ),
        "records": records,
    }
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
