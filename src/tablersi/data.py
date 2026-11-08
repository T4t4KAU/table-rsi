"""Ground-truth-free solver views and deterministic table-group partitions."""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import openpyxl
import xlrd


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def value(v):
    if v is None:
        return None
    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
        return v.isoformat()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v if isinstance(v, (str, int, float)) else str(v)


def sheets(path):
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8-sig", "gb18030", "utf-16"):
            try:
                with path.open(encoding=enc, newline="") as f:
                    rows = list(csv.reader(f))
                return [(path.stem, rows)]
            except UnicodeError:
                continue
        raise ValueError("undecodable CSV: " + path.name)
    if path.suffix.lower() == ".xlsx":
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        result = [
            (ws.title, [[value(v) for v in r] for r in ws.iter_rows(values_only=True)])
            for ws in wb
        ]
        wb.close()
        return result
    wb = xlrd.open_workbook(path)
    return [
        (ws.name, [[value(v) for v in ws.row_values(i)] for i in range(ws.nrows)])
        for ws in wb.sheets()
    ]


def folder_key(file_path):
    m = re.search(r"(chinese_table|english_table)/(.+)", file_path)
    if not m:
        raise ValueError("unknown table directory")
    return m.group(0)


def build_store(folder: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    db = dest / "tables.sqlite"
    db.unlink(missing_ok=True)
    conn = sqlite3.connect(db)
    catalog = []
    seen = set()
    errors = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in (".csv", ".xlsx", ".xls") or p.name.startswith("~$"):
            continue
        try:
            items = sheets(p)
        except Exception as e:
            errors.append({"file": p.name, "error": str(e)})
            continue
        for sheet, rows in items:
            rows = [[value(v) for v in r] for r in rows]
            while rows and not any(v not in (None, "") for v in rows[-1]):
                rows.pop()
            if not rows:
                continue
            width = max(len(r) for r in rows)
            while width > 1 and all(
                len(r) < width or r[width - 1] in (None, "") for r in rows
            ):
                width -= 1
            rows = [r[:width] + [None] * max(0, width - len(r)) for r in rows]
            sig = digest(rows)
            if sig in seen:
                continue
            seen.add(sig)
            name = f"t{len(catalog)}"
            # No invented flattening: preserve row positions and generic column identities.
            conn.execute(
                f'CREATE TABLE "{name}" (_row INTEGER PRIMARY KEY, '
                + ",".join(f'"c{i + 1}"' for i in range(width))
                + ")"
            )
            conn.executemany(
                f'INSERT INTO "{name}" VALUES ({",".join("?" for _ in range(width + 1))})',
                [(i + 1, *r) for i, r in enumerate(rows)],
            )
            catalog.append(
                {
                    "name": name,
                    "file": p.name,
                    "sheet": sheet,
                    "columns": [f"c{i + 1}" for i in range(width)],
                    "rows": len(rows),
                    "content_sha256": sig,
                }
            )
    conn.commit()
    conn.close()
    (dest / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2)
    )
    (dest / "ingest_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2)
    )
    return catalog, errors


def prepare(repo: Path, out: Path, seed="20261108"):
    raw = json.loads((repo / "dataset/samples.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    groups = sorted({folder_key(x["file_path"]) for x in raw})
    parent = {k: k for k in groups}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        parent[find(a)] = find(b)

    owner = {}
    records = {}
    catalogs = {}
    for k in groups:
        gid = digest(k)[:16]
        folder = repo / "dataset/tables" / k
        if not folder.exists():
            catalogs[k] = []
            continue
        cat, err = build_store(folder, out / "stores" / gid)
        catalogs[k] = cat
        for c in cat:
            sig = c["content_sha256"]
            if sig in owner:
                union(k, owner[sig])
            else:
                owner[sig] = k
        records[k] = {"gid": gid, "tables": len(cat), "ingest_errors": err}
    components = defaultdict(list)
    for k in groups:
        components[find(k)].append(k)
    comps = sorted(components.values(), key=lambda ks: digest([seed, sorted(ks)]))
    # All copies of the same raw sheet remain in the same group.
    splits = {}
    n = len(comps)
    cuts = [int(n * 0.15), int(n * 0.35), int(n * 0.50), int(n * 0.65)]
    for i, ks in enumerate(comps):
        split = (
            "dev"
            if i < cuts[0]
            else "proposal"
            if i < cuts[1]
            else "audit0"
            if i < cuts[2]
            else "audit1"
            if i < cuts[3]
            else "test"
        )
        for k in ks:
            splits[k] = split
    public = []
    oracle = {}
    for i, x in enumerate(raw):
        k = folder_key(x["file_path"])
        gid = digest(k)[:16]
        sid = f"tab{i:04d}"
        public.append(
            {
                "id": sid,
                "group": gid,
                "cluster": digest(sorted(components[find(k)]))[:16],
                "split": splits[k],
                "table_folder": k,
                "task": x["task"],
                "questions": [y.get("info_item") for y in x["design"]["checkout_list"]],
                "difficulty": x.get("difficulty"),
                "supported": bool(catalogs[k])
                and all(y.get("info_item") for y in x["design"]["checkout_list"]),
                "malformed_turns": [
                    j
                    for j, y in enumerate(x["design"]["checkout_list"])
                    if not y.get("info_item")
                ],
            }
        )
        oracle[sid] = [
            {
                "score_points": y["score_points"],
                "related_tables": y.get("related_tables", []),
            }
            for y in x["design"]["checkout_list"]
        ]
    (out / "public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2))
    (out / "oracle.json").write_text(json.dumps(oracle, ensure_ascii=False, indent=2))
    manifest = {
        "seed": seed,
        "samples_sha256": hashlib.sha256(
            (repo / "dataset/samples.json").read_bytes()
        ).hexdigest(),
        "dialogs": len(public),
        "turns": sum(len(t["questions"]) for t in public),
        "directory_groups": len(groups),
        "deduplicated_clusters": n,
        "split_dialogs": {
            s: sum(t["split"] == s for t in public)
            for s in ["dev", "proposal", "audit0", "audit1", "test"]
        },
        "split_clusters": {
            s: len({t["cluster"] for t in public if t["split"] == s})
            for s in ["dev", "proposal", "audit0", "audit1", "test"]
        },
        "supported_dialogs": sum(t["supported"] for t in public),
        "stores": records,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return manifest


def context(store: Path, question: str, max_chars=16000):
    catalog = json.loads((store / "catalog.json").read_text())
    conn = sqlite3.connect(f"file:{store / 'tables.sqlite'}?mode=ro", uri=True)
    terms = set(re.findall(r"[A-Za-z0-9_.%]+|[\u4e00-\u9fff]{2,}", question.lower()))
    terms |= {
        t[i : i + 2] for t in list(terms) for i in range(len(t) - 1) if len(t) > 2
    }
    # Every table schema is listed before bounded row retrieval, including low-ranked tables.
    schemas = [
        f"{c['name']}: file={c['file']}; sheet={c['sheet']}; rows={c['rows']}; columns=_row,c1..c{len(c['columns'])}"
        for c in catalog
    ]
    header = (
        "Complete source catalog (raw cell coordinates; omitted cells are blank):\n"
        + "\n".join(schemas)
        + "\n"
    )
    remaining = max(0, max_chars - len(header))
    quota = remaining // max(len(catalog), 1)
    chunks = []
    for c in catalog:
        rows = conn.execute(f'SELECT * FROM "{c["name"]}"').fetchall()

        def score(row):
            text = " ".join(str(v) for v in row[1:] if v not in (None, "")).lower()
            return sum(min(len(t), 6) for t in terms if t in text)

        nonblank = [
            i for i, r in enumerate(rows) if any(v not in (None, "") for v in r[1:])
        ]
        priority = list(nonblank[:6])
        priority += sorted(nonblank, key=lambda i: (-score(rows[i]), i))
        chosen = {}
        size = len(c["name"]) + 10
        for i in dict.fromkeys(priority):
            row = rows[i]
            # Explicit bindings prevent accidental one-column offsets and transposed-year errors.
            text = (
                "row "
                + str(row[0])
                + ": "
                + ", ".join(
                    f"c{j}=" + json.dumps(v, ensure_ascii=False, default=str)
                    for j, v in enumerate(row[1:], 1)
                    if v not in (None, "")
                )
            )
            if size + len(text) + 1 <= quota:
                chosen[i] = text
                size += len(text) + 1
        chunks.append(
            c["name"] + " cells:\n" + "\n".join(chosen[i] for i in sorted(chosen))
        )
    conn.close()
    return header + "\n".join(chunks)
