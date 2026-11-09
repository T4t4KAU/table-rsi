"""Read-only table observations with complete catalog and focused cell retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from pathlib import Path


def terms(text):
    words = set(re.findall(r"[A-Za-z0-9_.%]+|[\u4e00-\u9fff]{2,}", text.lower()))
    return words | {word[i : i + 2] for word in words for i in range(len(word) - 1)}


def row_text(row):
    return f"row {row[0]}: " + ", ".join(
        f"c{i}=" + json.dumps(value, ensure_ascii=False, default=str)
        for i, value in enumerate(row[1:], 1)
        if value not in (None, "")
    )


class Observations:
    def __init__(self, store):
        self.store = Path(store)
        self.catalog = json.loads((self.store / "catalog.json").read_text())
        self.names = {entry["name"] for entry in self.catalog}

    def catalog_text(self):
        return "\n".join(
            f"{c['name']}: file={c['file']}; sheet={c['sheet']}; "
            f"rows={c['rows']}; columns=_row,c1..c{len(c['columns'])}"
            for c in self.catalog
        )

    def render(self, question, selected=None, max_chars=24000):
        """Keep all catalog entries. Prefer selected tables without rewriting cells."""
        selected = [name for name in (selected or []) if name in self.names][:6]
        header = "完整表目录（空白单元格不等于零）：\n" + self.catalog_text() + "\n"
        if len(header) > max_chars:
            raise ValueError(
                "Catalog exceeds observation budget; cannot silently omit it"
            )
        candidates = [c for c in self.catalog if not selected or c["name"] in selected]
        words = terms(question)
        conn = sqlite3.connect(f"file:{self.store / 'tables.sqlite'}?mode=ro", uri=True)
        chunks = []
        quota = (max_chars - len(header)) // max(1, len(candidates))
        try:
            for entry in candidates:
                # Names originate in the catalog, but quote them as identifiers anyway.
                quoted = entry["name"].replace('"', '""')
                rows = conn.execute(f'SELECT * FROM "{quoted}"').fetchall()
                nonblank = [
                    i
                    for i, row in enumerate(rows)
                    if any(v not in (None, "") for v in row[1:])
                ]
                scores = {
                    i: sum(
                        min(len(w), 5) for w in words if w in row_text(rows[i]).lower()
                    )
                    for i in nonblank
                }
                ranked = sorted(nonblank, key=lambda i: (-scores[i], i))
                priority = nonblank[:6]
                for i in ranked:
                    priority.extend(j for j in (i - 1, i, i + 1) if j in scores)
                kept = {}
                size = len(entry["name"]) + 50
                for i in dict.fromkeys(priority):
                    text = row_text(rows[i])
                    if size + len(text) + 1 <= quota:
                        kept[i] = text
                        size += len(text) + 1
                chunks.append(
                    entry["name"]
                    + f" cells (shown {len(kept)}/{len(nonblank)} nonblank rows):\n"
                    + "\n".join(kept[i] for i in sorted(kept))
                )
        finally:
            conn.close()
        return header + "\n".join(chunks)


def _trim_evidence_rows(records, max_chars):
    """Keep the established row trimming order; signal metadata-only overflow."""
    while len(json.dumps(records, ensure_ascii=False, default=str)) > max_chars:
        longest = max(
            records, key=lambda r: len(json.dumps(r.get("rows", []))), default=None
        )
        if longest is None or not longest.get("rows"):
            return False
        longest["rows"].pop()
        longest["prompt_rows_omitted"] = longest.get("prompt_rows_omitted", 0) + 1
    return True


def _compact_evidence_metadata(records, max_chars):
    """Omit whole optional fields explicitly, reserving space for result rows."""
    while True:
        metadata = [{**record, "rows": []} for record in records]
        size = len(json.dumps(metadata, ensure_ascii=False, default=str))
        if size <= max_chars // 2:
            return
        candidates = [
            (len(json.dumps(record[key], ensure_ascii=False, default=str)), i, key)
            for i, record in enumerate(records)
            for key in (
                "sql",
                "purpose",
                "error",
                "columns",
                "source_cells",
                "verification_scope",
                "preconditions",
                "issues",
                "calculation_notes",
            )
            if key in record
        ]
        if not candidates:
            if size > max_chars:
                raise ValueError("Evidence budget too small for identities/status")
            return
        _, index, key = max(candidates)
        record = records[index]
        del record[key]
        omitted = record.setdefault("prompt_fields_omitted", [])
        omitted.append(key)
        if key == "columns":
            # Never show positional values after dropping their column labels.
            record["rows"] = []
            omitted.append("rows")


def evidence_text(evidence, max_chars=18000):
    """Bound prompt records without changing the authoritative execution trace."""
    output = []
    for item in evidence:
        record = {
            k: v
            for k, v in item.items()
            if k
            in {"id", "sql", "purpose", "ok", "error", "columns", "rows", "truncated"}
        }
        record["rows"] = list(record.get("rows", []))
        if item.get("kind") == "checked_metric":
            # Do not erase metric identity or the cumulative/period distinction
            # when projecting a typed computation into the next role's prompt.
            record.update(
                {
                    k: item[k]
                    for k in (
                        "kind",
                        "metric_contract",
                        "source_cells",
                        "normalized_value_unit",
                        "interval",
                        "preconditions",
                        "issues",
                        "verification_scope",
                        "calculation_notes",
                    )
                    if k in item
                }
            )
        output.append(record)
    original = deepcopy(output)
    if not _trim_evidence_rows(output, max_chars):
        # The old path has removed every row. Restore results before dropping
        # oversized metadata, so long SQL does not erase the useful evidence.
        output = original
        _compact_evidence_metadata(output, max_chars)
        if not _trim_evidence_rows(output, max_chars):
            raise ValueError("Evidence budget too small for identities/status")
    return json.dumps(output, ensure_ascii=False, default=str)
