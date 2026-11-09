"""Deterministic arithmetic over raw cells, with explicit model-proposed semantics.

The executor checks coordinates, numeric parsing and algebra. It cannot prove
that the model chose the right metric, unit, entity or calendar. No supplied
numeric constants, imputed initial cumulative values or arbitrary code run here.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation


def number(value):
    """Parse unambiguous numbers; never emulate SQLite's invalid-text-to-zero cast."""
    if value is None or isinstance(value, bool):
        raise ValueError("Missing or nonnumeric source value")
    text = str(value).strip().replace("−", "-")
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    if "," in text:
        if not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", text):
            raise ValueError("Ambiguous comma grouping")
        text = text.replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        raise ValueError("Source contains nonnumeric text or an undeclared unit")
    try:
        value = Decimal(text) / (100 if percent else 1)
    except InvalidOperation as exc:
        raise ValueError("Invalid number") from exc
    if not value.is_finite() or not math.isfinite(float(value)):
        raise ValueError("Nonfinite number")
    return value


def compile_metric(spec):
    """Return one bounded raw-cell SELECT; metadata stays outside executable SQL."""
    if not isinstance(spec, dict):
        raise ValueError("Metric must be an object")
    op = spec.get("operation")
    if op not in {"series", "ratio", "difference", "relative_change", "sum", "mean"}:
        raise ValueError("Unsupported metric operation")
    for field in ("need_id", "metric", "unit"):
        if not isinstance(spec.get(field), str) or not 0 < len(spec[field]) <= 300:
            raise ValueError(f"Nonempty {field} required")
    points = spec.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= 24:
        raise ValueError("One to 24 raw-cell points required")
    if op in {"ratio", "difference", "relative_change"} and len(points) != 2:
        raise ValueError("Binary operation requires exactly two points")
    if op == "series":
        if spec.get("value_kind") not in {"level", "cumulative"}:
            raise ValueError("Series requires level or cumulative value_kind")
        periods = [p.get("period") if isinstance(p, dict) else None for p in points]
        if any(type(p) is not int for p in periods) or any(
            a >= b for a, b in zip(periods, periods[1:])
        ):
            raise ValueError("Periods must be strictly increasing integers")
        if spec["value_kind"] == "cumulative" and any(
            not isinstance(p.get("cycle"), str) or not p["cycle"].strip()
            for p in points
        ):
            raise ValueError("Cumulative points require explicit accumulation cycles")
    selects = []
    seen = set()
    for index, point in enumerate(points):
        if not isinstance(point, dict) or not isinstance(point.get("cell"), dict):
            raise ValueError("Each point requires a raw cell")
        cell = point["cell"]
        table, column, row = cell.get("table"), cell.get("column"), cell.get("row")
        if (
            not isinstance(table, str)
            or not re.fullmatch(r"t\d+", table)
            or not isinstance(column, str)
            or not re.fullmatch(r"c[1-9]\d*", column)
            or type(row) is not int
            or row < 1
        ):
            raise ValueError("Invalid raw-cell coordinate")
        key = (table, row, column)
        if key in seen:
            raise ValueError("Duplicate source cell")
        seen.add(key)
        selects.append(
            f"SELECT {index} AS point_index, "
            f'(SELECT "{column}" FROM "{table}" WHERE _row={row}) AS raw_value'
        )
    return " UNION ALL ".join(selects)


def _float(value):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite arithmetic result")
    return result


def _change(current, previous):
    if current is None or previous is None:
        return None, None
    difference = current - previous
    # Relative growth on zero/negative bases is undefined in this contract.
    return difference, difference / previous if previous > 0 else None


def evaluate_metric(spec, source):
    """Compute only from the returned raw values; keep unknowns and lineage."""
    compile_metric(spec)
    if source.get("ok") is not True or source.get("truncated"):
        raise ValueError("Complete successful source execution required")
    points = spec["points"]
    rows = source.get("rows", [])
    if len(rows) != len(points) or any(
        len(row) != 2 or row[0] != i for i, row in enumerate(rows)
    ):
        raise ValueError("Raw result does not match the requested coordinates")
    values, issues = [], []
    for i, row in enumerate(rows):
        try:
            values.append(number(row[1]))
        except ValueError as exc:
            values.append(None)
            issues.append({"point": i, "issue": str(exc)})
    percent_modes = {
        str(row[1]).strip().endswith("%")
        for row, value in zip(rows, values)
        if value is not None
    }
    if len(percent_modes) > 1:
        raise ValueError(
            "Mixed explicit percentages and unmarked numbers require unit normalization"
        )
    op = spec["operation"]
    result = {
        "kind": "checked_metric",
        "metric_contract": {
            k: spec[k]
            for k in ("need_id", "metric", "unit", "operation", "value_kind")
            if k in spec
        },
        "source_cells": [
            {**point, "raw": rows[i][1]} for i, point in enumerate(points)
        ],
        "issues": issues,
        "normalized_value_unit": "fraction"
        if percent_modes == {True}
        else spec["unit"],
        "verification_scope": "Raw-cell arithmetic only. Binding, unit, calendar, and relevance remain model proposals; rates are fractions, not percentages.",
    }
    if op == "series":
        output, singles = [], []
        for i, (point, value) in enumerate(zip(points, values)):
            previous = values[i - 1] if i else None
            same_cycle = i > 0 and point.get("cycle") == points[i - 1].get("cycle")
            cumulative = spec["value_kind"] == "cumulative"
            comparable = not cumulative or same_cycle
            delta, rate = _change(value, previous if comparable else None)
            adjacent = i > 0 and point["period"] == points[i - 1]["period"] + 1
            single = delta if cumulative and same_cycle and adjacent else None
            singles.append(single)
            single_delta, single_rate = _change(
                single, singles[i - 1] if i and adjacent else None
            )
            output.append(
                [
                    point.get("label", str(point["period"])),
                    _float(value),
                    _float(delta),
                    _float(rate),
                    _float(single),
                    _float(single_delta),
                    _float(single_rate),
                ]
            )
        first, last = values[0], values[-1]
        comparable = spec["value_kind"] != "cumulative" or all(
            p["cycle"] == points[0]["cycle"] for p in points
        )
        delta, rate = _change(last, first if comparable else None)
        result.update(
            columns=[
                "label",
                "source_level",
                "level_change",
                "level_relative_change",
                "single_period_value",
                "single_period_change",
                "single_period_relative_change",
            ],
            rows=output,
            interval={
                "level_change": _float(delta),
                "level_relative_change": _float(rate),
            },
            preconditions="Single-period values require adjacent observed cumulative periods in the same declared cycle. First displayed cumulative value is never imputed as a single-period value. Missing/invalid operands and nonpositive growth bases produce null.",
        )
    else:
        value = None
        if all(v is not None for v in values):
            if op == "ratio":
                value = values[0] / values[1] if values[1] != 0 else None
            elif op == "difference":
                value = values[1] - values[0]
            elif op == "relative_change":
                _, value = _change(values[1], values[0])
            else:
                value = sum(values)
                if op == "mean":
                    value /= len(values)
        result.update(columns=[op], rows=[[_float(value)]])
    return {**result, "ok": True, "truncated": False}
