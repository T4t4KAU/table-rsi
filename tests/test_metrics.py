import copy
from decimal import Decimal

import pytest

from tablersi.data import build_store
from tablersi.metrics import compile_metric, evaluate_metric, number
from tablersi.observations import evidence_text
from tablersi.sql import execute


def series(periods=(2, 3, 4), cycles=("2019", "2019", "2019")):
    return {
        "need_id": "r1",
        "metric": "cumulative volume",
        "unit": "units",
        "operation": "series",
        "value_kind": "cumulative",
        "points": [
            {
                "period": p,
                "cycle": cycle,
                "label": str(p),
                "cell": {"table": "t0", "row": 2, "column": f"c{i + 1}"},
            }
            for i, (p, cycle) in enumerate(zip(periods, cycles))
        ],
    }


def source(values):
    return {"ok": True, "rows": list(enumerate(values)), "truncated": False}


def test_cumulative_regression_from_actual_raw_cells(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "a.csv").write_text("Q2,Q3,Q4\n3129.9,5587.7,7699.1\n")
    store = tmp_path / "store"
    build_store(folder, store)
    spec = series()
    original = copy.deepcopy(spec)
    observed = execute(store / "tables.sqlite", compile_metric(spec))
    report = evaluate_metric(spec, observed)
    rows = report["rows"]
    assert rows[0][4:] == [None, None, None]
    assert rows[1][4] == 2457.8
    assert rows[2][4] == 2111.4
    assert rows[1][3] == pytest.approx((5587.7 - 3129.9) / 3129.9)
    assert rows[1][6] is None
    assert rows[2][6] == pytest.approx((2111.4 - 2457.8) / 2457.8)
    assert sum(r[4] for r in rows[1:]) == pytest.approx(7699.1 - 3129.9)
    assert spec == original
    assert report["source_cells"][0]["raw"] == "3129.9"


def test_metric_meaning_and_unknown_predecessor_survive_role_projection():
    import json

    metric = {"id": "e1", **evaluate_metric(series(), source([10, 20, 35]))}
    original = copy.deepcopy(metric)
    rendered = json.loads(evidence_text([metric], 4000))[0]
    assert rendered["kind"] == "checked_metric"
    assert rendered["metric_contract"]["value_kind"] == "cumulative"
    assert rendered["metric_contract"]["metric"] == "cumulative volume"
    assert rendered["rows"][0][4] is None
    assert rendered["source_cells"][0]["cell"]["row"] == 2
    assert rendered["interval"]["level_change"] == 25
    assert metric == original
    metric["source_cells"][0]["raw"] = "x" * 20000
    compact = json.loads(evidence_text([metric], 2000))[0]
    assert "source_cells" in compact["prompt_fields_omitted"]
    assert compact["metric_contract"]["metric"] == "cumulative volume"
    assert compact["rows"] == original["rows"]


def test_missing_period_and_cycle_reset_do_not_create_single_period_values():
    report = evaluate_metric(series((1, 3, 4)), source([10, 35, 50]))
    assert [r[4] for r in report["rows"]] == [None, None, 15]
    assert all(r[6] is None for r in report["rows"])
    reset = evaluate_metric(
        series(cycles=("2019", "2020", "2020")), source([50, 8, 20])
    )
    assert reset["rows"][1][2:5] == [None, None, None]
    assert reset["rows"][2][4] == 12
    assert reset["interval"]["level_change"] is None


def test_missing_middle_value_is_not_zero_or_bridged():
    report = evaluate_metric(series(), source([10, "not reported", 40]))
    assert [r[4] for r in report["rows"]] == [None, None, None]
    assert report["rows"][1][1] is None
    assert report["interval"]["level_change"] == 30
    assert report["issues"][0]["point"] == 1


@pytest.mark.parametrize(
    "raw", ["12,34", "7 million", "NaN", "Infinity", True, None, "", "--", "1e9999"]
)
def test_invalid_source_never_casts_to_zero(raw):
    with pytest.raises(ValueError):
        number(raw)


def test_units_percentage_and_parenthesized_negative_are_explicit():
    assert number("1,234.50") == Decimal("1234.50")
    assert number("12.5%") == Decimal("0.125")
    assert number("(1,200)") == -1200
    assert number("−1.25") == Decimal("-1.25")
    spec = series()
    result = evaluate_metric(spec, source([0, 10, -2]))
    assert result["rows"][1][3] is None
    assert result["rows"][2][3] == -1.2
    percentages = evaluate_metric(spec, source(["10%", "20%", "30%"]))
    assert percentages["normalized_value_unit"] == "fraction"
    assert percentages["rows"][0][1] == 0.1
    with pytest.raises(ValueError, match="Mixed"):
        evaluate_metric(spec, source(["10%", 20, 30]))


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("ratio", 2),
        ("difference", -5),
        ("relative_change", -0.5),
        ("sum", 15),
        ("mean", 7.5),
    ],
)
def test_binary_and_aggregate_operations(operation, expected):
    spec = series((1, 2), ("a", "a"))
    spec["operation"] = operation
    assert evaluate_metric(spec, source([10, 5]))["rows"] == [[expected]]
    assert evaluate_metric(spec, source([10, None]))["rows"] == [[None]]


def test_rejects_injection_duplicates_unsorted_periods_and_incomplete_sources():
    spec = series()
    spec["points"][0]["cell"]["table"] = "t0; DROP TABLE t0"
    with pytest.raises(ValueError, match="coordinate"):
        compile_metric(spec)
    spec = series()
    spec["points"][1]["cell"] = spec["points"][0]["cell"]
    with pytest.raises(ValueError, match="Duplicate"):
        compile_metric(spec)
    with pytest.raises(ValueError, match="increasing"):
        compile_metric(series((2, 1, 3)))
    with pytest.raises(ValueError, match="Complete"):
        evaluate_metric(series(), {**source([1, 2, 3]), "truncated": True})
    with pytest.raises(ValueError, match="coordinates"):
        evaluate_metric(series(), source([1, 2]))
