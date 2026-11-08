import copy

import pytest

from tablersi.paired_outcomes import paired_outcomes


def episodes(scores):
    return [
        {
            "id": "tab0001",
            "cluster": "group",
            "turns": [
                {"question": f"q{i}", "score": {"coverage_ratio": score}}
                for i, score in enumerate(scores)
            ],
        }
    ]


def test_partial_gains_do_not_hide_damage_to_full_answers():
    report = paired_outcomes(episodes([1, 1, 0, 0.25]), episodes([1, 0.5, 1, 0.75]))
    assert report["counts"] == {
        "full_to_full": 1,
        "full_to_not_full": 1,
        "not_full_to_full": 1,
        "not_full_to_not_full": 1,
    }
    assert report["net_full_point_gain"] == 0
    assert report["mean_coverage_gain"] == 0.25
    assert report["damage_given_full"] == report["repair_given_not_full"] == 0.5


@pytest.mark.parametrize("scores", [[0, 0.5], [1, 1]])
def test_absent_conditional_denominator_is_not_zero_success(scores):
    report = paired_outcomes(episodes(scores), episodes(scores))
    field = "damage_given_full" if scores[0] == 0 else "repair_given_not_full"
    assert report[field] is None


def test_reject_unpaired_records_and_different_questions():
    with pytest.raises(ValueError, match="exactly paired"):
        paired_outcomes(episodes([1, 0]), episodes([1]))
    right = episodes([1])
    right[0]["turns"][0]["question"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        paired_outcomes(episodes([1]), right)
    left = episodes([1])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_outcomes(left + copy.deepcopy(left), left)


@pytest.mark.parametrize("value", [None, True, float("nan"), -1, 1.1])
def test_judge_failures_and_invalid_scores_cannot_be_silently_counted(value):
    with pytest.raises(ValueError, match="invalid coverage"):
        paired_outcomes(episodes([1]), episodes([value]))
