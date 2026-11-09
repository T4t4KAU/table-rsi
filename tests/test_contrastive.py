import copy

import pytest

from tablersi.contrastive import compare_training


def fixture():
    task = {"id": "t1", "cluster": "teaching", "questions": ["q1", "q2"]}
    episode = {
        **task,
        "turns": [
            {"question": q, "answer": "a", "trace": [], "score": {"coverage_ratio": s}}
            for q, s in zip(task["questions"], [1, 0])
        ],
    }
    return task, episode


def test_contrast_keeps_preservation_and_repair_examples():
    task, old = fixture()
    new = copy.deepcopy(old)
    new["turns"][1]["score"]["coverage_ratio"] = 1
    result = compare_training(task, old, new)
    assert len(result["experiences"]) == 2
    assert result["transitions"]["counts"]["full_to_full"] == 1
    assert result["transitions"]["counts"]["not_full_to_full"] == 1


@pytest.mark.parametrize("field,value", [("id", "heldout"), ("cluster", "audit")])
def test_cannot_substitute_another_task_or_cluster(field, value):
    task, old = fixture()
    new = copy.deepcopy(old)
    new[field] = value
    with pytest.raises(ValueError, match="identity"):
        compare_training(task, old, new)


def test_question_and_score_failures_abort_teaching_comparison():
    task, old = fixture()
    new = copy.deepcopy(old)
    new["turns"][0]["question"] = "wrong question"
    with pytest.raises(ValueError, match="question"):
        compare_training(task, old, new)
    new = copy.deepcopy(old)
    new["turns"][0]["score"] = {"error": "judge failure"}
    with pytest.raises(ValueError, match="invalid coverage"):
        compare_training(task, old, new)
