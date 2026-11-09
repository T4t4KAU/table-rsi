"""Build same-task teaching comparisons without mixing partition identities."""

from .paired_outcomes import paired_outcomes


def compare_training(task, baseline, candidate):
    """Caller supplies only designated teaching tasks; keep successes and failures."""
    for episode in (baseline, candidate):
        if episode["id"] != task["id"] or episode["cluster"] != task["cluster"]:
            raise ValueError("Teaching episode identity mismatch")
        if [t["question"] for t in episode["turns"]] != task["questions"]:
            raise ValueError("Teaching question sequence mismatch")
    transitions = paired_outcomes([baseline], [candidate])
    experiences = []
    for index, (left, right) in enumerate(zip(baseline["turns"], candidate["turns"])):
        experiences.append(
            {
                "question": task["questions"][index],
                "outcome": {
                    "baseline_full_coverage": left["score"]["coverage_ratio"] == 1,
                    "candidate_full_coverage": right["score"]["coverage_ratio"] == 1,
                },
                "baseline": _experience(left),
                "candidate": _experience(right),
            }
        )
    return {"transitions": transitions, "experiences": experiences}


def _experience(turn):
    return {
        "answer": turn["answer"],
        "score": {
            k: v for k, v in turn["score"].items() if k not in {"attempts", "usage"}
        },
        "trace": [
            {
                k: v
                for k, v in step.items()
                if k in {"role", "response", "execution", "events", "error"}
            }
            for step in turn["trace"]
        ],
    }
