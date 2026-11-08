import json

from tablersi.evaluate import Judge


def test_judge_repairs_denominator_without_losing_usage(tmp_path):
    folder = tmp_path / "src/prompts"
    folder.mkdir(parents=True)
    (folder / "AgentEvalPrompt.py").write_text(
        "SCORE_POINTS_ACC_JUDGE = 'Question: {query}; Reference: {answer}; Response: {model_answer}'"
    )

    class Fake:
        count = 0

        def chat(self, *a, **kw):
            self.count += 1
            return {
                "content": json.dumps(
                    {
                        "total_metrics": 4 if self.count == 1 else 1,
                        "covered_metrics": 1,
                        "reasoning": "test",
                    }
                ),
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                "elapsed_s": 0.2,
                "model": "fake",
            }

    score = Judge(Fake(), tmp_path).score("q", "a", ["one composite point"])
    assert score["coverage_ratio"] == 1
    assert score["usage"] == {"prompt_tokens": 20, "completion_tokens": 6}
    assert len(score["attempts"]) == 2
