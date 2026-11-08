import pytest

from tablersi.curriculum import teacher_groups


def fixture():
    tasks = [
        {"id": "t1", "cluster": "teach", "split": "predeploy_train"},
        {"id": "t2", "cluster": "teach", "split": "predeploy_train"},
        {"id": "s1", "cluster": "search", "split": "predeploy_train"},
        {"id": "a1", "cluster": "audit", "split": "predeploy_train"},
    ]
    protocol = {"partitions": {"teaching": ["t1"], "search": ["s1"], "audit": ["a1"]}}
    return tasks, protocol


def test_curriculum_expands_only_within_teacher_owned_groups():
    tasks, protocol = fixture()
    result = teacher_groups(tasks, protocol)
    assert list(result) == ["teach"]
    assert [task["id"] for task in result["teach"]] == ["t1", "t2"]


def test_curriculum_rejects_overlap_or_nontraining_dialogs():
    tasks, protocol = fixture()
    tasks[2]["cluster"] = "teach"
    with pytest.raises(ValueError, match="overlap"):
        teacher_groups(tasks, protocol)
    tasks, protocol = fixture()
    tasks[1]["split"] = "postdeploy_test"
    with pytest.raises(ValueError, match="Only predeployment"):
        teacher_groups(tasks, protocol)


def test_deployment_policy_cannot_expect_teacher_rubric():
    from tablersi.curriculum import validate_deployment_rules
    from tablersi.memory import Bundle

    with pytest.raises(ValueError, match="unavailable reference"):
        validate_deployment_rules(
            Bundle(role_guidance={"EvidenceVerifier": "对参考点逐条复算"})
        )
    bundle = Bundle(role_guidance={"EvidenceVerifier": "对实际查询结果逐条复算"})
    assert validate_deployment_rules(bundle) is bundle


def test_abstraction_retry_identifies_constant_without_weakening_guard():
    from tablersi.memory import Bundle, validate_teacher_abstraction

    bundle = Bundle(role_guidance={"QueryBuilder": "ratio * 100"})
    with pytest.raises(ValueError, match="'100'.*semantic variables"):
        validate_teacher_abstraction(bundle)
    bundle.role_guidance["QueryBuilder"] = "Format the ratio as a percentage"
    assert validate_teacher_abstraction(bundle) is bundle


def test_keyword_retry_identifies_overlong_phrase():
    from tablersi.memory import Skill

    skill = Skill(
        id="extrema",
        keywords=["multi-source extreme union sort"],
        role_views={"SemanticPlanner": "Find grouped extrema", "QueryBuilder": "Union"},
        applicability="Extrema across homogeneous tables",
    )
    with pytest.raises(
        ValueError, match="multi-source extreme union sort.*30 characters"
    ):
        skill.validate()
    skill.keywords = ["multi-source extrema"]
    skill.validate()


def test_keyword_compilation_changes_only_unavailable_retrieval_terms():
    from tablersi.curriculum import compile_teacher_keywords

    original = {
        "skills": [
            {
                "id": "s1",
                "keywords": ["参考点", "增长"],
                "role_views": {"EvidenceVerifier": "核对原始证据"},
            }
        ]
    }
    compiled, removed = compile_teacher_keywords(original)
    assert compiled["skills"][0]["keywords"] == ["增长"]
    assert original["skills"][0]["keywords"] == ["参考点", "增长"]
    assert compiled["skills"][0]["role_views"] == original["skills"][0]["role_views"]
    assert removed == [{"skill_id": "s1", "keyword": "参考点"}]


def test_teacher_preserves_empty_failure_and_recovers(tmp_path):
    from tablersi.curriculum import diagnose_with_retries

    class Teacher:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"content": "", "finish_reason": "length", "usage": {}}
            assert all(m["content"] for m in messages)
            return {
                "content": '{"transferable_skills": []}',
                "finish_reason": "stop",
                "usage": {},
            }

    diagnosis, response = diagnose_with_retries(Teacher(), "Teach", tmp_path, 24000)
    assert diagnosis == {"transferable_skills": []}
    assert response["finish_reason"] == "stop"
    assert (tmp_path / "0.json").exists()
    assert (tmp_path / "1.json").exists()


def test_student_feedback_rejects_nonowned_groups_and_changed_questions(tmp_path):
    import json

    from tablersi.curriculum import teaching_episodes

    groups = {
        "teach": [{"id": "tab0001", "cluster": "teach", "questions": ["Original?"]}]
    }
    file = tmp_path / "tab0001.json"
    episode = {
        "id": "tab0001",
        "cluster": "search",
        "turns": [{"question": "Original?"}],
    }
    file.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="outside teacher-owned"):
        teaching_episodes(tmp_path, groups)
    episode["cluster"] = "teach"
    episode["turns"][0]["question"] = "Changed?"
    file.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="do not match"):
        teaching_episodes(tmp_path, groups)
    episode["turns"][0]["question"] = "Original?"
    file.write_text(json.dumps(episode))
    assert teaching_episodes(tmp_path, groups) == {"tab0001": episode}
