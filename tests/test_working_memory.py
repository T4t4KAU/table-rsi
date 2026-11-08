from tablersi.working_memory import WorkingMemory


def test_correction_replaces_binding_and_retires_wrong_entity_evidence():
    contract = {"requirements": [{"id": "r1", "binding": "wrong row"}]}
    memory = WorkingMemory(contract)
    evidence = [
        {"id": "e1", "rows": [["wrong entity", 42]]},
        {"id": "e2", "rows": [["right entity", 19]]},
    ]
    events = memory.apply(
        {
            "withdraw_evidence": [{"id": "e1", "reason": "different entity"}],
            "binding_corrections": [
                {"need_id": "r1", "binding": "correct row", "evidence_ids": ["e2"]}
            ],
        },
        evidence,
    )
    assert memory.contract["requirements"][0]["binding"] == "correct row"
    assert contract["requirements"][0]["binding"] == "wrong row"
    assert [e["id"] for e in memory.active(evidence)] == ["e2"]
    assert len(events) == 2
    assert len(evidence) == 2


def test_unknown_evidence_and_malformed_fields_cannot_rebind():
    memory = WorkingMemory({"requirements": [{"id": "r1", "binding": "original"}]})
    evidence = [{"id": "e1"}]
    events = memory.apply(
        {
            "withdraw_evidence": [{"id": ["e1"], "reason": "bad type"}],
            "binding_corrections": [
                {"need_id": "r1", "binding": "hallucinated", "evidence_ids": ["e99"]},
                {"need_id": ["r1"], "binding": "bad type", "evidence_ids": ["e1"]},
                {"need_id": "r1", "binding": "unbacked", "evidence_ids": []},
            ],
        },
        evidence,
    )
    assert not events
    assert memory.contract["requirements"][0]["binding"] == "original"


def test_semantic_revision_changes_formula_only_with_successful_evidence():
    original = {"requirements": [{"id": "r1", "formula": "wrong", "binding": "row"}]}
    review = {
        "requirement_revisions": [
            {
                "need_id": "r1",
                "updates": {
                    "formula": "(end / start - 1) * 100",
                    "metric": "cumulative level change",
                },
                "evidence_ids": ["e1"],
                "reason": "Raw header defines cumulative values, not period totals",
            }
        ]
    }
    memory = WorkingMemory(original, allow_requirement_revisions=True)
    assert not memory.apply(review, [{"id": "e1", "ok": False}])
    events = memory.apply(review, [{"id": "e1", "ok": True}])
    assert len(events) == 1
    assert memory.contract["requirements"][0]["formula"] == "(end / start - 1) * 100"
    assert memory.contract["requirements"][0]["binding"] == "row"
    assert original["requirements"][0]["formula"] == "wrong"
    assert memory.revisions == events
    legacy = WorkingMemory(original)
    assert not legacy.apply(review, [{"id": "e1", "ok": True}])
    assert legacy.contract == original


def test_semantic_revision_rejects_contested_evidence_and_protected_fields():
    memory = WorkingMemory(
        {"requirements": [{"id": "r1", "formula": "original"}]},
        allow_requirement_revisions=True,
    )
    evidence = [{"id": "e1", "ok": True}, {"id": "e2", "ok": True}]
    memory.apply(
        {"withdraw_evidence": [{"id": "e1", "reason": "wrong entity"}]}, evidence
    )
    for updates, support in [
        ({"formula": "bad"}, ["e1"]),
        ({"formula": "bad", "sql_permissions": "write"}, ["e2"]),
        ({"formula": []}, ["e2"]),
        ({"formula": "bad"}, []),
    ]:
        assert not memory.apply(
            {
                "requirement_revisions": [
                    {
                        "need_id": "r1",
                        "updates": updates,
                        "evidence_ids": support,
                        "reason": "test",
                    }
                ]
            },
            evidence,
        )
    assert memory.contract["requirements"][0]["formula"] == "original"
