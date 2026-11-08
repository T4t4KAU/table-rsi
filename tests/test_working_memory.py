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
