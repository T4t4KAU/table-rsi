from tablersi.answer_patch import apply_answer_patch, is_code_only


def edit(old, new, support=None):
    return {
        "old": old,
        "new": new,
        "reason": "Correct arithmetic",
        "evidence_ids": support or ["e2"],
    }


def test_edit_preserves_unrelated_facts_and_adds_supported_omission():
    draft = "Revenue is 80. Insurance is the main contributor. Cash declined."
    updated, status = apply_answer_patch(
        draft,
        {
            "edits": [edit("Revenue is 80.", "Revenue is 8.")],
            "additions": [
                {
                    "text": "Debt increased.",
                    "reason": "Missing requested trend",
                    "evidence_ids": ["raw_cells"],
                }
            ],
        },
        ["e2"],
    )
    assert (
        updated
        == "Revenue is 8. Insurance is the main contributor. Cash declined.\n\nDebt increased."
    )
    assert status["applied"]


def test_ambiguous_or_missing_anchor_rejects_entire_patch():
    draft = "Revenue is 80. Total: 80."
    for old in ["80", "unknown", ""]:
        updated, status = apply_answer_patch(draft, {"edits": [edit(old, "8")]}, ["e2"])
        assert updated == draft
        assert not status["applied"]


def test_invalid_evidence_or_overlapping_edit_is_atomic():
    draft = "Revenue is 80. Cash declined."
    for edits in [
        [
            edit("Revenue is 80.", "Revenue is 8."),
            edit("Cash declined.", "Cash increased.", ["e99"]),
        ],
        [edit("Revenue is 80.", "Revenue is 8."), edit("80", "8")],
    ]:
        updated, status = apply_answer_patch(draft, {"edits": edits}, ["e2"])
        assert updated == draft
        assert not status["applied"]


def test_sql_only_replacement_is_not_a_final_analysis():
    draft = "The total is 8."
    sql = "```sql\nSELECT SUM(value) FROM records\n```"
    assert is_code_only(sql)
    assert not is_code_only("The query SELECT SUM(value) gives 8.")
    updated, status = apply_answer_patch(draft, {"edits": [edit(draft, sql)]}, ["e2"])
    assert updated == draft
    assert not status["applied"]
