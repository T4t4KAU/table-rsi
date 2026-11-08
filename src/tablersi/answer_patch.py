"""Apply evidence-linked literal answer edits while preserving untouched claims.

Structural validity does not establish semantic correctness. The model proposes
corrections; this module only prevents accidental whole-answer replacement,
ambiguous anchors, and references to unavailable evidence.
"""

from __future__ import annotations

import re


def is_code_only(text):
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:sql|json)?\s*([\s\S]*?)\s*```", stripped, re.I)
    body = fenced.group(1).strip() if fenced else stripped
    return bool(re.match(r"^(?:SELECT\b|WITH\b|\{\s*\"(?:queries|sql)\")", body, re.I))


def apply_answer_patch(draft, proposal, evidence_ids):
    allowed = set(evidence_ids) | {"raw_cells"}
    edits, additions = proposal.get("edits", []), proposal.get("additions", [])
    if not isinstance(edits, list) or not isinstance(additions, list):
        return draft, {"applied": False, "reason": "patch fields must be lists"}
    if len(edits) + len(additions) > 16:
        return draft, {"applied": False, "reason": "too many edits"}
    spans, suffixes = [], []
    for item, addition in [(e, False) for e in edits] + [(e, True) for e in additions]:
        if not isinstance(item, dict):
            return draft, {"applied": False, "reason": "edit must be an object"}
        support, reason = item.get("evidence_ids"), item.get("reason")
        replacement = item.get("text" if addition else "new")
        if (
            not isinstance(support, list)
            or not support
            or any(not isinstance(i, str) or i not in allowed for i in support)
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(replacement, str)
            or len(replacement) > 12000
            or is_code_only(replacement)
        ):
            return draft, {
                "applied": False,
                "reason": "invalid text or evidence support",
            }
        if addition:
            if replacement.strip():
                suffixes.append(replacement.strip())
            continue
        original = item.get("old")
        if not isinstance(original, str) or not original or draft.count(original) != 1:
            return draft, {"applied": False, "reason": "anchor must match exactly once"}
        start = draft.index(original)
        end = start + len(original)
        if any(start < right and end > left for left, right, _ in spans):
            return draft, {"applied": False, "reason": "overlapping edits"}
        spans.append((start, end, replacement))
    result = draft
    for start, end, replacement in sorted(spans, reverse=True):
        result = result[:start] + replacement + result[end:]
    if suffixes:
        result += "\n\n" + "\n\n".join(suffixes)
    if not result.strip() or is_code_only(result):
        return draft, {"applied": False, "reason": "not a usable answer"}
    return result, {
        "applied": result != draft,
        "replacements": len(spans),
        "additions": len(suffixes),
        "semantic_status": "model-proposed; not oracle verified",
    }
