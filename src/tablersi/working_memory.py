"""Task-local, typed corrections linking requirements to observed evidence.

These are model proposals, not correctness labels. Retired evidence remains in
the audit trace; only its use as the current answer basis is withdrawn.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    contract: dict
    allow_requirement_revisions: bool = False
    revisions: list[dict] = field(default_factory=list)
    contested: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.contract = copy.deepcopy(self.contract)
        requirements = self.contract.get("requirements", [])
        if not isinstance(requirements, list):
            self.contract["requirements"] = []

    def apply(self, review, evidence):
        known = {e["id"] for e in evidence}
        executed = {e["id"] for e in evidence if e.get("ok") is True}
        requirements = {
            r["id"]: r
            for r in self.contract.get("requirements", [])
            if isinstance(r, dict) and isinstance(r.get("id"), str)
        }
        events = []
        withdrawals = review.get("withdraw_evidence", [])
        if isinstance(withdrawals, list):
            for item in withdrawals[:12]:
                if not isinstance(item, dict):
                    continue
                ident = item.get("id")
                reason = item.get("reason")
                if (
                    isinstance(ident, str)
                    and ident in known
                    and isinstance(reason, str)
                    and reason.strip()
                ):
                    self.contested[ident] = reason[:600]
                    events.append(
                        {"action": "contest", "id": ident, "reason": reason[:600]}
                    )
        corrections = review.get("binding_corrections", [])
        if isinstance(corrections, list):
            for item in corrections[:12]:
                if not isinstance(item, dict):
                    continue
                need = item.get("need_id")
                binding = item.get("binding")
                support = item.get("evidence_ids")
                if (
                    not isinstance(need, str)
                    or need not in requirements
                    or not isinstance(binding, str)
                    or not binding.strip()
                    or len(binding) > 1200
                    or not isinstance(support, list)
                    or not support
                    or any(
                        not isinstance(i, str)
                        or i not in executed
                        or i in self.contested
                        for i in support
                    )
                ):
                    continue
                old = requirements[need].get("binding")
                requirements[need]["binding"] = binding
                requirements[need]["binding_evidence_ids"] = support
                events.append(
                    {
                        "action": "binding_correction",
                        "need_id": need,
                        "old_binding": old,
                        "new_binding": binding,
                        "evidence_ids": support,
                    }
                )
        self.revisions.extend(events)
        if self.allow_requirement_revisions:
            events.extend(self._revise_requirements(review, evidence, requirements))
        return events

    def _revise_requirements(self, review, evidence, requirements):
        """Revise a semantic proposal, never permissions or executable code."""
        revisions = review.get("requirement_revisions", [])
        if not isinstance(revisions, list):
            return []
        successful = {
            e["id"] for e in evidence if e.get("ok") and e["id"] not in self.contested
        }
        fields = {
            "need",
            "kind",
            "binding",
            "formula",
            "evidence_needed",
            "entity",
            "metric",
            "time_scope",
        }
        events = []
        for item in revisions[:12]:
            if not isinstance(item, dict):
                continue
            need, updates = item.get("need_id"), item.get("updates")
            support, reason = item.get("evidence_ids"), item.get("reason")
            if (
                not isinstance(need, str)
                or need not in requirements
                or not isinstance(updates, dict)
                or not updates
                or not set(updates) <= fields
                or not all(
                    isinstance(v, str) and 0 < len(v) <= 1200 for v in updates.values()
                )
                or not isinstance(reason, str)
                or not reason.strip()
                or not isinstance(support, list)
                or not support
                or any(not isinstance(i, str) or i not in successful for i in support)
            ):
                continue
            previous = {key: requirements[need].get(key) for key in updates}
            requirements[need].update(updates)
            event = {
                "action": "requirement_revision",
                "need_id": need,
                "old_fields": previous,
                "new_fields": copy.deepcopy(updates),
                "evidence_ids": list(support),
                "reason": reason[:600],
            }
            events.append(event)
        self.revisions.extend(events)
        return events

    def project(self):
        return {
            "current_contract": self.contract,
            "correction_history": self.revisions[-12:],
            "contested_evidence": self.contested,
            "epistemic_status": "Model-proposed semantic corrections; not reference labels",
        }

    def active(self, evidence):
        return [e for e in evidence if e["id"] not in self.contested]
