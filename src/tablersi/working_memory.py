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
    revisions: list[dict] = field(default_factory=list)
    contested: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.contract = copy.deepcopy(self.contract)
        requirements = self.contract.get("requirements", [])
        if not isinstance(requirements, list):
            self.contract["requirements"] = []

    def apply(self, review, evidence):
        known = {e["id"] for e in evidence}
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
                        not isinstance(i, str) or i not in known or i in self.contested
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
