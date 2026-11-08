"""Versioned cooperative policy bundles; provenance never enters solver prompts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROLES = ("SemanticPlanner", "QueryBuilder", "EvidenceVerifier", "AnswerComposer")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass
class Skill:
    id: str
    keywords: list[str]
    role_views: dict[str, str]
    applicability: str
    exclusions: str = ""
    provenance: list[dict] = field(default_factory=list)

    def validate(self):
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Skill needs an ID")
        if not set(self.role_views) <= set(ROLES) or len(self.role_views) < 2:
            raise ValueError("A cooperative skill must link at least two known roles")
        texts = [self.applicability, self.exclusions, *self.role_views.values()]
        if not all(isinstance(t, str) and len(t) <= 2000 for t in texts):
            raise ValueError("Oversized or non-text skill")
        if not isinstance(self.keywords, list) or len(self.keywords) > 20:
            raise ValueError("Invalid retrieval keywords")
        if not all(isinstance(k, str) and 0 < len(k) <= 30 for k in self.keywords):
            raise ValueError("Invalid retrieval keyword")


@dataclass
class Bundle:
    version: int = 0
    parent: str | None = None
    role_guidance: dict[str, str] = field(default_factory=dict)
    skills: list[Skill] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    status: str = "baseline"

    def validate(self):
        if self.status not in {"baseline", "candidate", "promoted", "rejected"}:
            raise ValueError("Invalid bundle status")
        if not set(self.role_guidance) <= set(ROLES):
            raise ValueError("Unknown role")
        if not all(
            isinstance(v, str) and len(v) <= 3500 for v in self.role_guidance.values()
        ):
            raise ValueError("Oversized role guidance")
        if len(self.skills) > 16 or len({s.id for s in self.skills}) != len(
            self.skills
        ):
            raise ValueError("Duplicate or excessive skills")
        for skill in self.skills:
            skill.validate()
        return self

    def dump(self):
        return asdict(self)

    @property
    def hash(self):
        return fingerprint(self.dump())

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        value["skills"] = [Skill(**s) for s in value.get("skills", [])]
        return cls(**value).validate()

    def retrieve(self, text, limit=3):
        """Retrieve a joint skill once, then project that same set to each role."""
        lowered = text.lower()
        ranked = sorted(
            (
                (sum(k.lower() in lowered for k in s.keywords), i, s)
                for i, s in enumerate(self.skills)
            ),
            key=lambda entry: (-entry[0], entry[1]),
        )
        return [s for score, _, s in ranked if score > 0][:limit]

    def project(self, role, selected):
        if role not in ROLES:
            raise ValueError("Unknown role")
        return {
            "guidance": self.role_guidance.get(role, ""),
            "skills": [
                {
                    "id": s.id,
                    "applicability": s.applicability,
                    "exclusions": s.exclusions,
                    "instruction": s.role_views[role],
                }
                for s in selected
                if role in s.role_views
            ],
        }


def validate_teacher_abstraction(bundle):
    """Reject obvious answer memorization; this is not a proof of abstraction."""
    text = json.dumps(
        {
            "guidance": bundle.role_guidance,
            "skills": [
                {k: v for k, v in asdict(s).items() if k not in {"id", "provenance"}}
                for s in bundle.skills
            ],
        },
        ensure_ascii=False,
    )
    if re.search(r"tab\d{4}|\b\d{3,}(?:\.\d+)?\b|[0-9a-f]{16}", text):
        raise ValueError("Teacher policy contains a task identifier or concrete number")
    return bundle.validate()
