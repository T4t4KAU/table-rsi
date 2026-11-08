"""Evidence-scoped state, finite edit operators, held-out promotion and RSI."""

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Evidence:
    category: str
    source: str
    scope: str
    property: str
    outcome: float | bool | None
    preconditions: bool = True


def admissible(e: Evidence, field: str, scope: str, prop: str = ""):
    if not e.preconditions or e.scope != scope or e.outcome is None:
        return False
    if field in {"cost", "executability"}:
        return e.category in {"E0", "E1", "E2"}
    if field == "property":
        return e.category in {"E1", "E2"} and e.property == prop
    if field == "success":
        return e.category == "E2" and e.source in {
            "authorized_development",
            "known_synthetic",
            "confirmed_correction",
        }
    return False


@dataclass
class Policy:
    initial_queries: int = 2
    replan_rounds: int = 1
    local_first: bool = True
    context_chars: int = 14000
    diversity: float = 0.15
    verify: bool = True

    # Legal choices only; no compiler, weights, or permission parameters.
    def validate(self):
        assert self.initial_queries in (1, 2, 3)
        assert self.replan_rounds in (0, 1, 2)
        assert self.context_chars in (9000, 14000, 18000)
        assert self.diversity in (0.0, 0.15, 0.35)


@dataclass
class State:
    version: int = 0
    policy: Policy = field(default_factory=Policy)
    memory: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    spent_audits: list = field(default_factory=list)

    def dump(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d["policy"] = Policy(**d["policy"])
        return cls(**d)

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.dump(), ensure_ascii=False, indent=2))
        os.replace(tmp, p)


OPERATORS = (
    "more_search",
    "less_search",
    "more_context",
    "more_branches",
    "local_first",
    "global_first",
    "memory_rule",
)
# Fixed, abstract operator-specific rules contain no dataset identifiers or answers.
RULES = [
    {
        "operator": "Aggregate",
        "condition": "ratio or percentage",
        "guidance": "Compute numerator and denominator from cited cells; preserve denominator scope and convert fractions to percentages explicitly.",
    },
    {
        "operator": "Filter",
        "condition": "range or period",
        "guidance": "Check inclusive/exclusive bounds and units against the question; do not infer missing periods.",
    },
    {
        "operator": "Join",
        "condition": "multiple tables",
        "guidance": "Verify join keys and duplicate multiplicities; do not join only because displayed row positions match.",
    },
    {
        "operator": "Project",
        "condition": "missing data",
        "guidance": "Distinguish missing data from zero, and state when requested statistics cannot be computed.",
    },
]


def propose(state: State, level=1, attempt=0, cap=4, seed=42):
    ops = list(OPERATORS)
    rng = random.Random(seed + attempt)
    # Common random numbers: L1 and L2 have identical proposals until L2 learns.
    weights = [
        math.exp(max(-5, min(5, state.meta.get(o, 0) if level == 2 else 0)))
        for o in ops
    ]
    ordered = []
    while ops:
        op = rng.choices(ops, weights, k=1)[0]
        j = ops.index(op)
        ordered.append(op)
        ops.pop(j)
        weights.pop(j)
    ops = ordered
    result = []
    for op in ops:
        z = copy.deepcopy(state)
        p = z.policy
        if op == "more_search":
            p.replan_rounds = min(2, p.replan_rounds + 1)
        elif op == "less_search":
            p.replan_rounds = max(0, p.replan_rounds - 1)
        elif op == "more_context":
            p.context_chars = 18000
        elif op == "more_branches":
            p.initial_queries = min(3, p.initial_queries + 1)
        elif op == "local_first":
            p.local_first = True
        elif op == "global_first":
            p.local_first = False
        elif op == "memory_rule":
            for r in RULES:
                if r not in z.memory:
                    z.memory.append(copy.deepcopy(r))
                    break
        p.validate()
        if z.policy != state.policy or z.memory != state.memory:
            result.append((op, z))
        if len(result) >= cap:
            break
    return result


def audit_gate(
    current,
    candidate,
    clusters,
    seed=42,
    min_clusters=4,
    delta=0.0,
    replicates=2000,
    regression_limit=0.20,
):
    if not (len(current) == len(candidate) == len(clusters)):
        raise ValueError("paired audit sizes differ")
    groups = sorted(set(clusters))
    diff = np.asarray(candidate) - np.asarray(current)
    gain = float(np.mean(diff)) if len(diff) else None
    if len(groups) < min_clusters:
        return {
            "accept": False,
            "gain": gain,
            "reason": "insufficient independent clusters",
        }
    rng = np.random.default_rng(seed)
    by = {g: np.where(np.asarray(clusters) == g)[0] for g in groups}
    draws = []
    for _ in range(replicates):
        idx = np.concatenate(
            [by[g] for g in rng.choice(groups, len(groups), replace=True)]
        )
        draws.append(float(diff[idx].mean()))
    lower = float(np.quantile(draws, 0.05))
    worst = min(float(diff[idx].mean()) for idx in by.values())
    return {
        "accept": lower > delta and worst >= -regression_limit,
        "gain": gain,
        "lower_95_one_sided": lower,
        "worst_cluster_gain": worst,
        "clusters": len(groups),
        "reason": "paired cluster bootstrap; approximate per-attempt bound",
    }


def commit_attempt(state: State, candidate: State, operator, audit_id, gate, level=1):
    if audit_id in state.spent_audits:
        raise ValueError("audit batch already consumed")
    z = copy.deepcopy(state)
    if gate.get("accept"):
        z.policy = copy.deepcopy(candidate.policy)
        z.memory = copy.deepcopy(candidate.memory)
        z.version += 1
    if level == 2 and gate.get("gain") is not None:
        z.meta[operator] = max(-5, min(5, z.meta.get(operator, 0) + 2 * gate["gain"]))
    z.spent_audits.append(audit_id)
    return z
