"""Answer a table question locally without loading labels or a cloud scorer."""

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent
from .model import Client
from .state import State


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--store",
        required=True,
        help="Directory containing tables.sqlite and catalog.json",
    )
    p.add_argument("--question", help="If omitted, read the question from stdin")
    p.add_argument(
        "--history",
        help="Optional JSON array of prior user/assistant messages; never evaluator outputs",
    )
    snapshot = p.add_mutually_exclusive_group()
    snapshot.add_argument(
        "--state", help="Committed policy/memory JSON; defaults to the fixed controller"
    )
    snapshot.add_argument("--bundle", help="Released cooperative policy/memory bundle")
    p.add_argument("--local-url", default="http://127.0.0.1:31001/v1")
    p.add_argument("--local-model", default="tablersi-qwen3-8b")
    p.add_argument("--output", help="Optional result JSON with trace and token ledger")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thinking-budget", type=int)
    a = p.parse_args()
    question = a.question if a.question is not None else sys.stdin.read().strip()
    if not question:
        p.error("a nonempty question is required")
    store = Path(a.store)
    if (
        not (store / "tables.sqlite").is_file()
        or not (store / "catalog.json").is_file()
    ):
        p.error("store must contain tables.sqlite and catalog.json")
    history = json.loads(Path(a.history).read_text()) if a.history else []
    thinking = (
        a.thinking_budget
        if a.thinking_budget is not None
        else (1024 if a.bundle else 0)
    )
    if thinking < 0 or thinking >= 4096:
        p.error("thinking budget must be in [0, 4096)")
    client = Client(a.local_url, a.local_model, local=True, thinking_budget=thinking)
    if a.bundle:
        from .cooperative import CooperativeAgent
        from .memory import Bundle

        bundle = Bundle.load(a.bundle)
        if bundle.status not in {"baseline", "promoted"}:
            p.error(
                "candidate/rejected bundles are not deployment releases; use the study evaluator"
            )
        agent = CooperativeAgent(client, bundle)
    else:
        agent = Agent(client, State.load(a.state) if a.state else State())
    result = agent.solve(store, question, history, a.seed)
    if a.output:
        dest = Path(a.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(result["answer"])


if __name__ == "__main__":
    main()
