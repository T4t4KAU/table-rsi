"""Materialize a bootstrap/deployment split without moving old tests into training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

PARTITIONS = {
    "dev": "predeploy_validation",
    "proposal": "predeploy_train",
    "audit0": "predeploy_train",
    "audit1": "predeploy_train",
    "test": "postdeploy_test",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def counts(tasks):
    return {
        "clusters": len({t["cluster"] for t in tasks}),
        "dialogs": len(tasks),
        "turns": sum(len(t["questions"]) for t in tasks),
        "unsupported_dialogs": sum(not t["supported"] for t in tasks),
    }


def partition(source: Path, protocol: Path, output: Path):
    """Keep all old test groups; annotate previous exposure at whole-cluster scope."""
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing frozen partition")
    tasks = read_json(source / "public.json")
    oracle = read_json(source / "oracle.json")
    prior = read_json(protocol)
    ids = {t["id"] for t in tasks}
    if len(ids) != len(tasks) or ids != set(oracle):
        raise ValueError("Duplicate task IDs or mismatched reference IDs")
    prior_ids = set(prior["partitions"]["test"]["ids"])
    if not prior_ids <= ids:
        raise ValueError("Unknown previous test IDs")
    if any(t["split"] != "test" for t in tasks if t["id"] in prior_ids):
        raise ValueError("A previously tested task would enter predeployment data")
    exposed = {t["cluster"] for t in tasks if t["id"] in prior_ids}
    buckets = defaultdict(list)
    ownership = {}
    for task in tasks:
        target = PARTITIONS[task["split"]]
        for key in (("cluster", task["cluster"]), ("store", task["group"])):
            if ownership.setdefault(key, target) != target:
                raise ValueError("A table group crosses the deployment boundary")
        if len(task["questions"]) != len(oracle[task["id"]]):
            raise ValueError("Question/reference turn counts differ")
        store = source / "stores" / task["group"]
        if store.exists():
            for sheet in read_json(store / "catalog.json"):
                key = ("sheet", sheet["content_sha256"])
                if ownership.setdefault(key, target) != target:
                    raise ValueError("Identical sheet content crosses partitions")
        elif task["supported"]:
            raise ValueError("A supported task has no table store")
        buckets[target].append(
            {
                **task,
                "original_split": task["split"],
                "split": target,
                "previous_terminal_group": task["cluster"] in exposed,
                "previous_terminal_dialog": task["id"] in prior_ids,
            }
        )
    if set(buckets) != set(PARTITIONS.values()):
        raise ValueError("One of the required partitions is empty")
    test_tasks = buckets["postdeploy_test"]
    manifest = {
        "version": "bootstrap-v1",
        "source_hashes": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("public.json", "oracle.json", "manifest.json")
        },
        "previous_protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "partitions": {name: counts(rows) for name, rows in buckets.items()},
        "postdeploy_subsets": {
            "previous_terminal_groups": counts(
                [t for t in test_tasks if t["previous_terminal_group"]]
            ),
            "not_in_previous_terminal_groups": counts(
                [t for t in test_tasks if not t["previous_terminal_group"]]
            ),
        },
        "checks": {
            "disjoint_clusters": True,
            "disjoint_stores": True,
            "disjoint_exact_sheet_content": True,
            "all_dialogs_retained": sum(map(len, buckets.values())) == len(tasks),
            "deployment_references_separate": True,
        },
        "limitations": [
            "Exact-sheet separation does not establish semantic-template independence.",
            "Prior terminal results have been inspected; restarting does not undo this.",
            "File separation is not an OS permission boundary.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partition-", dir=output.parent))
    try:
        for name, rows in buckets.items():
            dest = staging / name
            write_json(dest / "public.json", rows)
            write_json(dest / "manifest.json", {"partition": name, **manifest})
            references = {t["id"]: oracle[t["id"]] for t in rows}
            oracle_dir = staging / "scorer_only" if name == "postdeploy_test" else dest
            write_json(oracle_dir / "oracle.json", references)
            for group in sorted({t["group"] for t in rows}):
                store = source / "stores" / group
                if store.exists():
                    shutil.copytree(store, dest / "stores" / group)
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = partition(args.source, args.previous_protocol, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
