"""Read completed-file counts without exposing reference answers."""

import json
from pathlib import Path

root = Path("/data1/sdnware/tablersi/runs/study-v4")
result = {"status": json.loads((root / "status.json").read_text())["stage"]}
for level in ("L1", "L2"):
    a = root / level / "attempt1"
    result[level] = {
        p.name: len(list(p.glob("tab*.json")))
        for p in sorted(a.iterdir())
        if p.is_dir()
    }
    if (a / "decision.json").exists():
        result[level]["decision"] = json.loads((a / "decision.json").read_text())
if (root / "test").exists():
    result["test"] = {
        p.name: {
            "dialogs": len(list(p.glob("tab*.json"))),
            "complete": (p / "summary.json").exists(),
        }
        for p in sorted((root / "test").iterdir())
        if p.is_dir()
    }
print(json.dumps(result, ensure_ascii=False))
