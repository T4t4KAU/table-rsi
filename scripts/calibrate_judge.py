import json
from pathlib import Path

from tablersi.evaluate import Judge
from tablersi.model import Client

root = Path("/data1/sdnware/tablersi")
judge = Judge(
    Client(
        "https://api.deepseek.com",
        "deepseek-flash",
        "/home/sdnware/.config/tablersi/deepseek.key",
        False,
    ),
    root / "data/TableAgentBench",
)
cases = [
    ("complete", "2020年销售额为100，2021年为120，增长20%。", 1),
    ("wrong", "2020年销售额为50，2021年为60，下降20%。", 0),
    ("partial", "2020年销售额为100，2021年为120。", 2 / 3),
]
records = []
for name, answer, expected in cases:
    score = judge.score(
        "2020、2021年销售额及增长率？",
        answer,
        ["2020年销售额100", "2021年销售额120", "销售额增长20%"],
    )
    records.append(
        {
            "case": name,
            "expected": expected,
            "score": score,
            "passed": score["coverage_ratio"] == expected,
        }
    )
(root / "runs/judge_calibration.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2)
)
print(json.dumps([{"case": r["case"], "passed": r["passed"]} for r in records]))
