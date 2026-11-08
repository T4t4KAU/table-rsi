"""Create an inference-only store from a directory of CSV/XLS/XLSX files."""

import argparse
import json
from pathlib import Path

from tablersi.data import build_store

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
source, dest = Path(args.input_dir), Path(args.out)
if not source.is_dir():
    parser.error("--input-dir must be an existing directory")
if dest.exists():
    parser.error("--out must be a new directory; existing stores are never overwritten")
catalog, errors = build_store(source, dest)
print(
    json.dumps(
        {"store": str(dest.resolve()), "tables": len(catalog), "errors": errors},
        ensure_ascii=False,
        indent=2,
    )
)
raise SystemExit(0 if catalog and not errors else 1)
