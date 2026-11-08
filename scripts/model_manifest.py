import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-8B"
root = Path("/data1/sdnware/tablersi")
model = root / "models" / name
files = {}
for p in sorted(model.iterdir()):
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        files[p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
record = {
    "source": "https://modelscope.cn/models/Qwen/" + name,
    "model_id": "Qwen/" + name,
    "weight_training": False,
    "dtype": "bfloat16",
    "files": files,
    "python": platform.python_version(),
    "gpu": subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv",
        ],
        text=True,
    ),
    "dataset_commit": "f70f134f63e4a7b0c4d15b2540953b5e95388586",
}
(root / ("runs/model_manifest_" + name + ".json")).write_text(
    json.dumps(record, indent=2)
)
print("Wrote model manifest:", len(files), "files")
