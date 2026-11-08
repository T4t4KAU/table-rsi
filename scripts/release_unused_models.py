"""Release only this task's unused model servers after development selection."""

import json
import os
import signal
import time
from pathlib import Path

root = Path("/data1/sdnware/tablersi")
status = root / "runs/study-v3/status.json"
while True:
    if status.exists():
        s = json.loads(status.read_text())
        if s["stage"] in ("adapting", "terminal_evaluation", "complete"):
            break
        if s["stage"] == "failed":
            raise RuntimeError("study failed before selection")
    time.sleep(5)
selected = s["selected"]
stopped = []
files = []
if not selected.startswith("8b"):
    files += ["vllm.pid", "vllm8-thinking.pid"]
elif selected == "8b":
    files += ["vllm8-thinking.pid", "vllm32.pid", "vllm32-replica.pid"]
else:
    files += ["vllm.pid", "vllm32.pid", "vllm32-replica.pid"]
for name in files:
    p = root / "logs" / name
    if not p.exists():
        continue
    pid = int(p.read_text())
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        continue
    cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    if (
        proc.stat().st_uid != os.getuid()
        or str(root / "models") not in cmd
        or "vllm.entrypoints.openai.api_server" not in cmd
    ):
        raise RuntimeError("Refusing to signal an unrecognized process")
    os.kill(pid, signal.SIGTERM)
    stopped.append({"pid": pid, "pidfile": name})
(root / "runs/study-v3/released_unused_models.json").write_text(
    json.dumps(
        {"selected": selected, "stopped": stopped, "time": time.time()}, indent=2
    )
)
print(json.dumps({"selected": selected, "stopped": stopped}))
