"""Download a model checkpoint from ModelScope to an explicit local directory."""

import argparse
import json
import time
from pathlib import Path


def main(default_model="Qwen/Qwen3-8B"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from modelscope import snapshot_download

    path = snapshot_download(args.model, local_dir=str(args.out))
    print(
        json.dumps(
            {
                "model": args.model,
                "provider": "ModelScope",
                "local_dir": path,
                "completed": time.time(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
