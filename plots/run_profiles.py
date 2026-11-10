"""Run each requested command in a fresh current-solver environment."""

import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("nerd", "mpm"), default="nerd")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=False)
profiles = [
    ("forward", 0.5, 0.0, 0.0),
    ("backward", -0.5, 0.0, 0.0),
    ("left", 0.0, 0.5, 0.0),
    ("right", 0.0, -0.5, 0.0),
    ("forward_yaw_positive", 0.5, 0.0, 1.0),
    ("forward_yaw_negative", 0.5, 0.0, -1.0),
]
status = []
for name, vx, vy, wz in profiles:
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--extra",
        "ovrtx",
        "python",
        str(Path(__file__).with_name("record_full_profiles.py")),
        "--backend",
        args.backend,
        "--checkpoint",
        str(args.checkpoint),
        "--speed",
        str(vx),
        "--vy",
        str(vy),
        "--wz",
        str(wz),
        "--duration",
        "6",
        "--output",
        str(root / name),
    ]
    print("START", name, flush=True)
    with (root / f"{name}.log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    status.append({"profile": name, "command": cmd, "returncode": result.returncode})
    (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print("DONE", name, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect log before continuing")
