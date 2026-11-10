# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a local video gallery only from validated, paired MPM run artifacts."""

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


TITLES = {
    "horizontal": "Horizontal box push",
    "vertical": "Vertical box intrusion",
    "circle": "Box circling",
    "rolling": "Sphere rolling",
    "projectile": "Shallow box projectile",
    "franka": "Franka sand pushing",
    "quadruped": "Go2 crawl (leg IK)",
    "g1": "G1 walking (RL)",
}
DESCRIPTIONS = {
    "horizontal": "A 0.24 × 0.18 × 0.12 m box enters the sand, then follows a prescribed 1.30 m horizontal path.",
    "vertical": "The box enters the sand and continues downward by 0.12 m; its pose and velocity are prescribed.",
    "circle": "The box follows a prescribed circle of radius 0.35 m while immersed in the material.",
    "rolling": (
        "A sphere of radius 0.10 m translates with prescribed spin. Sand reacts to the imposed motion; this "
        "is not a freely rolling sphere."
    ),
    "projectile": (
        "A 2 kg shallow box (0.26 × 0.18 × 0.05 m) is released after one second with velocity (1, 0, 0.4) "
        "m/s and angular velocity (0, 1, 0) rad/s. Its subsequent motion is dynamic."
    ),
    "franka": (
        "A fixed-base Franka arm uses damped least-squares IK and joint drives to push its closed gripper "
        "through the sand. The dataset stores the achieved motion."
    ),
    "quadruped": (
        "A freely balancing Unitree Go2 follows a slow crawl with leg IK and PD drives. The 0.12 m sand "
        "layer lies on a rigid tray; feet may also contact its floor. No base pose or external support "
        "wrench is prescribed."
    ),
    "g1": (
        "The MPM-trained G1 policy jqur5jhm (iteration 19,999) controls the humanoid at 50 Hz, with a 0.35 "
        "m/s forward command. Physics runs at 200 Hz."
    ),
}


def publish(runs, output, checkpoint=None, release_url=None):
    output.mkdir(parents=True, exist_ok=True)
    catalog = []
    skipped = []
    for run in sorted(runs.iterdir()):
        if not (run / "validation.json").exists():
            continue
        validation = json.loads((run / "validation.json").read_text())
        if not validation["passed"]:
            skipped.append(run.name)
            continue
        metadata = json.loads((run / "metadata.json").read_text())
        for filename, digest in metadata["artifacts"].items():
            if sha256(run / filename) != digest:
                raise ValueError(f"Changed artifact: {run / filename}")
        if not (run / "simulation.mp4").exists():
            raise ValueError(f"Missing paired video: {run}")
        destination = output / "runs" / run.name
        if destination.exists():
            if sha256(destination / "metadata.json") != sha256(run / "metadata.json"):
                raise FileExistsError(
                    f"Refusing to replace a different published run: {destination}"
                )
            for filename, digest in metadata["artifacts"].items():
                if sha256(destination / filename) != digest:
                    raise ValueError(
                        f"Published artifact changed: {destination / filename}"
                    )
        else:
            shutil.copytree(run, destination)
        motion = metadata["motion"]
        command = (
            f"PYTHONPATH=examples uv run --no-sync python -m particle_nerd_benchmark.core.run "
            f"--motion {motion} --terrain {metadata['terrain']} --duration {metadata['duration_s']:g} "
            f"--output /tmp/{run.name}-new"
        )
        if motion == "g1":
            command += " --checkpoint " + str(
                checkpoint or metadata["policy_checkpoint"]
            )
        catalog.append(
            {
                "id": run.name,
                "title": TITLES[motion],
                "description": DESCRIPTIONS[motion],
                "motion": motion,
                "terrain": metadata["terrain"],
                "family": "robot"
                if motion in ("g1", "franka", "quadruped")
                else "object",
                "path": "runs/" + run.name,
                "dataset_url": f"{release_url}/{run.name}-trajectory.h5"
                if release_url
                else None,
                "duration": metadata["duration_s"],
                "particles": metadata["particle_count"],
                "controller": metadata["controller"],
                "control_hz": round(1.0 / metadata["control_dt_s"]),
                "bed": metadata["bed_size_m"],
                "policy": "jqur5jhm · iteration 19,999" if motion == "g1" else None,
                "command": command,
                "date": str(date.today()),
                "artifacts": metadata["artifacts"],
            }
        )
    catalog.sort(
        key=lambda c: (
            list(TITLES).index(c["motion"]),
            ["flat", "ascend", "descend", "wave"].index(c["terrain"]),
        )
    )
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (output / "catalog.js").write_text(
        "const BENCHMARK_CASES = " + json.dumps(catalog, indent=2) + ";\n"
    )
    print(f"Published {len(catalog)} cases; excluded failed cases: {skipped}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Persistent G1 checkpoint path for reproduction commands",
    )
    parser.add_argument(
        "--release-url", help="GitHub Release URL for large dataset downloads"
    )
    args = parser.parse_args()
    publish(args.runs, args.output, args.checkpoint, args.release_url)


if __name__ == "__main__":
    main()
