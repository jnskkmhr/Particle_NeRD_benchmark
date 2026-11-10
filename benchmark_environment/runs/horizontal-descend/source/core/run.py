# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one manager-based native MPM benchmark and record the same rollout as video/data."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from .motions import MOTIONS, SETTLE_SECONDS, TERRAINS, object_command
from .terrain import SPACING, VOXEL_SIZE


def create_config(args):
    """Compose a standalone task and its recording camera."""
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    from isaaclab.managers import ActionTermCfg
    from isaaclab.visualizers import WindowCfg

    from .mdp.actions import BodyWrenchAction
    from .terrain import configure_terrain

    if args.motion == "g1":
        from ..config.robot.g1_env_cfg import G1EnvCfg as ObjectEnvCfg
    elif args.motion == "quadruped":
        from ..config.robot.quadruped_env_cfg import QuadrupedEnvCfg as ObjectEnvCfg
    elif args.motion == "franka":
        from ..config.robot.franka_env_cfg import FrankaEnvCfg as ObjectEnvCfg
    elif args.motion == "rolling":
        from ..config.sphere.sphere_env_cfg import ObjectEnvCfg
    else:
        from ..config.cube.cube_env_cfg import ObjectEnvCfg
    cfg = ObjectEnvCfg()
    cfg.sim.device = args.device
    bed_size = {"franka": (1.2, 1.0, 0.25), "g1": (3.6, 1.6, 0.30), "quadruped": (2.0, 1.2, 0.12)}.get(
        args.motion, (2.0, 1.2, 0.30)
    )
    configure_terrain(cfg.scene, args.terrain, bed_size)
    if args.motion == "projectile":
        cfg.scene.object.spawn.size = (0.26, 0.18, 0.05)
        cfg.scene.object.spawn.rigid_props.kinematic_enabled = False
        cfg.scene.object.init_state.pos = (-0.55, 0.0, 0.35)
        cfg.actions.motion = ActionTermCfg(class_type=BodyWrenchAction, asset_name="object")
    cfg.sim.visualizer_cfgs = (
        []
        if args.no_video
        else [
            NewtonGLVisualizerCfg(
                headless=True,
                show_particles=True,
                particle_color=(0.72, 0.59, 0.36),
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                window=WindowCfg(size=(1280, 720)),
                streaming_view=False,
            )
        ]
    )
    if not args.no_video:
        cfg.sim.visualizer_cfgs[0].focal_length = 24.0
        if args.motion == "g1":
            cfg.sim.visualizer_cfgs[0].eye = (3.2, -4.4, 3.0)
            cfg.sim.visualizer_cfgs[0].lookat = (0.0, 0.0, 0.35)
    if args.no_video:
        args.visualizer = []
    return cfg, bed_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", choices=(*MOTIONS, "franka", "quadruped", "g1"), default="horizontal")
    parser.add_argument("--terrain", choices=TERRAINS, default="flat")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="G1 policy checkpoint; training_config.json must be alongside it"
    )
    parser.add_argument("--no-video", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if (
        not np.isfinite(args.duration)
        or args.duration <= 0
        or not np.isclose(args.duration * 50, round(args.duration * 50))
    ):
        parser.error("Duration must be positive and an integer number of 50 fps frames")
    if args.output.exists():
        parser.error("Output already exists; use a new run directory to preserve provenance")
    from isaaclab.envs import ManagerBasedEnv

    from .recording import Recorder, sha256

    cfg, bed_size = create_config(args)
    if args.motion == "g1" and (
        args.checkpoint is None
        or not args.checkpoint.is_file()
        or not (args.checkpoint.parent / "training_config.json").is_file()
    ):
        parser.error("G1 requires --checkpoint and its adjacent training_config.json")
    args.output.mkdir(parents=True)
    package = Path(__file__).parents[1]
    for source in package.rglob("*.py"):
        target = args.output / "source" / source.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    steps = round(args.duration / cfg.sim.dt)
    metadata = {
        "motion": args.motion,
        "terrain": args.terrain,
        "solver": "Implicit MPM",
        "physics_dt_s": cfg.sim.dt,
        "video_fps": 50,
        "duration_s": steps * cfg.sim.dt,
        "settling_time_s": SETTLE_SECONDS if args.motion in MOTIONS else 0.0,
        "seed": cfg.seed,
        "voxel_size_m": VOXEL_SIZE,
        "particle_spacing_m": SPACING,
        "bed_size_m": list(bed_size),
        "material": cfg.scene.sand.spawn.material.to_dict(),
        "interacting_asset": (
            cfg.scene.robot if args.motion in ("g1", "franka", "quadruped") else cfg.scene.object
        ).to_dict(),
        "controller": {
            "projectile": "Free projectile after release",
            "franka": "Franka damped least-squares IK",
            "quadruped": "Go2 free-base crawl with leg IK and PD joint drives",
            "g1": "G1 closed-loop MPM-trained RL policy",
        }.get(args.motion, "Prescribed kinematic collider"),
        "physics": cfg.sim.physics.to_dict(),
        "control_dt_s": cfg.sim.dt * cfg.decimation,
        "source_files": {
            str(f.relative_to(Path(__file__).parents[1])): sha256(f) for f in Path(__file__).parents[1].rglob("*.py")
        },
        "command": " ".join(sys.argv),
        "status": "running",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    encoder = None
    recorder = None
    original_update = None
    with launch_simulation(cfg, args):
        env = ManagerBasedEnv(cfg)
        try:
            with torch.inference_mode():
                obs, _ = env.reset()
                if args.motion == "quadruped":
                    from .quadruped_controller import CrawlController

                    controller = CrawlController(env)
                if args.motion == "g1":
                    from rsl_rl.models import MLPModel

                    saved = json.loads((args.checkpoint.parent / "training_config.json").read_text())
                    actor_cfg = saved["train_cfg"]["actor"].copy()
                    if actor_cfg.pop("class_name") != "MLPModel":
                        raise ValueError("Unsupported G1 actor class")
                    actor = MLPModel(
                        obs, saved["train_cfg"]["obs_groups"], "actor", env.action_manager.total_action_dim, **actor_cfg
                    ).to(env.device)
                    checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
                    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
                    actor.eval()
                    metadata.update(
                        policy_checkpoint=str(args.checkpoint.resolve()),
                        policy_sha256=sha256(args.checkpoint),
                        policy_iteration=int(checkpoint["iter"]),
                        controller="G1 closed-loop MPM-trained RL policy",
                        command_velocity_m_s=cfg.forward_speed,
                    )
                recorder = Recorder(env, args.output / "trajectory.h5", steps)
                recorder.record(0)
                physics_index = 0
                original_update = env.scene.update

                def record_physics(dt):
                    nonlocal physics_index
                    original_update(dt)
                    physics_index += 1
                    recorder.record(physics_index)

                env.scene.update = record_physics
                video_steps = []
                collection_start = time.perf_counter()
                for control_step in range(steps // cfg.decimation):
                    step = control_step * cfg.decimation
                    t = step * cfg.sim.dt
                    if args.motion == "g1":
                        action = actor(obs)
                    elif args.motion == "quadruped":
                        action = controller.command(t)
                    elif args.motion == "franka":
                        blend = np.clip((t - 1.0) / 1.5, 0.0, 1.0)
                        blend = blend * blend * (3 - 2 * blend)
                        travel = np.clip((t - 2.5) / 3.0, 0.0, 1.0)
                        x = -0.30 + 0.35 * travel
                        z = 0.30 * (1 - blend) - 0.04 * blend
                        # Command the fingertip frame in robot base coordinates.
                        action = torch.tensor([[x + 0.85, 0.0, z - 0.10, 1.0, 0.0, 0.0, 0.0]], device=env.device)
                    elif args.motion == "projectile":
                        action = torch.zeros((1, 6), device=env.device)
                        if t < SETTLE_SECONDS:
                            action[:, 2] = cfg.scene.object.spawn.mass_props.mass * 9.81
                        if step == round(SETTLE_SECONDS / cfg.sim.dt):
                            env.scene["object"].write_root_velocity_to_sim_index(
                                root_velocity=torch.tensor([[1.0, 0.0, 0.4, 0.0, 1.0, 0.0]], device=env.device)
                            )
                    else:
                        action = torch.as_tensor(
                            object_command(t, args.motion, args.terrain, cfg.sim.dt), device=env.device
                        )[None]
                    obs, _ = env.step(action)
                    if not args.no_video and physics_index % 4 == 0:
                        frame = env.sim._visualizers[0].render_rgb_array()
                        if frame is None:
                            raise RuntimeError("Visualizer did not produce a video frame")
                        if encoder is None:
                            h, w = frame.shape[:2]
                            encoder = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-v",
                                    "error",
                                    "-y",
                                    "-f",
                                    "rawvideo",
                                    "-pix_fmt",
                                    "rgb24",
                                    "-s",
                                    f"{w}x{h}",
                                    "-r",
                                    "50",
                                    "-i",
                                    "-",
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-crf",
                                    "20",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-movflags",
                                    "+faststart",
                                    str(args.output / "simulation.mp4"),
                                ],
                                stdin=subprocess.PIPE,
                            )
                        encoder.stdin.write(frame[:, :, :3].astype(np.uint8).tobytes())
                        video_steps.append(physics_index)
                    if physics_index % 200 == 0:
                        print(f"Recorded {physics_index * cfg.sim.dt:.1f} s / {steps * cfg.sim.dt:.1f} s", flush=True)
                metadata.update(
                    particle_count=recorder.model.particle_count,
                    body_names=recorder.model.body_label,
                    frame_state_indices=video_steps,
                    collection_wall_seconds=time.perf_counter() - collection_start,
                    timing_scope="Collection includes state downloads and video encoding; not solver performance",
                    status="complete",
                )
                env.scene.update = original_update
                recorder.close()
                if encoder:
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise RuntimeError("Video encoding failed")
                    encoder = None
                metadata["artifacts"] = {
                    name: sha256(args.output / name)
                    for name in ("trajectory.h5", *(() if args.no_video else ("simulation.mp4",)))
                }
                (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        finally:
            if original_update is not None:
                env.scene.update = original_update
            if recorder is not None:
                recorder.close()
            if encoder:
                encoder.stdin.close()
                encoder.wait()
            env.close()
    print(f"Completed {args.output}", flush=True)


if __name__ == "__main__":
    main()
