"""Verify matched initial states and assemble same-clock closed-loop comparison videos."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "g1_humanoid/data"
PROFILES = ("forward", "backward", "left", "right", "forward_yaw_positive", "forward_yaw_negative")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,r_frame_rate,duration,width,height",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]


def compare(name):
    folders = {"nerd": DATA / "full-20261109" / name, "mpm": DATA / "full-mpm-20261109" / name}
    meta = {k: read(p / "metadata.json") for k, p in folders.items()}
    matched_keys = [
        "seed",
        "checkpoint_sha256",
        "initial_particles_sha256",
        "material",
        "particle_grid",
        "robot_mass_kg",
        "particle_mass_kg",
        "particle_count",
        "command_body_frame",
        "physics_dt_s",
        "control_dt_s",
        "foot_names",
        "initial_root_position_m",
        "native_warmup_steps",
    ]
    for key in matched_keys:
        assert meta["mpm"][key] == meta["nerd"][key], (name, key)
    a, b = (meta[k]["physics"] for k in ("mpm", "nerd"))
    assert a["solver_cfg"]["entries"][0] == b["solver_cfg"]["entries"][0]
    assert a["solver_cfg"]["proxies"] == b["solver_cfg"]["proxies"]
    for key in ("collision_cfg", "default_shape_cfg", "default_joint_cfg", "num_substeps"):
        assert a.get(key) == b.get(key), key
    assert meta["mpm"]["solver_class"].endswith(".SolverImplicitMPM")
    with h5py.File(folders["mpm"] / "states.h5") as f, h5py.File(folders["nerd"] / "states.h5") as g:
        for key in ("particle_position", "body_pose"):
            np.testing.assert_array_equal(f[key][0], g[key][0])
    traces = {k: np.load(p / "trace.npz") for k, p in folders.items()}
    np.testing.assert_array_equal(traces["mpm"]["action"][0], traces["nerd"]["action"][0])
    times = {}
    common_end = min(m["recorded_seconds"] for m in meta.values())
    result = {"common_measurement_end_s": common_end, "profiles": {}}
    for backend, folder in folders.items():
        m, d = meta[backend], traces[backend]
        assert not m["simulation_uses_robot_reference"] and not m["simulation_uses_mpm_reference"]
        assert m["native_warmup_steps"] == 0
        assert m["outcome"] in ("duration", "reached_sand_bed_edge", "terminated")
        assert np.isclose(d["time"][-1], m["recorded_seconds"])
        np.testing.assert_allclose(d["time"], np.arange(1, len(d["time"]) + 1) * 0.005)
        np.testing.assert_allclose(d["force"], d["sand_force"] + d["rigid_force"], atol=1e-5)
        expected = np.array([m["command_body_frame"][k] for k in ("vx_m_s", "vy_m_s", "wz_rad_s")])
        assert np.allclose(d["command"][:, :3], expected)
        for key in d.files:
            assert np.isfinite(d[key]).all(), (name, backend, key)
        particles = np.load(folder / "particle_metrics.npz")
        times[backend] = particles["video_times"]
        metrics = particles["metrics"]
        assert np.isfinite(metrics).all()
        np.testing.assert_allclose(times[backend], np.arange(len(times[backend])) / 50)
        np.testing.assert_array_equal(times[backend], metrics[:, 0])
        with h5py.File(folder / "states.h5") as f:
            initial = f["particle_position"][0]
            assert hashlib.sha256(initial.tobytes()).hexdigest() == m["initial_particles_sha256"]
            for j, t in enumerate(f["time_s"][:]):
                q = f["particle_position"][j]
                delta = q - initial
                measured = [
                    np.sqrt(np.mean(np.sum(delta**2, axis=-1))),
                    delta[:, 2].mean(),
                    q[:, 2].min(),
                    q[:, 2].max(),
                ]
                np.testing.assert_allclose(metrics[round(float(t) * 50), 1:], measured, rtol=1e-4, atol=1e-7)
                if t > 0:
                    row = round(float(t) / 0.005) - 1
                    body = f["body_pose"][j, :, :3]
                    for xyz in [*d["foot_position"][row], d["root_position"][row]]:
                        assert np.min(np.linalg.norm(body - xyz, axis=1)) < 2e-5
        v = probe(folder / "simulation.mp4")
        assert v["r_frame_rate"] == "50/1" and int(v["nb_frames"]) == len(times[backend]) == m["video_frames"]
        assert np.isclose(float(v["duration"]), len(times[backend]) / 50)
        mask = d["time"] <= common_end + 1e-9
        actual = np.column_stack((d["root_velocity_body"][mask, :2], d["root_angular_velocity_body"][mask, 2]))
        result["profiles"][backend] = {
            "duration_s": m["recorded_seconds"],
            "outcome": m["outcome"],
            "terminal_terms": m.get("terminal_terms", []),
            "physics_samples": len(d["time"]),
            "common_samples": int(mask.sum()),
            "mean_body_velocity": actual.mean(0).tolist(),
            "tracking_rmse": np.sqrt(np.mean((actual - expected) ** 2, axis=0)).tolist(),
            "mean_foot_force_N": d["force"][mask].mean(0).tolist(),
            "std_foot_force_N": d["force"][mask].std(0).tolist(),
            "max_abs_rigid_force_N": float(abs(d["rigid_force"]).max()),
            "final_base_position_m": d["root_position"][-1].tolist(),
        }
    count = min(len(t) for t in times.values())
    np.testing.assert_array_equal(times["nerd"][:count], times["mpm"][:count])
    result.update(
        video_frames=count,
        video_fps=50,
        last_video_state_s=float(times["mpm"][count - 1]),
        encoded_video_duration_s=count / 50,
    )
    dst = DATA / "full-comparison-20261109" / name
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-i",
            str(folders["mpm"] / "simulation.mp4"),
            "-threads",
            "2",
            "-i",
            str(folders["nerd"] / "simulation.mp4"),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2:shortest=1[v]",
            "-map",
            "[v]",
            "-frames:v",
            str(count),
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst / "comparison.mp4"),
        ],
        check=True,
    )
    v = probe(dst / "comparison.mp4")
    assert v["r_frame_rate"] == "50/1" and int(v["nb_frames"]) == count and v["width"] == 2560
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(min(2.0, result["last_video_state_s"])),
            "-i",
            str(dst / "comparison.mp4"),
            "-frames:v",
            "1",
            str(dst / "comparison.jpg"),
        ],
        check=True,
    )
    result["verification"] = {
        "matched_metadata": matched_keys,
        "initial_states": "Bit-identical particle positions and rigid-body poses before the first step",
        "rigid_physics": "Identical rigid solver entry, proxy mapping, collision pipeline, default joint and shape settings",
        "video": "MPM left, NeRD right; matching simulation timestamps; stops at shorter recording, no frozen-frame extension",
        "snapshot_checks": "1 Hz saved particle states reproduce displacement statistics; body positions match force trace timestamps",
        "trace_checks": "200 Hz finite samples, constant command, force = sand force + rigid-geometry force",
        "interpretation": "Independent closed-loop trajectories; curve differences include changed robot motion and are not isolated force-estimator errors",
        "source_artifacts_sha256": {
            backend: {
                file: digest(folder / file)
                for file in ("metadata.json", "trace.npz", "particle_metrics.npz", "states.h5", "simulation.mp4")
            }
            for backend, folder in folders.items()
        },
        "comparison_video_sha256": digest(dst / "comparison.mp4"),
    }
    (dst / "verification.json").write_text(json.dumps(result["verification"], indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES)
    args = parser.parse_args()
    output = DATA / "full-comparison-20261109/summary.json"
    summary = read(output) if output.exists() else {}
    for name in (args.profile,) if args.profile else PROFILES:
        summary[name] = compare(name)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print("VERIFIED", name, {k: v["tracking_rmse"] for k, v in summary[name]["profiles"].items()}, flush=True)
