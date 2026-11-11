"""Export and plot matched dynamic-object curves; never run simulation here."""

import argparse
import hashlib
import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MPM, NERD = "#193c64", "#248044"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_curves(case, output):
    """Export only plotted scalar curves; full particle/body recordings stay private."""
    import h5py

    meta = json.loads((case / "evaluation.json").read_text())
    reference = Path(meta["reference_path"])
    if (
        digest(reference) != meta["reference_sha256"]
        or digest(case / "prediction.h5") != meta["prediction_sha256"]
    ):
        raise ValueError("Dynamic recording identity differs from its metadata")
    n = meta["completed_steps"]
    if (
        meta["failure"] is not None
        or n != meta["expected_steps"]
        or not meta["feedback_enabled"]
    ):
        raise ValueError("Publication requires a complete coupled simulation")
    with h5py.File(reference) as r, h5py.File(case / "prediction.h5") as p:
        if not p.attrs["complete"]:
            raise ValueError("Incomplete saved simulation")
        if not np.array_equal(r["particle_position"][0], p["particle_position"][0]):
            raise ValueError("Initial particle states differ")
        if not np.array_equal(r["body_state"][0], p["body_state"][0]):
            raise ValueError("Initial body states differ")
        if not np.array_equal(
            p["applied_material_wrench"][1:], p["material_wrench"][:-1]
        ):
            raise ValueError(
                "Saved force feedback is not the documented lagged coupling"
            )
        data = {
            "shape": meta["episode"]["shape"],
            "native_dt_s": meta["native_dt_s"],
            "completed_steps": n,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "reference_sha256": meta["reference_sha256"],
            "prediction_sha256": meta["prediction_sha256"],
            "time_s": (np.arange(n + 1) * meta["native_dt_s"]).tolist(),
            "mpm_body_com_m": r["body_state"][: n + 1, :3].tolist(),
            "nerd_body_com_m": p["body_state"][:, :3].tolist(),
            "mpm_material_force_N": r["material_force"][:n].tolist(),
            "nerd_material_force_N": p["material_wrench"][:, :3].tolist(),
            "contact_iteration_limit_count": int(
                (p["contact_stats"][:, 1] >= 2000).sum()
            ),
            "contact_max_residual": float(p["contact_stats"][:, 2].max()),
            "definition": "Independent fully dynamic MPM and NeRD rollouts from identical initial states. Body centers of mass and raw world-frame material-on-body forces. Forces are assigned to interval end times; the NeRD wrench is applied in the following interval. No smoothing or sample removal.",
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "plot_data.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def caption(shape, number):
    return (
        f"Figure {number}: {shape.title()}, one 20 s fully dynamic rollout per method from identical initial states. "
        "Top row: object center-of-mass x, y, z in world coordinates. Bottom row: raw material-on-object Fx, Fy, Fz; "
        "direct rigid-tray forces are excluded. The upward holding force is removed at 1 s. "
        "Each method evolves its own object and particles; NeRD force and torque feed back through the next 5 ms coupling exchange. "
        "All 200 Hz samples are shown without smoothing. The paired video renders these same saved states at 50 fps."
    )


def plot(root, number):
    data = json.loads((root / "plot_data.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    time = np.array(data["time_s"])
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    for prefix, label, color in (
        ("mpm", "Implicit MPM", MPM),
        ("nerd", model_label, NERD),
    ):
        position, force = (
            np.array(data[prefix + "_body_com_m"]),
            np.array(data[prefix + "_material_force_N"]),
        )
        for j, axis in enumerate("xyz"):
            axes[0, j].plot(time, position[:, j], color=color, lw=1.2, label=label)
            axes[1, j].plot(time[1:], force[:, j], color=color, lw=1.0, label=label)
            axes[0, j].set_ylabel(f"Object COM ${axis}$ (m)")
            axes[1, j].set_ylabel(f"$F_{axis}$ (N)")
    for ax in axes.flat:
        ax.set_xlabel("Simulation time (s)")
        ax.set_xlim(time[0], time[-1])
        ax.grid(alpha=0.15)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.08, right=0.985, top=0.90, bottom=0.24, wspace=0.30, hspace=0.48
    )
    fig.text(
        0.5,
        0.025,
        textwrap.fill(caption(data["shape"], number), 155),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    outputs = []
    for ext in ("png", "pdf", "svg"):
        path = root / f"dynamic_response.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    provenance = {
        "plotter_sha256": digest(Path(__file__)),
        "input_sha256": digest(root / "plot_data.json"),
        "outputs": {path.name: digest(path) for path in outputs},
        "figure_number": number,
    }
    (root / "plot_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--private-case",
        type=Path,
        help="Optional private recording to export before plotting",
    )
    parser.add_argument("--figure-number", type=int, required=True)
    args = parser.parse_args()
    if args.private_case:
        export_curves(args.private_case, args.directory)
    plot(args.directory, args.figure_number)


if __name__ == "__main__":
    main()
