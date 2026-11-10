# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot saved evaluation metrics without smoothing or shifting simulation time."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

MPM, NERD = "#193c64", "#248044"


def read_table(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return (
        {
            key: np.array([float(row[key]) if row.get(key) else np.nan for row in rows])
            for key in rows[0]
        }
        if rows
        else {}
    )


def save(figure, output, name):
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / (name + "." + suffix), dpi=300, pad_inches=0.20)
    plt.close(figure)


def write_provenance(root, meta, names):
    """Bind generated figures to their measured inputs and predicted states."""
    inputs = [
        "evaluation.json",
        "prediction.h5",
        "particle_metrics.csv",
        "field_metrics.csv",
    ]
    if (root / "mass_difference_maps.npz").exists():
        inputs.append("mass_difference_maps.npz")
    figures = [
        name + "." + suffix for name in names for suffix in ("png", "pdf", "svg")
    ]
    hashes = {}
    for name in inputs + figures:
        with (root / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (root / "plot_provenance.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "reference_sha256": meta["reference_sha256"],
                "prediction_sha256": hashes["prediction.h5"],
                "inputs_sha256": {name: hashes[name] for name in inputs},
                "figures_sha256": {name: hashes[name] for name in figures},
                "plot_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


def plot_mass_maps(root):
    """Show signed physical-mass errors at the start, middle, and end of the rollout."""
    path = root / "mass_difference_maps.npz"
    if not path.exists():
        return
    with np.load(path) as data:
        selected = np.unique(
            np.linspace(0, len(data["time_s"]) - 1, min(3, len(data["time_s"]))).astype(
                int
            )
        )
        figure, axes = plt.subplots(2, len(selected), figsize=(13.2, 7), squeeze=False)
        figure.subplots_adjust(
            top=0.85, bottom=0.20, hspace=0.35, wspace=0.28, right=0.88
        )
        limit = max(
            float(np.max(np.abs(data[key][selected]))) for key in ("xy_kg", "xz_kg")
        )
        limit = max(limit, 1e-9)
        palette = LinearSegmentedColormap.from_list(
            "mass_difference", [MPM, "#ffffff", NERD]
        )
        normalization = SymLogNorm(linthresh=0.01, vmin=-limit, vmax=limit, base=10)
        for row, (key, coordinate) in enumerate((("xy_kg", 1), ("xz_kg", 2))):
            for column, frame in enumerate(selected):
                axis = axes[row, column]
                values = data[key][frame]
                lower, spacing = data["grid_lower_m"], float(data["grid_spacing_m"])
                extent = [
                    lower[0] - spacing / 2,
                    lower[0] + (values.shape[0] - 0.5) * spacing,
                    lower[coordinate] - spacing / 2,
                    lower[coordinate] + (values.shape[1] - 0.5) * spacing,
                ]
                image = axis.imshow(
                    values.T,
                    origin="lower",
                    extent=extent,
                    cmap=palette,
                    norm=normalization,
                    interpolation="nearest",
                )
                axis.set_title(f"{data['time_s'][frame]:g} s", loc="left")
                axis.set(xlabel="World x (m)", ylabel=f"World {'xyz'[coordinate]} (m)")
        colorbar = figure.add_axes((0.91, 0.27, 0.015, 0.48))
        figure.colorbar(
            image, cax=colorbar, label="Mass difference per projected column (kg)"
        )
        times = data["time_s"]
        upper_labels = ", ".join(f"{times[i]:g} s" for k, i in enumerate(selected))
        lower_labels = ", ".join(f"{times[i]:g} s" for k, i in enumerate(selected))
        figure.text(
            0.5,
            0.035,
            "Figure 4: Local material redistribution error in one paired rollout at the labeled times.\n"
            f"Top row {upper_labels}: xy maps summed along z.\n"
            f"Bottom row {lower_labels}: xz maps summed along y.\n"
            "Green indicates excess predicted mass; navy indicates missing predicted mass. "
            "Both use the same 0.02 m grid.\n"
            "The shared color scale is symmetric logarithmic, linear within ±0.01 kg; no values are clipped.",
            ha="center",
            fontsize=10,
        )
        save(figure, root, "mass_difference")


def plot_body_mass(root, fields):
    """Account for local depletion, particle-centre penetration, and represented mass."""
    if not fields:
        return
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    figure.subplots_adjust(bottom=0.30, top=0.82, wspace=0.30)
    for prefix, label, color in (
        ("reference", "Implicit MPM", MPM),
        ("predicted", "Particle NeRD", NERD),
    ):
        below = fields[prefix + "_below_body_mass_kg"]
        for axis, values in zip(
            axes,
            (
                below - below[0],
                fields[prefix + "_inside_body_mass_kg"],
                fields[prefix + "_mass_kg"],
            ),
            strict=True,
        ):
            axis.plot(fields["time_s"], values, label=label, color=color)
            axis.ticklabel_format(axis="y", useOffset=False)
    for axis, label in zip(
        axes,
        (
            "Change of mass\nbeneath body (kg)",
            "Mass inside body (kg)",
            "Total represented mass (kg)",
        ),
        strict=True,
    ):
        axis.set(xlabel="Simulation time (s)", ylabel=label)
        axis.grid(alpha=0.15)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
    )
    axes[2].set_ylim(
        0,
        1.05
        * max(fields["reference_mass_kg"].max(), fields["predicted_mass_kg"].max()),
    )
    figure.text(
        0.5,
        0.035,
        "Figure 5: Physical material mass in one paired rollout. "
        "Left: change from the initial mass beneath the body;\n"
        "center: mass of particles whose centers lie inside the interacting body; right: mass of all represented particles.\n"
        "The lower region uses the horizontal rectangle enclosing the body, from 0 to 0.10 m below its lowest point,\n"
        "clipped above the rigid tray floor. Both methods use the same recorded body pose "
        "and physical particle masses.",
        ha="center",
        fontsize=10,
    )
    save(figure, root, "body_mass")


def plot_region_counts(root, particles):
    """Expose changing reference populations behind the regional position errors."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    figure.subplots_adjust(bottom=0.30, top=0.91, wspace=0.30)
    for axis, key, color in zip(
        axes,
        ("near_contact_count", "moving_count"),
        ("#536a2a", "#213f2f"),
        strict=True,
    ):
        axis.plot(particles["time_s"], particles[key], color=color)
        axis.set(xlabel="Simulation time (s)", ylabel="Reference\nparticle count")
        axis.set_title(
            "Near-contact particles"
            if key == "near_contact_count"
            else "Moving particles"
        )
        axis.set_yscale("symlog", linthresh=1)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.15)
    figure.text(
        0.5,
        0.035,
        "Figure 6: Reference populations in one paired rollout. Left: particles within 0.05 m of the body surface;\n"
        "right: particles moving faster than 0.01 m/s, using position-difference velocities at 200 Hz.\n"
        "Membership is recomputed at each step. Zero count makes the corresponding Figure 1 particle-position error undefined.\n"
        "The count axis is linear from 0 to 1 and logarithmic above 1; no smoothing or interpolation is applied.",
        ha="center",
        fontsize=10,
    )
    save(figure, root, "particle_region_counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    root = args.evaluation
    meta = json.loads((root / "evaluation.json").read_text())
    particles, fields = (
        read_table(root / name)
        for name in ("particle_metrics.csv", "field_metrics.csv")
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "mathtext.fontset": "stix",
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(13.2, 11.1), constrained_layout=False)
    figure.subplots_adjust(hspace=0.42, wspace=0.26, top=0.95, bottom=0.27)
    configs = [
        ("global_rmse_mm", "Entire sand bed", NERD),
        ("near_contact_rmse_mm", "Near-contact particles", "#536a2a"),
        ("moving_rmse_mm", "Moving particles", "#213f2f"),
    ]
    for key, label, color in configs:
        axes[0, 0].plot(
            particles["time_s"], particles[key], label=label, color=color, linewidth=1.4
        )
    axes[0, 0].set_ylabel("Particle-position\nRMSE (mm)")
    axes[0, 0].legend(fontsize=9)
    if fields:
        surface_time = fields["time_s"] / meta["surface_metric_interval_s"]
        surface_mask = np.isclose(surface_time, np.round(surface_time))
        axes[0, 1].plot(
            fields["time_s"][surface_mask],
            fields.get("surface_mean_mm", np.full_like(fields["time_s"], np.nan))[
                surface_mask
            ],
            color=NERD,
            label="Mean",
        )
        axes[0, 1].plot(
            fields["time_s"][surface_mask],
            fields.get("surface_p95_mm", np.full_like(fields["time_s"], np.nan))[
                surface_mask
            ],
            color=NERD,
            linestyle="--",
            label="95th percentile",
        )
        axes[0, 1].set_ylabel("Symmetric surface\ndistance (mm)")
        axes[0, 1].legend(fontsize=9)
        coverage = axes[1, 1].twinx()
        coverage.plot(
            fields["time_s"],
            100 * fields["velocity_reference_mass_coverage"],
            color="#888888",
            linestyle=":",
            linewidth=1.1,
        )
        coverage.set_ylabel("Reference mass coverage (%)", fontsize=10, color="#666666")
        coverage.set_ylim(0, 105)
        for axis, key, label in (
            (axes[1, 0], "mass_discrepancy", "Material-mass\ndiscrepancy"),
            (axes[1, 1], "velocity_rmse_m_s", "Velocity-field\nRMSE (m/s)"),
        ):
            axis.plot(fields["time_s"], fields[key], color=NERD)
            axis.set_ylabel(label)
        for axis, suffix, label in (
            (axes[2, 0], "ke_j", "Particle kinetic energy (J)"),
            (axes[2, 1], "below_body_mass_kg", "Material mass\nbeneath body (kg)"),
        ):
            axis.plot(
                fields["time_s"],
                fields["reference_" + suffix],
                color=MPM,
                label="Implicit MPM",
            )
            axis.plot(
                fields["time_s"],
                fields["predicted_" + suffix],
                color=NERD,
                label="Particle NeRD",
            )
            axis.set_ylabel(label)
            axis.legend(fontsize=9)
    if not fields:
        for axis in list(axes.flat)[1:]:
            axis.text(
                0.5,
                0.5,
                "Unavailable: paired field grid\nexceeded its node limit",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=11,
            )
    for axis in axes.flat:
        axis.set_xlabel("Simulation time (s)")
        axis.grid(alpha=0.15)
    caption = (
        "Figure 1: One paired rollout with identical initial particles and recorded body motion.\n"
        "Top row: corresponding-particle position error, "
        "and sand-surface error;\n"
        "Middle row: physical-mass redistribution error; and material velocity-field error; Bottom row: particle kinetic energy;\n"
        "and sand mass in the body's horizontal bounding rectangle, "
        "within 0.10 m below its lowest point and above the floor.\n"
        "Near-contact reference distance ≤0.05 m; moving reference speed >0.01 m/s. "
        "Empty groups produce gaps; Figure 6 shows counts.\n"
        "Mass discrepancy: sum of absolute grid-mass differences divided by total reference mass (dimensionless).\n"
        "Velocity-field error: reference-mass-weighted velocity RMSE in cells occupied by both methods; dotted curve is their\n"
        "fraction of total reference mass. The surface percentile is over sampled distances, not repeated trials.\n"
        "The model advances its own particle history at 200 Hz. "
        "No displacement gate, particle projection, or force filter."
    )
    if meta["failure"]:
        caption += f"\nRollout failed at {meta['failure']['time_s']:.3f} s; only the completed interval is shown."
    figure.text(0.5, 0.035, caption, ha="center", fontsize=10)
    save(figure, root, "particle_evaluation")
    plot_mass_maps(root)
    plot_body_mass(root, fields)
    plot_region_counts(root, particles)
    names = ["particle_evaluation", "particle_region_counts"]
    if fields:
        names.append("body_mass")
    if (root / "mass_difference_maps.npz").exists():
        names.append("mass_difference")
    with h5py.File(root / "prediction.h5", "r") as data:
        if "force_reference" not in data:
            write_provenance(root, meta, names)
            return
        reference = data["force_reference"][:]
        for number, key, label, name in (
            (2, "force_predicted_particles", "Particle NeRD", "forces"),
            (
                3,
                "force_reference_particles",
                "Reconstruction using MPM particles",
                "force_reconstruction",
            ),
        ):
            predicted = data[key][:]
            t = np.arange(1, len(reference) + 1) * meta["native_dt_s"]
            support = meta["episode"]["id"].endswith("-static-support") and t[-1] > 3.0
            windows = [np.ones(len(t), dtype=bool)]
            if support:
                windows.append(t >= 3.0)
            fig, axes = plt.subplots(
                len(windows), 3, figsize=(14.4, 8.3 if support else 5.3), squeeze=False
            )
            fig.subplots_adjust(
                bottom=0.26 if support else 0.33, top=0.82, wspace=0.28, hspace=0.55
            )
            for row, window in enumerate(windows):
                for component, axis in enumerate(axes[row]):
                    axis.plot(
                        t[window],
                        reference[window, component],
                        color=MPM,
                        label="Implicit MPM",
                        linewidth=1.2,
                    )
                    axis.plot(
                        t[window],
                        predicted[window, component],
                        color=NERD,
                        label=label,
                        linewidth=1.2,
                    )
                    axis.set(
                        xlabel="Simulation time (s)",
                        ylabel=f"$F_{'xyz'[component]}$ (N)",
                    )
                    axis.set_title("Full recording" if row == 0 else "From 3 s onward")
                    axis.grid(alpha=0.15)
            fig.legend(
                *axes[0, 0].get_legend_handles_labels(),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=2,
            )
            fig.text(
                0.5,
                0.04,
                f"Figure {number}: One paired rollout; "
                "raw world-frame material-on-object forces at 200 Hz, +z upward.\n"
                "Left to right: Fx, Fy, Fz over the full recording.\n"
                + (
                    "Bottom row: Fx, Fy, Fz from 3 s onward, with expanded force scales to show sustained support.\n"
                    if support
                    else ""
                )
                + "Recorded MPM body motion and free velocities are imposed; estimates do not feed back to the body.\n"
                + (
                    "Particle NeRD uses recurrent particle predictions; direct rigid-tray forces are excluded."
                    if number == 2
                    else (
                        "The same estimator uses MPM particles and position-difference velocities; "
                        "no second neural model.\nDirect rigid-tray forces are excluded."
                    )
                ),
                ha="center",
                fontsize=10,
            )
            save(fig, root, name)
            names.append(name)
    write_provenance(root, meta, names)


if __name__ == "__main__":
    main()
