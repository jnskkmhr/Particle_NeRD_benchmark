# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Regenerate Figure 7 from public scalar consistency_series.json; no private states required."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root, data):
    """Plot both methods relative to their own state at the same measurement start."""
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 2, figsize=(13.2, 11.1))
    figure.subplots_adjust(top=0.95, bottom=0.23, wspace=0.27, hspace=0.42)
    keys = (
        ("particles", "rms_displacement_mm", "RMS particle drift (mm)"),
        ("particles", "mean_vertical_change_mm", "Mean vertical drift (mm)"),
        ("fields", "surface_mean_mm", "Sand-surface change (mm)"),
        ("fields", "temporal_mass_discrepancy", "Normalized grid-mass redistribution"),
        ("fields", "below_body_mass_kg", "Sand mass beneath object (kg)"),
        ("particles", "below_floor_mass_percent", "Mass below rigid tray floor (%)"),
    )
    for method, label, color in (("reference", "Implicit MPM", "#193c64"), ("predicted", "Particle NeRD", "#248044")):
        for axis, (kind, key, ylabel) in zip(axes.flat, keys, strict=True):
            rows = data[method][kind]
            selected = [r for r in rows if key in r]
            axis.plot(
                [r["time_s"] for r in selected], [r[key] for r in selected], label=label, color=color, linewidth=1.3
            )
            axis.set_ylabel(ylabel)
            if key == "surface_mean_mm":
                axis.plot(
                    [r["time_s"] for r in selected],
                    [r["surface_p95_mm"] for r in selected],
                    color=color,
                    linestyle="--",
                    linewidth=1.1,
                )
    for index, axis in enumerate(axes.flat):
        axis.set_title(f"({chr(97 + index)})", loc="left")
        axis.set_xlabel("Simulation time (s)")
        axis.grid(alpha=0.15)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    figure.text(
        0.5,
        0.025,
        "Figure 7: Fixed-object hold, measured at 3–23 s. Both methods start from identical particles at 0 s.\n"
        "(a) Mass-weighted RMS drift from each method's own 3 s state; (b) "
        "mass-weighted mean vertical drift (negative down).\n"
        "(c) Distance to that method's 3 s material isosurface: solid mean, dashed "
        "95th percentile (includes sides/bottom).\n"
        "(d) Sum of absolute grid-cell mass changes from 3 s, divided by total "
        "physical mass; dimensionless, range 0–2.\n"
        "(e) Absolute mass in the rectangle enclosing the object's world x/y projection, extending 0.10 m below its\n"
        "lowest z and clipped above the tray floor at z = −0.30 m. (f) Absolute mass "
        "fraction with centers below that floor.\n"
        "The object stops at 2 s. The 3 s anchor does not reset particles or assert "
        "equilibrium. One paired run per shape.\n"
        "No alignment, gate, projection, filtering, or particle removal; MPM settling and leakage remain included.",
        ha="center",
        fontsize=10,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(root / f"spatiotemporal_consistency.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot(args.output, json.loads((args.case / "consistency_series.json").read_text()))
