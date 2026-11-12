"""Plot measured object-benchmark results from public JSON without running simulation."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MPM, NERD, DIAGNOSTIC = "#193c64", "#248044", "#718170"
SHAPES = ("box", "sphere", "cylinder")
REFERENCES = (
    "box-static-support",
    "sphere-static-support",
    "cylinder-static-support",
)
CASE_LABELS = (
    "Box\n[Dyn: Drop &\nSupport 20s]",
    "Sphere\n[Dyn: Drop &\nSupport 20s]",
    "Cylinder\n[Dyn: Drop &\nSupport 20s]",
)
HOLD_LABELS = (
    "Box\n[Kin: Insert 1–2s,\nFixed Hold 2–23s]",
    "Sphere\n[Kin: Insert 1–2s,\nFixed Hold 2–23s]",
    "Cylinder\n[Kin: Insert 1–2s,\nFixed Hold 2–23s]",
)
PLOTS = {
    "test_particle_errors": (
        "1a",
        "By interacting object (Box, Sphere, Cylinder across 8 object motions)",
        (
            "Elementary objects (Box, Sphere, Cylinder), each aggregated across 20 trajectories: "
            "12 Kinematic (Horizontal, Vertical, Circular, Hold) and 8 Dynamic (Drop, Slide, Throw, Tumble). "
            "Rows show particle-position RMSE for the entire bed, near-contact particles (≤0.05 m), and moving particles (>0.01 m/s). "
            "Left column: time-mean over 0–6 s; right column: terminal state at 6 s."
        ),
    ),
    "test_particle_errors_by_motion": (
        "1b",
        "By motion type (Kinematic object motions, Dynamic object motions, and Unconfined sand free-fall)",
        (
            "Breakdown by motion across Kinematic object motions (Horizontal, Vertical, Circle, Hold; n=9 each across Box/Sphere/Cylinder), "
            "Dynamic object motions (Obj Drop, Slide, Throw, Tumble; n=6 each across Box/Sphere/Cylinder), "
            "and Unconfined Sand Free-Fall on flat ground (no rigid object; n=8)."
        ),
    ),
    "test_surface_errors": (
        "2a",
        "By interacting object (Box, Sphere, Cylinder across 8 object motions)",
        (
            "Elementary objects (Box, Sphere, Cylinder) across 12 Kinematic (Horizontal, Vertical, Circular, Hold) "
            "and 8 Dynamic (Drop, Slide, Throw, Tumble) trajectories per shape. "
            "Rows show mean and 95th-percentile symmetric surface distance on a 0.02 m measurement grid."
        ),
    ),
    "test_surface_errors_by_motion": (
        "2b",
        "By motion type (Kinematic object motions, Dynamic object motions, and Unconfined sand free-fall)",
        (
            "Symmetric sand-surface distance grouped by motion: Kinematic object motions (Horizontal, Vertical, Circle, Hold), "
            "Dynamic object motions (Obj Drop, Slide, Throw, Tumble), and Unconfined Sand Free-Fall on flat ground (no object)."
        ),
    ),
    "test_material_distribution": (
        "3a",
        "By interacting object (Box, Sphere, Cylinder across 8 object motions)",
        (
            "Elementary objects (Box, Sphere, Cylinder) across 12 Kinematic (Horizontal, Vertical, Circular, Hold) "
            "and 8 Dynamic (Drop, Slide, Throw, Tumble) trajectories per shape. "
            "First row: time-mean and terminal-state material-mass discrepancy. Second row: terminal mass beneath and inside the body; last panel: total represented mass."
        ),
    ),
    "test_material_distribution_by_motion": (
        "3b",
        "By motion type (Kinematic object motions, Dynamic object motions, and Unconfined sand free-fall)",
        (
            "Material-mass discrepancy and represented mass grouped by motion: Kinematic object motions (Horizontal, Vertical, Circle, Hold), "
            "Dynamic object motions (Obj Drop, Slide, Throw, Tumble), and Unconfined Sand Free-Fall on flat ground."
        ),
    ),
    "test_velocity_energy": (
        "4a",
        "By interacting object (Box, Sphere, Cylinder across 8 object motions)",
        (
            "Elementary objects (Box, Sphere, Cylinder) across 12 Kinematic (Horizontal, Vertical, Circular, Hold) "
            "and 8 Dynamic (Drop, Slide, Throw, Tumble) trajectories per shape. "
            "Rows show velocity-field RMSE, reference-mass coverage, and particle kinetic energy."
        ),
    ),
    "test_velocity_energy_by_motion": (
        "4b",
        "By motion type (Kinematic object motions, Dynamic object motions, and Unconfined sand free-fall)",
        (
            "Velocity-field RMSE, reference-mass coverage, and particle kinetic energy grouped by motion: "
            "Kinematic object motions (Horizontal, Vertical, Circle, Hold), Dynamic object motions (Obj Drop, Slide, Throw, Tumble), "
            "and Unconfined Sand Free-Fall on flat ground."
        ),
    ),
    "reference_force_errors": (
        5,
        "Per-body force RMSE and signed mean error (Dynamic Motion: 20 s Drop & Gravity Support)",
        (
            "Dynamic Motion: 2 kg object held until 1 s, then released to drop and settle on sand through 20 s. "
            "Columns show Fx, Fy, Fz. Top row: component RMSE; bottom row: signed mean error (estimate minus native MPM material force)."
        ),
    ),
    "reference_summary": (
        6,
        "Dedicated dynamic drop-and-support reference recordings (0–20 s)",
        (
            "Dynamic Motion: 20 s Drop & Gravity Support for Box, Sphere, and Cylinder. "
            "Left: terminal particle-position RMSE. Center: force-vector RMSE. Right: mean Fz during the final second (19–20 s)."
        ),
    ),
    "hold_summary": (
        7,
        "Spatio-temporal consistency (Kinematic Motion: 1–2 s Vertical Insertion + 2–23 s Fixed-Pose Hold)",
        (
            "Kinematic Motion: Object is inserted vertically (1–2 s) and held at an exactly fixed pose (2–23 s); measured relative to 3 s. "
            "Top row: mean change in world z at 23 s relative to 3 s, and late drift rate (13–23 s). "
            "Middle row: RMS particle drift and mean surface change. Bottom row: grid-mass redistribution and mass remaining beneath object."
        ),
    ),
    "hold_material": (
        8,
        "Material beneath fixed object and floor leakage (Kinematic Motion: 2–23 s Fixed-Pose Hold)",
        (
            "Kinematic Motion: 2–23 s Fixed-Pose Hold for Box, Sphere, and Cylinder. "
            "Top row: physical mass beneath the object at 3 and 23 s. Middle row: mass below the z=−0.30 m floor. Last panel: lowest particle z at 23 s."
        ),
    ),
}


def figure_html(stem, number=None, caption=None):
    if number is None:
        number, _, caption = PLOTS[stem]
    return (
        f'<figure class="result-figure" id="{stem}"><div class="plot-scroll" tabindex="0">'
        f'<img src="{stem}.svg" alt="{html.escape(caption)}" loading="lazy"></div>'
        f"<figcaption><strong>Figure {number}:</strong> {caption}</figcaption></figure>"
    )


def canvas(rows=2, cols=3, width=14.0):
    fig, axes = plt.subplots(rows, cols, figsize=(width, 4.1 * rows + 1.5), squeeze=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.22 if rows > 1 else 0.35,
        top=0.84 if rows == 1 else 0.90,
        hspace=0.62,
        wspace=0.40,
    )
    return fig, list(axes.flat)


def finish(fig, axes, root, stem, caption, number):
    import textwrap

    for ax in axes:
        if ax.axison:
            ax.grid(axis="y", alpha=0.15)
            ax.tick_params(axis="x", labelsize=9.5)
    text = textwrap.fill(f"Figure {number}: {caption}", width=155)
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=9)
    paths = []
    for ext in ("svg", "png", "pdf"):
        path = root / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def distribution(ax, records, series, ylabel):
    """Plot ONLY elementary rigid objects (Box, Sphere, Cylinder) — never mix free-fall motion here."""
    obj_records = [r for r in records if r["shape"] in SHAPES]
    offsets = np.linspace(-0.17, 0.17, len(series)) if len(series) > 1 else [0]
    counts = []
    for j, (key, label, color) in enumerate(series):
        groups = [[r["metrics"].get(key) for r in obj_records if r["shape"] == shape] for shape in SHAPES]
        groups = [np.asarray([v for v in g if v is not None and np.isfinite(v)]) for g in groups]
        positions = np.arange(3) + offsets[j]
        counts.append([len(g) for g in groups])
        boxes = ax.boxplot(
            groups,
            positions=positions,
            widths=0.22 if len(series) > 1 else 0.42,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": color,
                "markeredgecolor": "white",
                "markersize": 5,
            },
            medianprops={"color": "#303030"},
        )
        for box in boxes["boxes"]:
            box.set(facecolor=color, alpha=0.22)
        for x, values in zip(positions, groups, strict=True):
            jitter = np.random.default_rng(42).uniform(-0.065, 0.065, len(values))
            ax.scatter(x + jitter, values, s=12, color=color, alpha=0.65, zorder=3)
        ax.plot([], [], color=color, marker="D", linestyle="none", label=label)
    ticks = [
        f"{shape.title()}\n(12 Kin + 8 Dyn, n={min(c[i] for c in counts)})"
        for i, shape in enumerate(SHAPES)
    ]
    ax.set(xticks=np.arange(3), xticklabels=ticks, ylabel=ylabel, xlim=(-0.6, 2.6))
    if len(series) > 1:
        ax.legend(fontsize=9, loc="best")


MOTION_SPECS = (
    ("horizontal", "prescribed", "Horizontal\n[Kin]"),
    ("vertical", "prescribed", "Vertical\n[Kin]"),
    ("circle", "prescribed", "Circle\n[Kin]"),
    ("hold", "prescribed", "Hold\n[Kin]"),
    ("drop", "dynamic_obj", "Obj Drop\n[Dyn]"),
    ("slide", "dynamic_obj", "Slide\n[Dyn]"),
    ("throw", "dynamic_obj", "Throw\n[Dyn]"),
    ("tumble", "dynamic_obj", "Tumble\n[Dyn]"),
    ("drop", "freefall", "Free-fall\n[Sand]"),
)


def _match_motion_group(r, m_name, m_cat):
    if m_cat == "freefall":
        return r["shape"] == "freefall"
    if m_cat == "prescribed":
        return r["shape"] in SHAPES and r["motion_mode"] == "prescribed" and r["motion"] == m_name
    return r["shape"] in SHAPES and r["motion_mode"] == "dynamic" and r["motion"] == m_name


def distribution_by_motion(ax, records, series, ylabel):
    """Plot distribution across individual motions (Kinematic, Dynamic object, and Free-fall sand)."""
    active_specs = []
    for m_name, m_cat, m_label in MOTION_SPECS:
        if any(
            _match_motion_group(r, m_name, m_cat)
            and any(r["metrics"].get(k) is not None and np.isfinite(r["metrics"].get(k)) for k, _, _ in series)
            for r in records
        ):
            active_specs.append((m_name, m_cat, m_label))
    n_m = len(active_specs)
    offsets = np.linspace(-0.17, 0.17, len(series)) if len(series) > 1 else [0]
    counts = []
    for j, (key, label, color) in enumerate(series):
        groups = [
            [r["metrics"].get(key) for r in records if _match_motion_group(r, m_name, m_cat)]
            for m_name, m_cat, _ in active_specs
        ]
        groups = [np.asarray([v for v in g if v is not None and np.isfinite(v)]) for g in groups]
        positions = np.arange(n_m) + offsets[j]
        counts.append([len(g) for g in groups])
        boxes = ax.boxplot(
            groups,
            positions=positions,
            widths=0.22 if len(series) > 1 else 0.42,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": color,
                "markeredgecolor": "white",
                "markersize": 4.5,
            },
            medianprops={"color": "#303030"},
        )
        for box in boxes["boxes"]:
            box.set(facecolor=color, alpha=0.22)
        for x, values in zip(positions, groups, strict=True):
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(values))
            ax.scatter(x + jitter, values, s=10, color=color, alpha=0.65, zorder=3)
        ax.plot([], [], color=color, marker="D", linestyle="none", label=label)
    # Draw vertical dashed separators between Kinematic (0..3), Dynamic Object (4..7), and Free-fall Sand (8)
    cats = [c for _, c, _ in active_specs]
    for idx in range(1, len(cats)):
        if cats[idx] != cats[idx - 1]:
            ax.axvline(idx - 0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)
    ticks = [
        f"{m_label}\n(n={min(c[i] for c in counts)})"
        for i, (_, _, m_label) in enumerate(active_specs)
    ]
    ax.set(xticks=np.arange(n_m), xticklabels=ticks, ylabel=ylabel, xlim=(-0.6, n_m - 0.4))
    ax.tick_params(axis="x", labelsize=8.0)
    if len(series) > 1:
        ax.legend(fontsize=8.5, loc="best")


def paired_bars(ax, values, labels, ylabel, *, methods=None, std=None):
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    methods = methods or (("Implicit MPM", MPM), (model_label, NERD))
    x = np.arange(len(labels))
    width = 0.72 / len(methods)
    for j, (label, color) in enumerate(methods):
        ax.bar(
            x + (j - (len(methods) - 1) / 2) * width,
            values[j],
            width,
            color=color,
            alpha=0.92,
            label=label,
            yerr=None if std is None else std[j],
            capsize=3,
        )
    for j, (_, color) in enumerate(methods):
        zero = np.asarray(values[j]) == 0
        ax.scatter(
            (x + (j - (len(methods) - 1) / 2) * width)[zero],
            np.zeros(zero.sum()),
            s=24,
            facecolors="white",
            edgecolors=color,
            zorder=5,
            clip_on=False,
        )
    ax.set(xticks=x, xticklabels=labels, ylabel=ylabel)
    ax.axhline(0, color="#555555", linewidth=0.6)


def generate(root):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )
    inputs, outputs = {}, []

    def read(path):
        inputs[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    report = read(root / "test_report.json")
    training = read(root / "training_provenance.json")
    step = training["evaluated_training_state"]["checkpoint_step"]
    assert training["checkpoint_sha256"] == report["checkpoint_sha256"]

    def save(fig, axes, directory, stem, caption, number):
        context = f"Frozen object-model checkpoint {step:,}. "
        if directory != root:
            meta = read(directory / "evaluation_settings.json")
            if directory.name in REFERENCES:
                case_name = CASE_LABELS[REFERENCES.index(directory.name)].replace("\n", " ")
            else:
                case_name = meta["episode"]["shape"].title() + ": fixed-object hold"
            duration = meta["completed_steps"] * meta["native_dt_s"]
            context += f"{case_name}; {duration:g} s recording; one recorded-motion replay. "
        return finish(fig, axes, directory, stem, context + caption, number)

    records = report["episodes"]
    checkpoint = report["checkpoint_sha256"]
    layouts = {
        "test_particle_errors": [
            (f"{region}_rmse_mm_{stat}", f"{label}\nposition RMSE (mm)")
            for region, label in (
                ("global", "Whole bed"),
                ("near_contact", "Near-contact particles"),
                ("moving", "Moving particles"),
            )
            for stat in ("time_mean", "end")
        ],
        "test_surface_errors": [
            (
                f"surface_{stat}_mm_{when}",
                f"{label} symmetric\nsurface distance (mm)",
            )
            for stat, label in (("mean", "Mean"), ("p95", "95th-percentile"))
            for when in ("time_mean", "end")
        ],
        "test_material_distribution": [
            ("mass_discrepancy_time_mean", "Time-mean material-mass\ndiscrepancy"),
            ("mass_discrepancy_end", "Final material-mass\ndiscrepancy"),
            ("predicted_below_body_mass_kg_end", "Terminal mass\nbeneath body (kg)"),
            ("predicted_inside_body_mass_kg_end", "Terminal mass\ninside body (kg)"),
            ("predicted_mass_kg_end", "Terminal total\nrepresented mass (kg)"),
        ],
        "test_velocity_energy": [
            ("velocity_rmse_m_s_time_mean", "Time-mean velocity-field\nRMSE (m/s)"),
            ("velocity_rmse_m_s_end", "Final velocity-field\nRMSE (m/s)"),
            (
                "velocity_reference_mass_coverage_time_mean",
                "Time-mean reference-mass coverage",
            ),
            ("velocity_reference_mass_coverage_end", "Final reference-mass coverage"),
            ("predicted_ke_j_time_mean", "Time-mean kinetic energy (J)"),
            ("predicted_ke_j_end", "Final kinetic energy (J)"),
        ],
    }
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL") or (
        "Particle World"
        if "cconv_warp.py" in report.get("network_source_sha256", {})
        else ("Particle NeRD-Local" if len(report.get("network_source_sha256", {})) == 2 else "Particle NeRD")
    )
    for stem, keys in layouts.items():
        # Group A: By Interacting Object (Box, Sphere, Cylinder only)
        fig, axes = canvas(3, 2) if len(keys) > 4 else canvas(2, 2)
        for index, (key, label) in enumerate(keys):
            series = [(key, model_label, NERD)]
            if key.startswith("predicted_"):
                series.insert(0, (key.replace("predicted_", "reference_"), "Implicit MPM", MPM))
            distribution(axes[index], records, series, label)
        for ax in axes[len(keys) :]:
            ax.set_axis_off()
        if stem != "test_material_distribution":
            axes[0].set_title("Time-mean (0–6 s)", pad=14)
            axes[1].set_title("Terminal state (6 s)", pad=14)
        fig.suptitle(
            "By Interacting Object (Box, Sphere, Cylinder) — Motions: Kinematic (Horizontal, Vertical, Circle, Hold) & Dynamic (Drop, Slide, Throw, Tumble)",
            fontsize=11,
            fontweight="bold",
            y=0.97,
        )
        number, _, caption = PLOTS[stem]
        outputs.extend(save(fig, axes, root, stem, caption, number))

        # Group B: By Motion (Horizontal, Vertical, Circle, Hold | Obj Drop, Slide, Throw, Tumble | Free-fall Sand)
        motion_stem = f"{stem}_by_motion"
        fig_m, axes_m = canvas(3, 2, width=16.0) if len(keys) > 4 else canvas(2, 2, width=16.0)
        for index, (key, label) in enumerate(keys):
            series = [(key, model_label, NERD)]
            if key.startswith("predicted_"):
                series.insert(0, (key.replace("predicted_", "reference_"), "Implicit MPM", MPM))
            distribution_by_motion(axes_m[index], records, series, label)
        for ax in axes_m[len(keys) :]:
            ax.set_axis_off()
        if stem != "test_material_distribution":
            axes_m[0].set_title("Time-mean (0–6 s)", pad=14)
            axes_m[1].set_title("Terminal state (6 s)", pad=14)
        fig_m.suptitle(
            "By Motion Type — Kinematic Object Motions (Horizontal, Vertical, Circle, Hold) | Dynamic Object Motions (Obj Drop, Slide, Throw, Tumble) | Sand Free-Fall (Flat Ground)",
            fontsize=11,
            fontweight="bold",
            y=0.97,
        )
        num_m, _, cap_m = PLOTS[motion_stem]
        outputs.extend(save(fig_m, axes_m, root, motion_stem, cap_m, num_m))

    metadata = [read(root / name / "evaluation_settings.json") for name in REFERENCES]
    windows = [read(root / name / "force_windows.json") for name in REFERENCES]
    assert all(m["checkpoint_sha256"] == checkpoint for m in metadata)
    estimation = ((model_label, NERD), ("Estimator with MPM particles", DIAGNOSTIC))
    fig, axes = canvas(2, 3)
    for row, metric in enumerate(("component_rmse_n", "component_bias_n")):
        for i in range(3):
            values = [
                [m[source][metric][i] for m in metadata]
                for source in ("force_predicted_particles", "force_reference_particles")
            ]
            paired_bars(
                axes[row * 3 + i],
                values,
                CASE_LABELS,
                f"$F_{'xyz'[i]}$ {'RMSE' if row == 0 else 'signed bias'} (N)",
                methods=estimation,
            )
    fig.suptitle(
        "Dynamic Motion: 20 s Vertical Drop & Gravity Support (2 kg Object Released at t = 1 s)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2)
    outputs.extend(
        save(
            fig,
            axes,
            root,
            "reference_force_errors",
            PLOTS["reference_force_errors"][2],
            PLOTS["reference_force_errors"][0],
        )
    )
    fig, axes = canvas(1, 3)
    paired_bars(
        axes[0],
        [[m["final_position_rmse_mm"] for m in metadata]],
        CASE_LABELS,
        "Final position RMSE (mm)",
        methods=((model_label, NERD),),
    )
    paired_bars(
        axes[1],
        [[m[source]["vector_rmse_n"] for m in metadata] for source in ("force_predicted_particles", "force_reference_particles")],
        CASE_LABELS,
        "Force-vector RMSE (N)",
        methods=estimation,
    )
    axes[1].legend(fontsize=8)
    paired_bars(
        axes[2],
        [[w[source][-1]["mean_fx_fy_fz_n"][2] for w in windows] for source in ("reference", "predicted_particles")],
        CASE_LABELS,
        "Final-second $F_z$ (N)",
        std=[
            [w[source][-1]["frame_std_fx_fy_fz_n"][2] for w in windows]
            for source in ("reference", "predicted_particles")
        ],
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Dynamic Motion: 20 s Vertical Drop & Gravity Support Summary",
        fontsize=11.5,
        fontweight="bold",
        y=0.96,
    )
    outputs.extend(
        save(
            fig,
            axes,
            root,
            "reference_summary",
            PLOTS["reference_summary"][2],
            PLOTS["reference_summary"][0],
        )
    )

    summaries = read(root / "consistency_results.json")
    assert all(s["checkpoint_sha256"] == checkpoint for s in summaries.values())
    for stem, metrics in (
        (
            "hold_summary",
            (
                ("final.mean_vertical_change_mm", "Mean vertical drift (mm)"),
                ("late_vertical_drift_rate_mm_s", "Late vertical drift rate (mm/s)"),
                ("final.rms_displacement_mm", "RMS particle drift (mm)"),
                ("final.surface_mean_mm", "Mean surface change (mm)"),
                (
                    "final.temporal_mass_discrepancy",
                    "Normalized grid-mass redistribution",
                ),
                ("final_below_body_mass_fraction", "Mass remaining beneath object (%)"),
            ),
        ),
        (
            "hold_material",
            (
                ("initial_below_body_mass_kg", "Mass beneath object at 3 s (kg)"),
                ("final.below_body_mass_kg", "Mass beneath object at 23 s (kg)"),
                ("final.below_floor_mass_percent", "Mass below floor (%)"),
                ("final.below_floor_1cm_mass_percent", "Mass >1 cm below floor (%)"),
                ("final.minimum_particle_z_m", "Lowest particle z (m)"),
            ),
        ),
    ):
        fig, axes = canvas(3, 2)
        for i, (key, label) in enumerate(metrics):
            values = []
            for method in ("reference", "predicted"):
                group = []
                for shape in SHAPES:
                    value = summaries[f"{shape}-fixed-hold"][method]
                    for part in key.split("."):
                        value = value[part]
                    group.append(value * 100 if key == "final_below_body_mass_fraction" else value)
                values.append(group)
            paired_bars(axes[i], values, HOLD_LABELS, label)
            if key.endswith("minimum_particle_z_m"):
                axes[i].set_yscale("symlog", linthresh=1)
                axes[i].set_yticks([-100, -10, -1, 0], labels=["−100", "−10", "−1", "0"])
        for ax in axes[len(metrics) :]:
            ax.set_axis_off()
        fig.suptitle(
            "Kinematic Motion: Vertical Insertion (1–2 s) + Stationary Fixed-Pose Hold (2–23 s)",
            fontsize=11.5,
            fontweight="bold",
            y=0.98,
        )
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2)
        number, _, caption = PLOTS[stem]
        outputs.extend(save(fig, axes, root, stem, caption, number))

    for case in REFERENCES:
        folder = root / case
        meta = read(folder / "evaluation_settings.json")
        force_windows = read(folder / "force_windows.json")
        assert meta["checkpoint_sha256"] == checkpoint
        fig, axes = canvas(1, 2)
        for ax, key, label in zip(
            axes,
            ("component_rmse_n", "component_bias_n"),
            ("Force RMSE (N)", "Signed force error (N)"),
            strict=True,
        ):
            paired_bars(
                ax,
                [meta[k][key] for k in ("force_predicted_particles", "force_reference_particles")],
                ("Fx", "Fy", "Fz"),
                label,
                methods=estimation,
            )
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2)
        caption = "Left: component RMSE; right: signed mean error (estimate minus native MPM material force) over the full recording. World-frame forces, +z upward. Raw 200 Hz forces, no filtering or time shifts. The gray diagnostic uses the same estimator with MPM particles, not a second neural model."
        outputs.extend(
            save(
                fig,
                axes,
                folder,
                "force_error_summary",
                caption,
                8 if case.endswith("fixed-hold") else 7,
            )
        )
        fig, axes = canvas(1, 3)
        end = meta["completed_steps"] * meta["native_dt_s"]
        for source, label, color in (
            ("reference", "Implicit MPM", MPM),
            ("predicted_particles", model_label, NERD),
            ("reference_particles", "Estimator with MPM particles", DIAGNOSTIC),
        ):
            rows = [r for r in force_windows[source] if r["start_exclusive_s"] in {1.0, 3.0, end - 1}]
            offset = {
                "reference": -0.18,
                "predicted_particles": 0.0,
                "reference_particles": 0.18,
            }[source]
            centers = np.arange(len(rows)) + offset
            for i, ax in enumerate(axes):
                ax.errorbar(
                    centers,
                    [r["mean_fx_fy_fz_n"][i] for r in rows],
                    yerr=[r["frame_std_fx_fy_fz_n"][i] for r in rows],
                    fmt={
                        "reference": "s",
                        "predicted_particles": "o",
                        "reference_particles": "x",
                    }[source],
                    markersize=7 if source == "reference" else 5,
                    markerfacecolor="none" if source == "reference" else color,
                    capsize=4,
                    color=color,
                    label=label,
                )
                ax.set(
                    xlabel="Simulation-time window (s)",
                    ylabel=f"$F_{'xyz'[i]}$ (N)",
                    xticks=np.arange(len(rows)),
                    xticklabels=[f"({r['start_exclusive_s']:g}, {r['end_inclusive_s']:g}]" for r in rows],
                    xlim=(-0.5, len(rows) - 0.5),
                )
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            ncol=3,
            fontsize=10,
        )
        caption = "Left to right: Fx, Fy, Fz. Means and population standard deviations over the labeled one-second windows. Methods are offset horizontally within each window; force samples are not shifted in time. All raw 200 Hz samples are used. Whiskers show temporal variability in one recording, not repeated-trial uncertainty. Full force traces appear above."
        outputs.extend(
            save(
                fig,
                axes,
                folder,
                "force_window_summary",
                caption,
                9 if case.endswith("fixed-hold") else 8,
            )
        )
    provenance = {
        "checkpoint_sha256": checkpoint,
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs_sha256": inputs,
        "figures_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        "statement": "Presentation-only transformation of existing measured summaries. No new simulation, training, filtering, or removal of trajectories.",
    }
    (root / "comparison_plot_provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    generate(parser.parse_args().report)
