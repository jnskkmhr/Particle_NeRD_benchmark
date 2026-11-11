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
    "box-horizontal-flat",
    "box-static-support",
    "sphere-static-support",
    "cylinder-static-support",
)
CASE_LABELS = (
    "Box\nhorizontal\n(6 s)",
    "Box\nsupport\n(20 s)",
    "Sphere\nsupport\n(20 s)",
    "Cylinder\nsupport\n(20 s)",
)
DISTRIBUTION = (
    "Each dot is one trajectory summary; boxes show the median and interquartile range, "
    "whiskers extend to 1.5 IQR, diamonds mark means. Twenty trajectories per shape unless labeled otherwise. "
    "Time means use available samples; empty particle regions are not assigned zero error."
)
PLOTS = {
    "test_particle_errors": (
        1,
        "Particle-position RMSE, globally and by region",
        (
            "Rows show particle-position RMSE for the entire bed, near-contact "
            "particles, and moving particles. Left column: time-mean over 0–6 s; "
            "right column: terminal state at 6 s. NeRD errors use corresponding MPM "
            "particles. Near-contact reference distance ≤0.05 m; moving reference "
            "speed >0.01 m/s. One terminal-state moving region is empty and remains "
            "unavailable. Each dot is one trajectory summary; boxes show the median "
            "and interquartile range, whiskers extend to 1.5 IQR, diamonds mark "
            "means. Twenty trajectories per shape unless labeled otherwise. Time "
            "means use available samples; empty particle regions are not assigned "
            "zero error."
        ),
    ),
    "test_surface_errors": (
        2,
        "Mean and 95th-percentile symmetric surface distance",
        (
            "Rows show the mean and 95th-percentile symmetric surface distance. Left "
            "column: time-mean over 0–6 s; right column: terminal state at 6 s. "
            "Distances use area-uniform surface samples without alignment. The 95th "
            "percentile is over surface distances within a frame, not across "
            "trajectories. Sand surfaces use a 0.02 m measurement grid. Each dot is "
            "one trajectory summary; boxes show the median and interquartile range, "
            "whiskers extend to 1.5 IQR, diamonds mark means. Twenty trajectories per "
            "shape unless labeled otherwise. Time means use available samples; empty "
            "particle regions are not assigned zero error."
        ),
    ),
    "test_material_distribution": (
        3,
        "Material-mass discrepancy on a common grid",
        (
            "The first row shows time-mean and terminal-state material-mass "
            "discrepancy: sum of absolute cell-mass differences divided by "
            "total physical mass. The second row shows terminal-state mass "
            "beneath and inside the body; the last panel shows total "
            "represented mass. Terminal state is 6 s. Equal total mass does "
            "not imply correct distribution or confinement to the tray. Each "
            "dot is one trajectory summary; boxes show the median and "
            "interquartile range, whiskers extend to 1.5 IQR, diamonds mark "
            "means. Twenty trajectories per shape unless labeled otherwise. "
            "Time means use available samples; empty particle regions are not "
            "assigned zero error."
        ),
    ),
    "test_velocity_energy": (
        4,
        "Velocity fields and kinetic energy",
        (
            "Rows show velocity-field RMSE, the fraction of reference mass covered "
            "by shared occupied cells, and particle kinetic energy. Left column: "
            "time-mean over 0–6 s; right column: terminal state at 6 s. Velocities "
            "use matched consecutive-position differences. Each dot is one "
            "trajectory summary; boxes show the median and interquartile range, "
            "whiskers extend to 1.5 IQR, diamonds mark means. Twenty trajectories "
            "per shape unless labeled otherwise. Time means use available samples; "
            "empty particle regions are not assigned zero error."
        ),
    ),
    "reference_force_errors": (
        5,
        "Per-body force RMSE and signed mean error for Fx, Fy, Fz",
        (
            "Columns show Fx, Fy, Fz. Top row: component RMSE; bottom row: signed "
            "mean error (estimate minus native MPM material force). Direct "
            "rigid-tray force is excluded. Each case is one raw 200 Hz recording. "
            "The gray diagnostic supplies MPM particles to the same estimator. No "
            "force reference exists for the 60-test dataset."
        ),
    ),
    "reference_summary": (
        6,
        "Dedicated reference recordings",
        (
            "Left: terminal particle-position RMSE. Center: force-vector RMSE without "
            "division by three. Right: mean Fz during the final second, with temporal "
            "population standard deviation. Open circles mark measured zeros. Whiskers "
            "describe temporal variation in one recording, not repeated-run "
            "uncertainty. Horizontal motion lasts 6 s; support tests last 20 s. Body "
            "poses are replayed from MPM and do not respond to estimated force."
        ),
    ),
    "hold_summary": (
        7,
        "Spatio-temporal consistency under a stationary object",
        (
            "One fixed-object run per shape; open circles mark measured zeros. Top row: "
            "mass-weighted mean change in world z at 23 s relative to 3 s, and its "
            "least-squares rate over 13–23 s. Middle row: RMS particle drift and mean "
            "material-isosurface change. Bottom row: normalized absolute grid-mass "
            "redistribution and mass remaining beneath the object relative to 3 s. Negative "
            "z means downward. Each solver is compared with its own 3 s state without reset. "
            "One escaped MPM particle dominates box/cylinder RMS (>99% of squared "
            "displacement); this does not describe bulk-bed drift."
        ),
    ),
    "hold_material": (
        8,
        "Material beneath the fixed object and floor leakage",
        (
            "Open circles mark measured zeros. Top row: physical mass beneath the object at "
            "3 and 23 s. The region is its world x/y bounding rectangle extending 0.10 m "
            "below the lowest body point, clipped above the floor. Middle row: mass with "
            "centers below the z=−0.30 m floor (1 μm tolerance), and mass more than 1 cm "
            "below it. Last panel: lowest particle-center z at 23 s, with symmetric-log "
            "scale (linear within ±1 m). Percentages use total represented physical mass. "
            "One paired run per shape; all escaped particles are retained. MPM also settles "
            "and leaks."
        ),
    ),
}


def figure_html(stem, number=None, caption=None):
    if number is None:
        number, _, caption = PLOTS[stem]
    return (
        f'<figure class="result-figure" id="{stem}"><div class="plot-scroll" tabindex="0">'
        f'<a href="{stem}.svg"><img src="{stem}.svg" alt="{html.escape(caption)}" loading="lazy"></a></div>'
        f"<figcaption><strong>Figure {number}:</strong> {caption} "
        f'<a href="{stem}.png">PNG</a> · <a href="{stem}.pdf">PDF</a></figcaption></figure>'
    )


def canvas(rows=2, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.1 * rows + 1.5), squeeze=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.22 if rows > 1 else 0.35,
        top=0.84 if rows == 1 else 0.91,
        hspace=0.58,
        wspace=0.40,
    )
    return fig, list(axes.flat)


def finish(fig, axes, root, stem, caption, number):
    import textwrap

    for ax in axes:
        if ax.axison:
            ax.grid(axis="y", alpha=0.15)
            ax.tick_params(axis="x", labelsize=10)
    text = textwrap.fill(f"Figure {number}: {caption}", width=155)
    fig.text(0.5, 0.025, text, ha="center", va="bottom", fontsize=9)
    paths = []
    for ext in ("svg", "png", "pdf"):
        path = root / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def distribution(ax, records, series, ylabel):
    """Draw every available per-trajectory scalar without clipping or imputing missing regions."""
    offsets = np.linspace(-0.17, 0.17, len(series)) if len(series) > 1 else [0]
    counts = []
    for j, (key, label, color) in enumerate(series):
        groups = [
            [r["metrics"].get(key) for r in records if r["shape"] == shape]
            for shape in SHAPES
        ]
        groups = [
            np.asarray([v for v in g if v is not None and np.isfinite(v)])
            for g in groups
        ]
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
        shape.title() + f"\n(n={min(c[i] for c in counts)})"
        for i, shape in enumerate(SHAPES)
    ]
    ax.set(xticks=np.arange(3), xticklabels=ticks, ylabel=ylabel, xlim=(-0.6, 2.6))
    if len(series) > 1:
        ax.legend(fontsize=9, loc="best")


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
        inputs[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
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
                case_name = CASE_LABELS[REFERENCES.index(directory.name)].replace(
                    "\n", " "
                )
            else:
                case_name = meta["episode"]["shape"].title() + ": fixed-object hold"
            duration = meta["completed_steps"] * meta["native_dt_s"]
            context += (
                f"{case_name}; {duration:g} s recording; one recorded-motion replay. "
            )
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
        fig, axes = canvas(3, 2) if len(keys) > 4 else canvas(2, 2)
        for index, (key, label) in enumerate(keys):
            series = [(key, model_label, NERD)]
            if key.startswith("predicted_"):
                series.insert(
                    0, (key.replace("predicted_", "reference_"), "Implicit MPM", MPM)
                )
            distribution(axes[index], records, series, label)
        for ax in axes[len(keys) :]:
            ax.set_axis_off()
        if stem != "test_material_distribution":
            axes[0].set_title("Time-mean (0–6 s)", pad=14)
            axes[1].set_title("Terminal state (6 s)", pad=14)
        number, _, caption = PLOTS[stem]
        outputs.extend(save(fig, axes, root, stem, caption, number))

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
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2)
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
        [
            [m[source]["vector_rmse_n"] for m in metadata]
            for source in ("force_predicted_particles", "force_reference_particles")
        ],
        CASE_LABELS,
        "Force-vector RMSE (N)",
        methods=estimation,
    )
    axes[1].legend(fontsize=8)
    paired_bars(
        axes[2],
        [
            [w[source][-1]["mean_fx_fy_fz_n"][2] for w in windows]
            for source in ("reference", "predicted_particles")
        ],
        CASE_LABELS,
        "Final-second $F_z$ (N)",
        std=[
            [w[source][-1]["frame_std_fx_fy_fz_n"][2] for w in windows]
            for source in ("reference", "predicted_particles")
        ],
    )
    axes[2].legend(fontsize=8)
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
                    group.append(
                        value * 100
                        if key == "final_below_body_mass_fraction"
                        else value
                    )
                values.append(group)
            paired_bars(axes[i], values, [s.title() for s in SHAPES], label)
            if key.endswith("minimum_particle_z_m"):
                axes[i].set_yscale("symlog", linthresh=1)
                axes[i].set_yticks(
                    [-100, -10, -1, 0], labels=["−100", "−10", "−1", "0"]
                )
        for ax in axes[len(metrics) :]:
            ax.set_axis_off()
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2)
        number, _, caption = PLOTS[stem]
        outputs.extend(save(fig, axes, root, stem, caption, number))

    for case in (*REFERENCES, *(f"{s}-fixed-hold" for s in SHAPES)):
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
                [
                    meta[k][key]
                    for k in ("force_predicted_particles", "force_reference_particles")
                ],
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
            rows = [
                r
                for r in force_windows[source]
                if r["start_exclusive_s"] in {1.0, 3.0, end - 1}
            ]
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
                    xticklabels=[
                        f"({r['start_exclusive_s']:g}, {r['end_inclusive_s']:g}]"
                        for r in rows
                    ],
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
        "figures_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in outputs
        },
        "statement": "Presentation-only transformation of existing measured summaries. No new simulation, training, filtering, or removal of trajectories.",
    }
    (root / "comparison_plot_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    generate(parser.parse_args().report)
