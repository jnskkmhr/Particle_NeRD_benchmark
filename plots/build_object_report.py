"""Build public object-model reports from completed, matching evaluation artifacts."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

CASES = {
    "box-horizontal-flat": "Box: horizontal motion on flat sand",
    "box-static-support": "Box: 20-second support test",
    "sphere-static-support": "Sphere: 20-second support test",
    "cylinder-static-support": "Cylinder: 20-second support test",
}
FIGURES = (
    (
        1,
        "particle_evaluation",
        "One paired rollout. (a) Particle-position RMSE for the whole bed, near-contact particles and moving particles, with a whole-bed unchanged-initial-position baseline; empty reference groups produce gaps (counts in Figure 6). (b) Symmetric sand-surface distance; (c) sum of absolute grid-mass differences divided by total reference mass; (d) reference-mass-weighted velocity RMSE in cells occupied by both methods, with their fraction of total reference mass; (e) particle kinetic energy; (f) material mass beneath the interacting body.",
    ),
    (
        2,
        "forces",
        "Raw material force on the interacting body: (a) Fx; (b) Fy; (c) Fz. Particle NeRD uses its recurrent particle predictions. Recorded MPM body motion and free velocities are imposed; estimated forces do not feed back to body motion. Direct rigid-tray forces are excluded.",
    ),
    (
        3,
        "force_reconstruction",
        "Force-reconstruction diagnostic: (a) Fx; (b) Fy; (c) Fz. Recorded MPM particles replace predicted particles in the same force estimator. This input substitution isolates the estimator; it is not a second neural model. Recorded MPM body motion and free velocities are imposed; estimated forces do not feed back to body motion.",
    ),
    (
        4,
        "mass_difference",
        "Predicted minus reference physical mass per projected grid column. (a–c) xy maps at the times printed above the panels; (d–f) xz maps at those same times. Green indicates excess predicted mass; navy indicates missing predicted mass. The shared color scale is symmetric logarithmic, linear within ±0.01 kg, with no clipping.",
    ),
    (
        5,
        "body_mass",
        "(a) Change from initial material mass beneath the body; (b) mass of particle centers inside the body; (c) total represented particle mass. Physical particle masses are used throughout.",
    ),
    (
        6,
        "particle_region_counts",
        "Reference particle populations: (a) particles within 0.05 m of the body surface; (b) particles moving faster than 0.01 m/s. Membership is recomputed each step. Zero count makes the corresponding regional position error undefined. Count axes are logarithmic above one particle and linear from zero to one.",
    ),
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(caption, headings, rows):
    head = "".join(f'<th scope="col">{html.escape(str(value))}</th>' for value in headings)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def page(title, content, *, depth):
    assets = "../" * depth
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Particle NeRD</title><link rel="stylesheet" href="{assets}assets/slides.css"></head>
<body class="benchmark-report"><header><a class="brand" href="{assets}index.html">Particle NeRD / Benchmarks</a><span class="archive">Shared object-model evaluation</span></header>
<main><section class="report-heading"><p class="eyebrow">BOX · SPHERE · CYLINDER</p><h1>{html.escape(title)}</h1>
<p>One shared particle model, evaluated against implicit MPM with the same recorded interacting-body motion.</p></section>{content}</main>
<footer class="site-footer"><span class="footer-brand">Particle NeRD</span><span>Recorded-motion evaluation · Raw forces · Matched plots and video</span></footer></body></html>'''


def section(title, content):
    return f'<section class="report-section"><h2>{html.escape(title)}</h2>{content}</section>'


def measurement(value):
    if value["mean"] is None:
        return "Unavailable"
    result = f"{value['mean']:.4g}"
    if value["sample_std"] is not None:
        result += f" ± {value['sample_std']:.3g}"
    return result + f" (n={value['available_trajectories']})"


def training_details(root, config, records, checkpoint):
    """Describe the selected weights, accepting an explicit frozen in-progress snapshot."""
    validation = [row for row in records if "validation_loss" in row]
    if not validation:
        raise ValueError("Training log contains no completed validation checkpoint")
    selected = min(validation, key=lambda row: row["validation_loss"])
    observed_step = max(row["step"] for row in records)
    complete = any(row["step"] == config["steps"] for row in validation)
    snapshot_path = root / "training_snapshot.json"
    snapshot = read(snapshot_path) if snapshot_path.exists() else None
    if snapshot is not None:
        if (
            snapshot.get("schema") != "particle_nerd_object_evaluation_snapshot_v1"
            or snapshot.get("checkpoint_sha256") != checkpoint
            or snapshot.get("checkpoint_step") != selected["step"]
            or snapshot.get("training_step_at_snapshot") != observed_step
            or snapshot.get("training_complete") != complete
        ):
            raise ValueError("Frozen checkpoint metadata does not match the training log and weights")
        for key in ("selection", "source_workflow", "snapshot_utc"):
            if not snapshot.get(key):
                raise ValueError(f"Frozen checkpoint metadata is missing {key}")
    elif not complete:
        raise ValueError("In-progress evaluation requires an explicit frozen checkpoint snapshot")
    horizons = [row["horizon"] for row in records if "loss" in row and row["step"] <= selected["step"]]
    if not horizons:
        raise ValueError("No training-window records exist for the selected checkpoint")
    return {
        "checkpoint_step": selected["step"],
        "checkpoint_training_horizon": max(horizons),
        "selected_validation": selected,
        "training_step_at_snapshot": observed_step,
        "planned_updates": config["steps"],
        "training_complete": complete,
        "snapshot": snapshot,
    }


def verify_case(root, checkpoint):
    meta, plots, video = (
        read(root / name) for name in ("evaluation.json", "plot_provenance.json", "video_provenance.json")
    )
    if any(item["checkpoint_sha256"] != checkpoint for item in (meta, plots, video)):
        raise ValueError(f"Checkpoint mismatch: {root}")
    for key in ("reference_sha256", "prediction_sha256"):
        if plots[key] != video[key]:
            raise ValueError(f"Plot/video {key} mismatch: {root}")
    if meta["reference_sha256"] != plots["reference_sha256"]:
        raise ValueError(f"Reference mismatch: {root}")
    if video["fps"] != 50 or video["native_dt_s"] != meta["native_dt_s"]:
        raise ValueError(f"Video timing mismatch: {root}")
    expected_indices = list(range(4, meta["completed_steps"] + 1, 4))
    if video["state_indices"] != expected_indices:
        raise ValueError(f"Video does not cover the evaluated interval: {root}")
    expected_hashes = {**plots["inputs_sha256"], **plots["figures_sha256"], "comparison.mp4": video["video_sha256"]}
    for name, expected in expected_hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f"Artifact changed after generation: {root / name}")
    if digest(root / "prediction.h5") != video["prediction_sha256"]:
        raise ValueError(f"Prediction mismatch: {root}")
    for _, stem, _ in FIGURES:
        if stem + ".svg" not in plots["figures_sha256"]:
            raise ValueError(f"Missing required reference-case plot: {root / stem}")
    return meta, plots, video


def build_case(root, output, checkpoint):
    meta, plots, video = verify_case(root, checkpoint)
    episode = meta["episode"]
    report = read(root.parent / "report.json")
    for key in ("checkpoint_sha256", "network_source_sha256", "evaluation_source_sha256"):
        if report[key] != meta[key]:
            raise ValueError(f"Force-window report identity mismatch: {root}")
    summary = next(record for record in report["episodes"] if record["id"] == episode["id"])
    output.mkdir(parents=True, exist_ok=True)
    for name in [*plots["figures_sha256"], "comparison.mp4", "comparison_poster.png"]:
        shutil.copy2(root / name, output / name)
    # Publish settings and hashes, never raw state arrays or network weights.
    public_meta = {key: value for key, value in meta.items() if key != "reference_path"}
    for name, value in (
        ("evaluation_settings.json", public_meta),
        ("plot_provenance.json", plots),
        ("video_provenance.json", video),
        ("force_windows.json", summary["force_windows"]),
    ):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    completed = meta["completed_steps"] * meta["native_dt_s"]
    condition_rows = [
        (
            "Reference motion",
            f"{episode['shape'].capitalize()}, {episode['mode']}, {episode['motion']}; reference duration {episode['duration_s']:g} s",
        ),
        ("Evaluated interval", f"0–{completed:g} s; {meta['completed_steps']}/{meta['expected_steps']} steps"),
        (
            "Particle initialization",
            "Identical initial MPM positions; repeated initial position supplies zero-velocity history. Subsequent predictions are never replaced by MPM positions.",
        ),
        ("Body input", meta["body_input"]),
        (
            "Body response in force reconstruction",
            "Zero body mobility for prescribed objects; recorded physical mass and inertia for dynamic objects. Particle velocities use forward differences of consecutive positions.",
        ),
        (
            "Particle update",
            "Every particle is evaluated; no active-particle selection, displacement gate, particle projection, or force filter.",
        ),
        (
            "Regional metrics",
            f"Near-contact reference distance ≤{meta['contact_distance_m']:g} m; moving reference speed >{meta['moving_speed_m_s']:g} m/s. Regions use the same reference-defined particle IDs.",
        ),
        ("Mass measurement beneath body", meta["below_body_region"]),
        (
            "Surface reconstruction",
            f"Grid {meta['grid_spacing_m']:g} m; sensitivity grid {meta['surface_sensitivity_spacing_m']:g} m; Gaussian σ={meta['surface_kernel_sigma_m']:g} m; density threshold {meta['surface_density_threshold_kg_m3']:g} kg/m³; {meta['surface_samples_per_direction']} area-uniform samples per direction.",
        ),
        (
            "Force reconstruction",
            f"Normal and tangential material effective-mass factors {meta['force_parameters']['effective_mass_factor']:g} and {meta['force_parameters']['tangent_mass_factor']:g}; stabilization {meta['force_parameters']['stabilization']:g}; collider friction {meta['force_parameters']['friction']:g}.",
        ),
    ]
    content = section(
        "1. Simulation conditions and evaluation method",
        '<p><a href="../">Shared model, training conditions, and held-out test results</a></p>'
        + table("Table 1: Conditions for this recorded-motion comparison.", ("Item", "Definition"), condition_rows)
        + "<p>The reference body moves dynamically or follows its prescribed trajectory during MPM recording. During evaluation, its recorded poses and solver-entry free velocities are imposed on Particle NeRD. The recorded free velocities retain information from the MPM coupling. Estimated forces do not change body motion; these results do not establish closed-loop stability.</p>",
    )
    if meta["failure"]:
        content += section("Rollout failure", "<p>" + html.escape(json.dumps(meta["failure"])) + "</p>")
    first, last = (video["state_indices"][i] * meta["native_dt_s"] for i in (0, -1))
    content += section(
        "2. Matched simulation video",
        f'<figure class="video-figure"><video controls playsinline preload="metadata" poster="comparison_poster.png"><source src="comparison.mp4" type="video/mp4"></video><figcaption><strong>Video 1:</strong> Left: recorded implicit MPM. Right: recurrent Particle NeRD predictions. Both panels show the same recorded body poses; an object can therefore remain suspended when predicted sand loses support. These exact states supply the plots below. {len(video["state_indices"])} frames at 50 fps, simulation time {first:g}–{last:g} s.</figcaption></figure>',
    )
    figures = ""
    for number, stem, caption in FIGURES:
        if number in (2, 3) and episode["id"].endswith("-static-support"):
            caption += " Panels (d), (e), and (f) show Fx, Fy, and Fz from 3 s onward with expanded force scales."
        figures += f'<figure class="result-figure"><div class="plot-scroll" tabindex="0"><a href="{stem}.svg"><img src="{stem}.svg" alt="{html.escape(caption)}" loading="lazy"></a></div><figcaption><strong>Figure {number}:</strong> {caption} <a href="{stem}.png">PNG</a> · <a href="{stem}.pdf">PDF</a></figcaption></figure>'
    content += section("3. Particle, material, and force measurements", figures)
    rows = []
    for key, label in (
        ("force_predicted_particles", "Predicted particles"),
        ("force_reference_particles", "Reference MPM particles"),
    ):
        force = meta[key]
        for axis, rmse, bias in zip("xyz", force["component_rmse_n"], force["component_bias_n"], strict=True):
            rows.append((label, "F" + axis, f"{rmse:.4g}", f"{bias:+.4g}"))
    window_rows = []
    for start in sorted({1.0, 3.0, completed - 1.0}):
        for source, label in (
            ("reference", "Implicit MPM"),
            ("predicted_particles", "Estimator: predicted particles"),
            ("reference_particles", "Estimator: MPM particles"),
        ):
            window = next(item for item in summary["force_windows"][source] if item["start_exclusive_s"] == start)
            values = [
                f"{mean:.3g} ± {std:.3g}"
                for mean, std in zip(window["mean_fx_fy_fz_n"], window["frame_std_fx_fy_fz_n"], strict=True)
            ]
            window_rows.append((f"({start:g}, {window['end_inclusive_s']:g}]", label, window["samples"], *values))
    content += section(
        "4. Force errors and provenance",
        table(
            "Table 2: Raw force-component errors against MPM over the evaluated interval.",
            ("Estimator particle input", "Component", "RMSE (N)", "Signed mean error (N)"),
            rows,
        )
        + table(
            "Table 3: Force mean ± temporal standard deviation in selected one-second windows.",
            ("Time (s)", "Force source", "Frames", "Fx (N)", "Fy (N)", "Fz (N)"),
            window_rows,
        )
        + '<p>Each window uses every raw 200 Hz sample after its start through its end. The population standard deviation measures variation over time in this single recording, not uncertainty across repeated trials. Early, intermediate, and final windows are shown; <a href="force_windows.json">all one-second windows</a> and full force traces remain available.</p>'
        + '<p><a href="evaluation_settings.json">Evaluation settings and metrics</a> · <a href="plot_provenance.json">Plot hashes</a> · <a href="video_provenance.json">Video timestamps and hashes</a></p><p>Only figures, videos, settings, and result summaries are public. Raw trajectories and model weights remain private.</p>',
    )
    (output / "index.html").write_text(page(CASES[episode["id"]], content, depth=3))
    return public_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training",
        type=Path,
        required=True,
        help="Downloaded private training output containing model/ and data_audit.json",
    )
    parser.add_argument(
        "--test-report", type=Path, required=True, help="Aggregate report.json for all 60 held-out trajectories"
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Four reference-case evaluation directories with matching plots and video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New staging directory; publish beneath benchmark_environment/object_prediction",
    )
    args = parser.parse_args()
    config, audit, report = (
        read(args.training / "model/config.json"),
        read(args.training / "data_audit.json"),
        read(args.test_report),
    )
    checkpoint = digest(args.training / "model/best.pt")
    if (
        config["smoke"]
        or audit["trajectories"] != 600
        or audit["splits"] != {"train": 480, "validation": 60, "test": 60}
    ):
        raise ValueError("This report requires the completed 600-trajectory campaign")
    records = [json.loads(line) for line in (args.training / "model/metrics.jsonl").read_text().splitlines()]
    training = training_details(args.training, config, records, checkpoint)
    if checkpoint != report["checkpoint_sha256"] or report["groups"]["combined"]["trajectories"] != 60:
        raise ValueError("Test report must contain 60 trajectories evaluated with this checkpoint")
    if args.output.exists():
        raise ValueError("Use a new staging directory; existing reports are never overwritten")
    expected_test = {f"{shape}-{index:04d}" for shape in ("box", "sphere", "cylinder") for index in range(9, 200, 10)}
    if len(report["episodes"]) != 60 or {episode["id"] for episode in report["episodes"]} != expected_test:
        raise ValueError("Report does not contain the exact 60 held-out test IDs")
    # Validate every input before producing a partial export.
    for case in CASES:
        meta, _, _ = verify_case(args.references / case, checkpoint)
        for key in ("network_source_sha256", "evaluation_source_sha256"):
            if meta[key] != report[key]:
                raise ValueError(f"Reference/test implementation mismatch in {case}: {key}")
    args.output.mkdir(parents=True)
    for case in CASES:
        build_case(args.references / case, args.output / case, checkpoint)
    rows = [
        (
            "Object geometry",
            "Box: 0.24 × 0.18 × 0.12 m; Sphere: radius 0.10 m; Cylinder: radius 0.10 m, height 0.12 m, local z axis.",
        ),
        (
            "Coordinates",
            "Shared world coordinates; z points upward, gravity is (0, 0, −9.81) m/s². The nominal initial sand fill level is z = 0 m and the rigid tray floor is z = −0.30 m.",
        ),
        (
            "Training data",
            "600 six-second trajectories: 200 each for Box, Sphere and Cylinder; 480 training, 60 validation, 60 held-out test trajectories.",
        ),
        (
            "Prescribed motion",
            "360 trajectories: horizontal motion, vertical motion, circles and holding. Object poses are imposed.",
        ),
        (
            "Dynamic motion",
            "240 trajectories: drops, slides, throws and tumbles. Two-kilogram objects are released after the first second and respond to gravity and contacts.",
        ),
        (
            "Sand and resolution",
            "2 × 1.2 × 0.30 m flat bed; 20,646 particles; MPM grid 0.04 m; particle spacing 0.032 m, radius 0.016 m; seeded bounded position jitter up to 0.0064 m per coordinate.",
        ),
        (
            "Material",
            "Density 1,600 kg/m³, Young’s modulus 10¹⁵ Pa, Poisson ratio 0.3, material friction 0.48, yield pressure 10¹⁵ Pa; collider friction 1.0.",
        ),
        (
            "Sampling",
            "Native interval 0.005 s (200 Hz). Particle velocities are derived from consecutive positions. Initial settling is included.",
        ),
        (
            "Shared model",
            f"One checkpoint from update {training['checkpoint_step']:,}, selected by the lowest validation loss "
            "with equal weighting of moving and stationary particles. "
            f"Its training windows reached {training['checkpoint_training_horizon']} steps. "
            "Test trajectories are never used for selection.",
        ),
        (
            "Model architecture",
            f"Particle Transolver: {config['model']['layers']} attention blocks, hidden size {config['model']['hidden']}, "
            f"{config['model']['heads']} heads, {config['model']['slices']} slices, "
            f"{config['model']['history']} position-history frames; displacement scale {config['model']['displacement_scale']:g} m.",
        ),
        (
            "Checkpoint validation",
            f"{training['selected_validation']['validation_trajectories']} of the 60 reserved validation trajectories; "
            f"{training['selected_validation']['validation_windows']} windows of {config['validation_horizon'] * 0.005:g} s. "
            "Each selected trajectory contributes initial settling and a motion-specific window. "
            "The loss separates moving/stationary groups at 0.5 mm reference displacement per step; "
            "this differs from the 0.01 m/s threshold used by the reported regional position metric.",
        ),
        (
            "Training progress when weights were frozen",
            f"{training['training_step_at_snapshot']:,} of {training['planned_updates']:,} planned updates "
            "had been recorded. All results on this page use the same frozen checkpoint; "
            "subsequent training does not change them.",
        ),
        (
            "Neural update",
            "Full-particle recurrent prediction. No Top-K selection, foot-speed displacement gate, particle projection, or force filtering. This object model is separate from the deployed G1 solver.",
        ),
        (
            "Evaluation",
            "Recorded MPM body inputs with recurrent predicted particles; no reference particle replacement after initialization. Four separate reference cases add raw material-force comparisons and 20-second support tests.",
        ),
        ("Checkpoint SHA-256", checkpoint),
    ]
    content = section(
        "1. Simulation conditions and evaluation method",
        table("Table 1: Shared object-model experiment.", ("Item", "Condition"), rows),
    )
    metric_keys = (
        ("global_rmse_mm_time_mean", "Position RMSE (mm)"),
        ("surface_mean_mm_time_mean", "Surface mean distance (mm)"),
        ("surface_p95_mm_time_mean", "Surface p95 distance (mm)"),
        ("mass_discrepancy_time_mean", "Mass discrepancy"),
        ("velocity_rmse_m_s_time_mean", "Velocity RMSE (m/s)"),
        ("velocity_reference_mass_coverage_time_mean", "Velocity mass coverage"),
    )
    metric_rows = []
    for group in ("combined", "box", "sphere", "cylinder"):
        values = report["groups"][group]
        metrics = values["available_duration_including_failed_rollouts"]
        metric_rows.append(
            (
                group.capitalize(),
                f"{values['complete_trajectories']}/{values['trajectories']}",
                *(measurement(metrics[key]) for key, _ in metric_keys),
            )
        )
    content += section(
        "2. Held-out recurrent rollouts",
        "<p>Each value first averages time within one trajectory, then weights trajectories equally. ± denotes sample standard deviation across trajectory summaries; n is the number with an available measurement. Failed rollouts contribute their available duration and remain in the denominator of completion counts. No smoothing, alignment, or time shifting is applied.</p>"
        + "<p>Across the 60 test trajectories, the unchanged-initial-position baseline has a time-averaged "
        + f"particle-position RMSE of {measurement(report['groups']['combined']['available_duration_including_failed_rollouts']['frozen_rmse_mm_time_mean'])} mm. "
        + "Particle NeRD's error at the final six-second frame is "
        + f"{measurement(report['groups']['combined']['complete_rollouts']['global_rmse_mm_end'])} mm. "
        + "The table below reports errors averaged over time, which are smaller than final-frame errors when predictions drift.</p>"
        + table(
            "Table 2: All 60 test trajectories, including available portions of failed rollouts.",
            ("Shape", "Numerically completed", *(label for _, label in metric_keys)),
            metric_rows,
        )
        + '<p>Numerical completion does not establish physical accuracy. The complete summary includes regional position errors, surface-distance percentiles and resolution sensitivity, velocity coverage, unavailable measurements, and failure times. <a href="test_report.json">Read the full metric summary</a>.</p>',
    )
    reference_report = read(args.references / "report.json")
    reference_rows = []
    for record in reference_report["episodes"]:
        final_forces = [record["force_windows"][source][-1] for source in ("reference", "predicted_particles")]
        reference_rows.append(
            (
                CASES[record["id"]],
                f"{record['completed_duration_s']:g}",
                f"{record['metrics']['global_rmse_mm_end']:.4g}",
                *(
                    f"{window['mean_fx_fy_fz_n'][2]:.4g} ± {window['frame_std_fx_fy_fz_n'][2]:.3g}"
                    for window in final_forces
                ),
            )
        )
    content += section(
        "3. Detailed reference cases",
        "<p>These four additional recordings use the same trained checkpoint. The 20-second support tests extend beyond the six-second training recordings. Each case includes conditions, an exact matching 50 fps video, force traces, particle errors, and material-distribution measurements.</p><ul>"
        + "".join(f'<li><a href="{case}/">{title}</a></li>' for case, title in CASES.items())
        + "</ul>"
        + table(
            "Table 3: Reference-case final position errors and material-on-object Fz during the last second.",
            ("Case", "Duration (s)", "Final position RMSE (mm)", "MPM Fz (N)", "NeRD Fz (N)"),
            reference_rows,
        )
        + "<p>Force entries give the mean ± population standard deviation over time, not repeated-trial uncertainty. "
        "Body poses are prescribed from the MPM recording during evaluation; support loss does not change the replayed object motion. "
        "Fx and Fy, complete force traces, and estimator-only diagnostics appear on each case page.</p>",
    )
    (args.output / "index.html").write_text(page("Shared object-model prediction accuracy", content, depth=2))
    (args.output / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "training_provenance.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint,
                "dataset_splits": audit["splits"],
                "dataset_groups": audit["groups"],
                "trajectory_hashes": {key: value["sha256"] for key, value in audit["files"].items()},
                "training_config": {key: value for key, value in config.items() if key not in {"data", "output"}},
                "evaluated_training_state": training,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built report and four matched case pages in {args.output}; no raw trajectories or weights copied.")


if __name__ == "__main__":
    main()
