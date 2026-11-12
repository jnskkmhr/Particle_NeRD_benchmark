"""Build public object-model reports from completed, matching evaluation artifacts."""

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path

from object_dynamic_results import dynamic_results, evaluation_modes
from object_metric_results import (
    HOLD_CASES,
    hold_results,
    metric_results,
    metric_section,
)
from object_motion_videos import install_test_videos, motion_gallery, reference_videos
from plot_object_metric_comparisons import figure_html, generate

CASES = {
    "box-static-support": "Box: 20-second support test",
    "sphere-static-support": "Sphere: 20-second support test",
    "cylinder-static-support": "Cylinder: 20-second support test",
}
FIGURES = (
    (
        1,
        "particle_evaluation",
        "One paired rollout. Top row: particle-position RMSE for the whole bed, near-contact particles and moving particles; symmetric sand-surface distance. Middle row: material-mass discrepancy and velocity-field RMSE, with reference-mass coverage. Bottom row: particle kinetic energy and material mass beneath the interacting body. Empty reference groups produce gaps; Figure 6 shows counts.",
    ),
    (
        2,
        "forces",
        "Raw material force on the interacting body: Left to right: Fx, Fy, Fz. Particle NeRD uses its recurrent particle predictions. Recorded MPM body motion and free velocities are imposed; estimated forces do not feed back to body motion. Direct rigid-tray forces are excluded.",
    ),
    (
        3,
        "force_reconstruction",
        "Force-reconstruction diagnostic: Left to right: Fx, Fy, Fz. Recorded MPM particles replace predicted particles in the same force estimator. This input substitution isolates the estimator; it is not a second neural model. Recorded MPM body motion and free velocities are imposed; estimated forces do not feed back to body motion.",
    ),
    (
        4,
        "mass_difference",
        "Predicted minus reference physical mass per projected grid column. Top row: xy maps at the printed times. Bottom row: xz maps at those same times. Green indicates excess predicted mass; navy indicates missing predicted mass. The shared color scale is symmetric logarithmic, linear within ±0.01 kg, with no clipping.",
    ),
    (
        5,
        "body_mass",
        "Left to right: change from initial material mass beneath the body; mass of particle centers inside the body; total represented particle mass. Physical particle masses are used throughout.",
    ),
    (
        6,
        "particle_region_counts",
        "Reference particle populations: Left: near-contact particles within 0.05 m of the body surface. Right: moving particles with speed above 0.01 m/s. Membership is recomputed each step. Zero count makes the corresponding regional position error undefined. Count axes are logarithmic above one particle and linear from zero to one.",
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


def page(
    title,
    content,
    *,
    depth,
    extra_head="",
    introduction="One shared particle model, evaluated against implicit MPM with the same recorded interacting-body motion.",
):
    assets = "../" * depth
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Particle NeRD</title><link rel="stylesheet" href="{assets}assets/slides.css">{extra_head}</head>
<body class="benchmark-report"><header><a class="brand" href="{assets}index.html">Particle NeRD / Benchmarks</a><span class="archive">Shared object-model evaluation</span></header>
<main><section class="report-heading"><p class="eyebrow">BOX · SPHERE · CYLINDER</p><h1>{html.escape(title)}</h1>
<p>{html.escape(introduction)}</p></section>{content}</main>
<footer class="site-footer"><span class="footer-brand">Particle NeRD</span><span>Explicit evaluation conditions · Raw forces · Matched plots and video</span></footer></body></html>'''


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
    expected_hashes = {
        **plots["inputs_sha256"],
        **plots["figures_sha256"],
        "comparison.mp4": video["video_sha256"],
    }
    for name, expected in expected_hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f"Artifact changed after generation: {root / name}")
    if digest(root / "prediction.h5") != video["prediction_sha256"]:
        raise ValueError(f"Prediction mismatch: {root}")
    required = FIGURES if meta["episode"]["id"] not in HOLD_CASES else [f for f in FIGURES if f[0] in (1, 6)]
    for _, stem, _ in required:
        if stem + ".svg" not in plots["figures_sha256"]:
            raise ValueError(f"Missing required reference-case plot: {root / stem}")
    return meta, plots, video


def build_case(root, output, checkpoint):
    meta, plots, video = verify_case(root, checkpoint)
    episode = meta["episode"]
    report = read(root.parent / "report.json")
    for key in (
        "checkpoint_sha256",
        "network_source_sha256",
        "evaluation_source_sha256",
    ):
        if report[key] != meta[key]:
            raise ValueError(f"Force-window report identity mismatch: {root}")
    summary = next(record for record in report["episodes"] if record["id"] == episode["id"])
    output.mkdir(parents=True, exist_ok=True)
    for name in [*plots["figures_sha256"], "comparison.mp4", "comparison_poster.png"]:
        shutil.copy2(root / name, output / name)
    # Publish settings and hashes, never raw state arrays or network weights.
    public_meta = {key: value for key, value in meta.items() if key != "reference_path"}
    export_files = [
        ("evaluation_settings.json", public_meta),
        ("plot_provenance.json", plots),
        ("video_provenance.json", video),
    ]
    if "force_windows" in summary:
        export_files.append(("force_windows.json", summary["force_windows"]))
    for name, value in export_files:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    completed = meta["completed_steps"] * meta["native_dt_s"]
    condition_rows = [
        (
            "Reference motion",
            f"{episode['shape'].capitalize()}, {episode['mode']}, {episode['motion']}; reference duration {episode['duration_s']:g} s",
        ),
        (
            "Evaluated interval",
            f"0–{completed:g} s; {meta['completed_steps']}/{meta['expected_steps']} steps",
        ),
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
            f"Offline surface-reconstruction grid {meta['grid_spacing_m']:g} m; resolution-sensitivity check {meta['surface_sensitivity_spacing_m']:g} m (measurement settings, not simulation resolution); Gaussian σ={meta['surface_kernel_sigma_m']:g} m; density threshold {meta['surface_density_threshold_kg_m3']:g} kg/m³; {meta['surface_samples_per_direction']} area-uniform samples per direction.",
        ),
        (
            "Force reconstruction",
            f"Normal and tangential material effective-mass factors {meta['force_parameters']['effective_mass_factor']:g} and {meta['force_parameters']['tangent_mass_factor']:g}; stabilization {meta['force_parameters']['stabilization']:g}; collider friction {meta['force_parameters']['friction']:g}.",
        ),
    ]
    content = section(
        "Evaluation conditions",
        '<p><a href="../">Shared model, training conditions, and held-out test results</a></p>'
        + table(
            "Table 1: Conditions for this recorded-motion comparison.",
            ("Item", "Definition"),
            condition_rows,
        )
        + "<p>The reference body moves dynamically or follows its prescribed trajectory during MPM recording. During evaluation, its recorded poses and solver-entry free velocities are imposed on Particle NeRD. The recorded free velocities retain information from the MPM coupling. Estimated forces do not change body motion; these results do not establish closed-loop stability.</p>",
    )
    if meta["failure"]:
        content += section("Rollout failure", "<p>" + html.escape(json.dumps(meta["failure"])) + "</p>")
    if meta.get("field_failure"):
        content += section(
            "Paired field measurement unavailable",
            "<p>" + html.escape(meta["field_failure"]) + "</p><p>All particles remain in the position and force "
            "measurements. Temporal metrics below use a larger grid and retain escaped particles.</p>",
        )
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    first, last = (video["state_indices"][i] * meta["native_dt_s"] for i in (0, -1))
    content += section(
        "Matched simulation video",
        f'<figure class="video-figure" id="hold-video"><video controls playsinline preload="metadata" poster="comparison_poster.png"><source src="comparison.mp4" type="video/mp4"></video><figcaption><strong>Video 1:</strong> Left: recorded implicit MPM. Right: recurrent {html.escape(model_label)} predictions. Both panels show the same recorded body poses; an object can therefore remain suspended when predicted sand loses support. These exact states supply the plots below. {len(video["state_indices"])} frames at 50 fps, simulation time {first:g}–{last:g} s.</figcaption></figure>',
    )
    figures = ""
    for number, stem, caption in FIGURES:
        if stem + ".svg" not in plots["figures_sha256"]:
            continue
        if number == 1 and meta.get("field_failure"):
            caption = "Particle-position errors. The remaining field measurements are unavailable because the complete particle domain exceeded the paired field evaluator's grid limit; see the failure notice above. No particles were clipped."
        if number in (2, 3) and episode["id"].endswith("-static-support"):
            caption += " The bottom row shows Fx, Fy, and Fz from 3 s onward with expanded force scales."
        figures += f'<figure class="result-figure"><div class="plot-scroll" tabindex="0"><a href="{stem}.svg"><img src="{stem}.svg" alt="{html.escape(caption)}" loading="lazy"></a></div><figcaption><strong>Figure {number}:</strong> {caption} <a href="{stem}.png">PNG</a> · <a href="{stem}.pdf">PDF</a></figcaption></figure>'
    content += section("Time series for metrics 1–5", figures)
    if episode["id"] in HOLD_CASES:
        content += metric_section(
            6,
            "<p>The object is held at an exactly fixed pose after 2 s. Each method is measured relative to its own "
            "3 s state, without reset or alignment. The reference bed also settles; escaped MPM particles are retained. "
            "The video above includes insertion and this entire measurement interval. There is one run per shape.</p>"
            '<figure class="result-figure"><div class="plot-scroll"><a href="spatiotemporal_consistency.svg">'
            '<img src="spatiotemporal_consistency.svg" alt="Sand drift during the fixed-object hold"></a></div>'
            "<figcaption><strong>Figure 7:</strong> Top row: RMS particle drift and signed mean vertical drift; "
            "middle row: mean and 95th-percentile surface change and normalized grid-mass change; "
            "bottom row: mass beneath the object and mass below the tray floor. "
            '<a href="spatiotemporal_consistency.png">PNG</a> · <a href="spatiotemporal_consistency.pdf">PDF</a>'
            '</figcaption></figure><p><a href="consistency_summary.json">Measured results and settings</a> · '
            '<a href="consistency_series.json">Scalar time series</a> · '
            '<a href="consistency_provenance.json">Input and figure hashes</a></p>',
            section,
            numbered=False,
        )
    if episode["id"] not in HOLD_CASES:
        content += metric_section(
            5,
            figure_html(
                "force_error_summary",
                7,
                "Left: Fx/Fy/Fz component RMSE; right: signed mean error (estimate minus native MPM material force). "
                "Raw 200 Hz forces over the full recording. The gray diagnostic supplies MPM particles to the same estimator.",
            )
            + figure_html(
                "force_window_summary",
                8,
                "Left to right: Fx, Fy, Fz. Means and temporal population standard deviations over one-second "
                "windows (1,2], (3,4], and the final second. Every raw 200 Hz sample is used. "
                "Whiskers describe variation within one recording, not repeated-trial uncertainty.",
            )
            + '<p><a href="force_windows.json">Exact one-second window results</a> · '
            '<a href="evaluation_settings.json">Measured force errors and evaluation settings</a> · '
            '<a href="../comparison_plot_provenance.json">Summary-plot hashes</a> · '
            '<a href="plot_provenance.json">Time-series plot hashes</a> · '
            '<a href="video_provenance.json">Video timestamps and hashes</a></p>'
            "<p>Only figures, videos, settings, and result summaries are public. Raw trajectories and weights remain private.</p>",
            section,
            numbered=False,
        )
    (output / "index.html").write_text(page(({**CASES, **HOLD_CASES})[episode["id"]], content, depth=3))
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
        "--test-report",
        type=Path,
        required=True,
        help="Aggregate report.json for all 60 held-out trajectories",
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Four reference-case evaluation directories with matching plots and video",
    )
    parser.add_argument(
        "--holds",
        type=Path,
        required=True,
        help="Three completed fixed-hold evaluations with temporal metrics and matched video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New staging directory; publish beneath benchmark_environment/object_prediction",
    )
    parser.add_argument(
        "--test-videos",
        type=Path,
        required=True,
        help="Verified videos of the first held-out episode in each shape/motion group",
    )
    parser.add_argument(
        "--dynamics",
        type=Path,
        required=True,
        help="Three fully coupled object/NeRD simulations with matched videos",
    )
    parser.add_argument(
        "--dynamic-page-output",
        type=Path,
        default=None,
        help="Output directory for the dedicated elementary_object_dynamic_simulation page",
    )
    args = parser.parse_args()
    config, audit, report = (
        read(args.training / "model/config.json"),
        read(args.training / "data_audit.json"),
        read(args.test_report),
    )
    checkpoint = digest(args.training / "model/best.pt")
    valid_600 = audit["trajectories"] == 600 and audit["splits"] == {"train": 480, "validation": 60, "test": 60}
    valid_680 = audit["trajectories"] == 680 and audit["splits"] == {"train": 544, "validation": 68, "test": 68}
    if config["smoke"] or not (valid_600 or valid_680):
        raise ValueError("This report requires the completed 600- or 680-trajectory campaign")
    records = [json.loads(line) for line in (args.training / "model/metrics.jsonl").read_text().splitlines()]
    training = training_details(args.training, config, records, checkpoint)
    expected_test_count = 68 if valid_680 else 60
    if checkpoint != report["checkpoint_sha256"] or report["groups"]["combined"]["trajectories"] != expected_test_count:
        raise ValueError(f"Test report must contain {expected_test_count} trajectories evaluated with this checkpoint")
    if args.output.exists():
        raise ValueError("Use a new staging directory; existing reports are never overwritten")
    expected_test = {f"{shape}-{index:04d}" for shape in ("box", "sphere", "cylinder") for index in range(9, 200, 10)}
    if valid_680:
        expected_test |= {f"freefall-{index:04d}" for index in range(129, 200, 10)}
    if len(report["episodes"]) != expected_test_count or {episode["id"] for episode in report["episodes"]} != expected_test:
        raise ValueError(f"Report does not contain the exact {expected_test_count} held-out test IDs")
    # Validate every input before producing a partial export.
    for case in CASES:
        meta, _, _ = verify_case(args.references / case, checkpoint)
        for key in ("network_source_sha256", "evaluation_source_sha256"):
            if meta[key] != report[key]:
                raise ValueError(f"Reference/test implementation mismatch in {case}: {key}")
    for case in HOLD_CASES:
        meta, _, _ = verify_case(args.holds / case, checkpoint)
        for key in ("network_source_sha256", "evaluation_source_sha256"):
            if meta[key] != report[key]:
                raise ValueError(f"Hold/test implementation mismatch in {case}: {key}")
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL") or (
        "Particle World"
        if "d_model" in config["model"]
        else ("Particle NeRD-Local" if "cconv_dim" in config["model"] else "Particle NeRD")
    )
    os.environ["BENCHMARK_MODEL_LABEL"] = model_label
    args.output.mkdir(parents=True)
    motion_manifest = install_test_videos(args.test_videos, args.output, report, digest)
    for case in CASES:
        build_case(args.references / case, args.output / case, checkpoint)
    for case in HOLD_CASES:
        build_case(args.holds / case, args.output / case, checkpoint)
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
            "Dataset and split",
            (
                "680 six-second trajectories: 200 each for Box, Sphere and Cylinder, plus 80 dynamic Free-fall sand trajectories; "
                "544 training, 68 validation, 68 held-out test trajectories."
                if valid_680
                else "600 six-second trajectories: 200 each for Box, Sphere and Cylinder; 480 training, 60 validation, 60 held-out test trajectories."
            ),
        ),
        (
            "MPM dataset motion: kinematic",
            "360 trajectories: horizontal motion, vertical motion, circles and holding. Object poses are imposed.",
        ),
        (
            "MPM dataset motion: dynamic",
            (
                "320 trajectories: 240 with 2 kg objects (Box, Sphere, Cylinder) released after 1 s and 80 unconfined Free-fall sand blocks/columns (drops, slides, throws, and tumbles)."
                if valid_680
                else "240 trajectories: drops, slides, throws and tumbles. Two-kilogram objects are released after the first second and respond to gravity and contacts."
            ),
        ),
        (
            "Sand and resolution",
            "2 × 1.2 × 0.30 m flat bed (20,646 particles) for object interactions; unconfined 3D sand blocks/columns for free-fall sand; MPM grid 0.04 m; particle spacing 0.032 m, radius 0.016 m; seeded bounded position jitter up to 0.0064 m per coordinate.",
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
            (
                f"One checkpoint from update {training['checkpoint_step']:,}, selected by the lowest validation loss "
                "with equal weighting of moving and stationary particles. "
                f"Its training windows reached {training['checkpoint_training_horizon']} steps. "
                "Test trajectories are never used for selection."
            ),
        ),
        (
            "Model architecture",
            (
                f"WorldParticle (translation-invariant + smooth tanh collider decay): d_model {config['model']['d_model']}, "
                f"cconv {config['model']['cconv_dim']}, {config['model']['heads']} heads, "
                f"{config['model']['slot_attn_iters']} ToMe encoder layers, {config['model']['num_decoder_layers']} cross-attn decoder layers, "
                f"{config['model']['history']} history frames; displacement scale {config['model']['displacement_scale']:g} m."
                if "d_model" in config["model"]
                else (
                    f"ParticleNeRD-Local (translation-invariant + CConv + 3D-RoPE slices): {config['model']['layers']} attention blocks, "
                    f"hidden size {config['model']['hidden']}, cconv {config['model']['cconv_dim']}, "
                    f"{config['model']['heads']} heads, {config['model']['slices']} slices, "
                    f"{config['model']['history']} history frames; displacement scale {config['model']['displacement_scale']:g} m."
                    if "cconv_dim" in config["model"]
                    else (
                        f"Particle Transolver: {config['model']['layers']} attention blocks, hidden size {config['model']['hidden']}, "
                        f"{config['model']['heads']} heads, {config['model']['slices']} slices, "
                        f"{config['model']['history']} position-history frames; displacement scale {config['model']['displacement_scale']:g} m."
                    )
                )
            ),
        ),
        (
            "Checkpoint validation",
            (
                f"{training['selected_validation']['validation_trajectories']} reserved validation trajectories; "
                f"{training['selected_validation']['validation_windows']} windows of {config['validation_horizon'] * 0.005:g} s. "
                "Each selected trajectory contributes initial settling and a motion-specific window. "
                "The loss separates moving/stationary groups at 0.5 mm reference displacement per step; "
                "this differs from the 0.01 m/s threshold used by the reported regional position metric."
            ),
        ),
        (
            "Training progress when weights were frozen",
            (
                f"{training['training_step_at_snapshot']:,} of {training['planned_updates']:,} planned updates "
                "had been recorded. All results on this page use the same frozen checkpoint; "
                "subsequent training does not change them."
            ),
        ),
        (
            "Time to evaluated checkpoint",
            (
                f"{training['selected_validation']['elapsed_s'] / 60:.1f} minutes through checkpoint validation "
                f"on {config['world_size']} GPUs, excluding setup and data transfer."
            ),
        ),
        (
            "Neural update",
            "Full-particle recurrent prediction. No Top-K selection, foot-speed displacement gate, particle projection, or force filtering. This object model is separate from the deployed G1 solver.",
        ),
        (
            "Evaluation",
            "Recorded-motion tests use imposed MPM body inputs with recurrent predicted particles; no reference particle replacement after initialization. Dynamic contact-force accuracy and 20-second closed-loop dynamic simulations are reported on the dedicated Dynamic Object Simulation page.",
        ),
        ("Checkpoint SHA-256", checkpoint),
    ]
    content = section(
        "Evaluation conditions",
        table("Table 1: Shared object-model experiment.", ("Item", "Condition"), rows),
    )
    content += motion_gallery(motion_manifest)
    content += metric_results(report, section)
    content += metric_section(
        5,
        "<p>Contact-force accuracy on the three 20-second dynamic object support tests (Box, Sphere, and Cylinder; "
        "kinematic prescribed-object force estimation is omitted) and the three 20-second fully dynamic closed-loop "
        "simulations with NeRD force feedback are published on the dedicated benchmark page:</p>"
        '<p><a href="../elementary_object_dynamic_simulation/"><strong>Open Elementary object dynamic simulation &amp; contact-force benchmark →</strong></a></p>',
        section,
    )
    content += hold_results(args.holds, args.output, checkpoint, digest, table, section)
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL") or (
        "Particle World"
        if "d_model" in config["model"]
        else ("Particle NeRD-Local" if "cconv_dim" in config["model"] else "Particle NeRD")
    )
    os.environ["BENCHMARK_MODEL_LABEL"] = model_label
    (args.output / "index.html").write_text(
        page(
            f"{model_label}: Elementary object benchmark",
            content,
            depth=2,
            introduction=f"One shared {model_label} particle model: recorded-motion particle evaluation (kinematic, dynamic, and free-fall sand) and spatio-temporal consistency.",
            extra_head='<link rel="stylesheet" href="../../assets/object-motion-gallery.css?v=20261112-v2"><script src="../../assets/object-motion-gallery.js?v=20261112-v2" defer></script>',
        )
    )
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
    generate(args.output)

    # Also build the dedicated elementary_object_dynamic_simulation page
    dyn_page_dir = args.dynamic_page_output or (args.output.parent / "elementary_object_dynamic_simulation")
    if dyn_page_dir.exists():
        shutil.rmtree(dyn_page_dir)
    dyn_page_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        shutil.copytree(args.output / case, dyn_page_dir / case)
    for stem in ("reference_force_errors", "reference_summary"):
        for ext in ("svg", "png", "pdf"):
            shutil.copy2(args.output / f"{stem}.{ext}", dyn_page_dir / f"{stem}.{ext}")
    shutil.copy2(args.output / "comparison_plot_provenance.json", dyn_page_dir / "comparison_plot_provenance.json")

    dyn_content = section(
        "Overview",
        '<p><a href="../particle_nerd_local/">← Back to Elementary object &amp; free-fall sand particle prediction benchmark</a></p>'
        f"<p>This page evaluates contact-force reconstruction and 20-second closed-loop dynamic coupling for the shared <strong>{html.escape(model_label)}</strong> model (checkpoint <code>{checkpoint[:16]}…</code>, update {training['checkpoint_step']:,}). "
        "Force estimation is evaluated exclusively on <strong>dynamic</strong> 2 kg objects (Box, Sphere, Cylinder) that respond to gravity and sand contact; force estimation for kinematic prescribed objects is omitted.</p>",
    )
    dyn_content += metric_section(
        5,
        "<p>Force accuracy is measured on three dedicated 20-second dynamic drop-and-support recordings (Box, Sphere, and Cylinder) with the same frozen checkpoint. "
        "Each includes an exact matching 50 fps video and full curves. In this first section, body poses are replayed from the dynamic MPM recording so predicted and reference particles are evaluated along the identical object trajectory.</p><ul>"
        + "".join(f'<li><a href="{case}/">{title}</a></li>' for case, title in CASES.items())
        + "</ul>"
        + reference_videos(CASES)
        + figure_html("reference_force_errors")
        + figure_html("reference_summary"),
        section,
    )
    if args.dynamics:
        dyn_content += dynamic_results(args.dynamics, dyn_page_dir, checkpoint, section)
        # Also keep dynamic/ in args.output so existing relative links continue to resolve
        if (dyn_page_dir / "dynamic").exists() and not (args.output / "dynamic").exists():
            shutil.copytree(dyn_page_dir / "dynamic", args.output / "dynamic")
    (dyn_page_dir / "index.html").write_text(
        page(
            f"{model_label}: Elementary object dynamic simulation & force benchmark",
            dyn_content,
            depth=2,
            introduction=f"Dynamic-object contact-force accuracy (20 s drop-and-support tests) and 20 s fully dynamic closed-loop object simulations with {model_label} force feedback.",
        )
    )
    print(f"Built report in {args.output} and dynamic simulation page in {dyn_page_dir}; no raw trajectories or weights copied.")


if __name__ == "__main__":
    main()
