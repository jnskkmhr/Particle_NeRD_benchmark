"""Expose measured metric results and fixed-object diagnostics in the public report."""

import json
import shutil

HOLD_CASES = {
    f"{shape}-fixed-hold": f"{shape.capitalize()}: fixed-object hold" for shape in ("box", "sphere", "cylinder")
}


def metric_results(report, references, cases, table, section, measurement):
    """Report every core metric, retaining missing regional samples and force coverage."""
    content = "<p>Results below use the same checkpoint as Tables 1–3. Each trajectory has equal weight. "
    content += "Values are mean ± sample standard deviation across trajectories; n counts available measurements.</p>"
    groups = [
        (name.capitalize(), report["groups"][name]["available_duration_including_failed_rollouts"])
        for name in ("combined", "box", "sphere", "cylinder")
    ]
    families = (
        (
            4,
            "Particle trajectories",
            (
                ("global_rmse_mm_time_mean", "Whole bed: time mean (mm)"),
                ("global_rmse_mm_end", "Whole bed: final (mm)"),
                ("near_contact_rmse_mm_time_mean", "Near body: time mean (mm)"),
                ("near_contact_rmse_mm_end", "Near body: final (mm)"),
                ("moving_rmse_mm_time_mean", "Moving: time mean (mm)"),
                ("moving_rmse_mm_end", "Moving: final (mm)"),
                ("near_contact_count_time_mean", "Near-body particles: time mean"),
                ("moving_count_time_mean", "Moving particles: time mean"),
            ),
        ),
        (
            5,
            "Sand shape and reconstruction sensitivity",
            (
                ("surface_mean_mm_time_mean", "Mean surface distance: time mean (mm)"),
                ("surface_p95_mm_time_mean", "Surface p95: time mean (mm)"),
                ("surface_mean_mm_end", "Mean surface distance: final (mm)"),
                ("surface_p95_mm_end", "Surface p95: final (mm)"),
                ("surface_mean_mm_30mm_grid_time_mean", "Mean surface distance, 0.03 m grid (mm)"),
                ("surface_p95_mm_30mm_grid_time_mean", "Surface p95, 0.03 m grid (mm)"),
            ),
        ),
        (
            6,
            "Physical material distribution",
            (
                ("mass_discrepancy_time_mean", "Grid-mass discrepancy: time mean"),
                ("mass_discrepancy_end", "Grid-mass discrepancy: final"),
                ("reference_below_body_mass_kg_end", "MPM: mass beneath body, final (kg)"),
                ("predicted_below_body_mass_kg_end", "NeRD: mass beneath body, final (kg)"),
                ("reference_inside_body_mass_kg_end", "MPM: mass inside body, final (kg)"),
                ("predicted_inside_body_mass_kg_end", "NeRD: mass inside body, final (kg)"),
                ("reference_mass_kg_end", "MPM: total represented mass (kg)"),
                ("predicted_mass_kg_end", "NeRD: total represented mass (kg)"),
            ),
        ),
        (
            7,
            "Material motion",
            (
                ("velocity_rmse_m_s_time_mean", "Velocity-field RMSE: time mean (m/s)"),
                ("velocity_rmse_m_s_end", "Velocity-field RMSE: final (m/s)"),
                ("velocity_reference_mass_coverage_time_mean", "Velocity reference-mass coverage: time mean"),
                ("velocity_reference_mass_coverage_end", "Velocity reference-mass coverage: final"),
                ("reference_ke_j_time_mean", "MPM: kinetic energy, time mean (J)"),
                ("predicted_ke_j_time_mean", "NeRD: kinetic energy, time mean (J)"),
            ),
        ),
    )
    for number, title, keys in families:
        content += f"<h3>{title}</h3>" + table(
            f"Table {number}: {title}; 60 six-second test trajectories.",
            ("Metric", *(label for label, _ in groups)),
            [(label, *(measurement(values[key]) for _, values in groups)) for key, label in keys],
        )
    content += "<p>Near-body membership uses reference distance ≤0.05 m; moving particles have reference speed "
    content += "&gt;0.01 m/s. One trajectory has no moving particles at its final frame, so that error has n=59. "
    content += "A surface p95 is a spatial distance percentile, not confidence across trajectories. "
    content += "Total mass stays constant by particle accounting even when particles leave the physical tray.</p>"
    content += "<h3>Material force on the interacting object</h3><p>The 60 held-out dataset recordings do not contain "
    content += "reference forces. Force accuracy is measured on the four dedicated reference recordings below; "
    content += "it is not a 60-trajectory force result. Each row is one recording. "
    content += "The reference-particle row uses the same estimator with recorded MPM particles to isolate force reconstruction.</p>"
    force_rows = []
    for case in cases:
        meta = json.loads((references / case / "evaluation.json").read_text())
        for key, label in (
            ("force_predicted_particles", "NeRD particles"),
            ("force_reference_particles", "Reference MPM particles"),
        ):
            values = meta[key]
            force_rows.append(
                (
                    cases[case],
                    label,
                    *(f"{v:.4g}" for v in values["component_rmse_n"]),
                    f"{values['vector_rmse_n']:.4g}",
                    ", ".join(f"{v:+.4g}" for v in values["component_bias_n"]),
                )
            )
    content += table(
        "Table 8: Raw 200 Hz force errors over each complete reference recording.",
        (
            "Case",
            "Estimator input",
            "Fx RMSE (N)",
            "Fy RMSE (N)",
            "Fz RMSE (N)",
            "Vector RMSE (N)",
            "Signed bias: Fx, Fy, Fz (N)",
        ),
        force_rows,
    )
    content += "<p>Native MPM material forces are the reference; direct rigid-tray forces are excluded. "
    content += "Neither estimated forces nor errors are filtered. Matching videos and force plots are on the case pages above. "
    content += "Chamfer and Wasserstein distances remain optional, unmeasured diagnostics; they are not counted as completed metrics.</p>"
    return section("4. Results for each proposed core metric", content)


def hold_results(root, output, checkpoint, digest, table, section):
    """Validate and publish new temporal metrics without exporting particle states."""
    rows, mass_rows, diagnostics, summaries = [], [], [], {}
    for case, label in HOLD_CASES.items():
        folder = root / case
        summary = json.loads((folder / "consistency_summary.json").read_text())
        provenance = json.loads((folder / "consistency_provenance.json").read_text())
        video = json.loads((folder / "video_provenance.json").read_text())
        if summary["checkpoint_sha256"] != checkpoint:
            raise ValueError(f"Temporal metric checkpoint mismatch: {case}")
        for key in ("checkpoint_sha256", "reference_sha256", "prediction_sha256"):
            if summary[key] != provenance[key] or summary[key] != video[key]:
                raise ValueError(f"Temporal metrics and video do not use the same states: {case}")
        for name, expected in provenance["files_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"Temporal artifact changed: {folder / name}")
            shutil.copy2(folder / name, output / case / name)
        shutil.copy2(folder / "consistency_provenance.json", output / case / "consistency_provenance.json")
        summaries[case] = summary
        for method, method_label in (("reference", "Implicit MPM"), ("predicted", "Particle NeRD")):
            values = summary[method]
            final = values["final"]
            surface = final.get("surface_mean_mm")
            rows.append(
                (
                    label,
                    method_label,
                    f"{final['rms_displacement_mm']:.4g}",
                    f"{final['mean_vertical_change_mm']:+.4g}",
                    f"{values['late_vertical_drift_rate_mm_s']:+.4g}",
                    "Unavailable" if surface is None else f"{surface:.4g}",
                    f"{final['temporal_mass_discrepancy']:.4g}",
                )
            )
            fraction = values["final_below_body_mass_fraction"]
            mass_rows.append(
                (
                    label,
                    method_label,
                    f"{values['initial_below_body_mass_kg']:.4g}",
                    f"{final['below_body_mass_kg']:.4g}",
                    "Undefined" if fraction is None else f"{100 * fraction:.4g}",
                    f"{final['below_floor_mass_percent']:.4g}",
                    f"{final['below_floor_1cm_mass_percent']:.4g}",
                    f"{final['minimum_particle_z_m']:.4g}",
                )
            )
        diagnostics.append(f"{label}: {len(summary['surface_failures'])} temporal surface failures.")
    (output / "consistency_results.json").write_text(json.dumps(summaries, indent=2) + "\n")
    content = (
        "<p><strong>New experiment:</strong> box, sphere and cylinder each follow a vertical insertion from 1–2 s "
    )
    content += (
        "and are then held at an exactly fixed pose through 23 s. Translation, rotation, and input linear/angular "
    )
    content += (
        "velocities are verified constant/zero throughout measurement. There is one paired run per shape, seed 42.</p>"
    )
    content += "<p>Each method evolves from identical particles at 0 s. Measurements compare each method with its own "
    content += "state at 3 s, then track 20 seconds of change. The 3 s anchor neither resets particles nor establishes "
    content += "equilibrium. This is a prescribed-object material test, not freely balancing locomotion. "
    content += "The fixed pose can remain suspended if sand moves away. No retraining or stabilization was added.</p>"
    content += "<p>Particle drift is the physical-mass-weighted RMS displacement. Signed vertical drift is the "
    content += "physical-mass-weighted mean change in world z (negative downward). The late drift rate is the "
    content += "least-squares slope over 13–23 s using every 200 Hz sample. Surface change uses the same method’s "
    content += (
        "3 s surface without alignment; mass change is the L1 difference of grid masses divided by total mass.</p>"
    )
    content += table(
        "Table 9: Spatio-temporal consistency, measured from 3 to 23 s; one run per shape.",
        (
            "Case",
            "Solver",
            "Final RMS drift (mm)",
            "Final mean Δz (mm)",
            "Late Δz rate (mm/s)",
            "Final mean surface change (mm)",
            "Final grid-mass change",
        ),
        rows,
    )
    content += table(
        "Table 10: Material beneath the fixed object and boundary leakage at 23 s.",
        (
            "Case",
            "Solver",
            "Beneath object at 3 s (kg)",
            "At 23 s (kg)",
            "Remaining (%)",
            "Below floor (%)",
            "More than 1 cm below floor (%)",
            "Lowest particle z (m)",
        ),
        mass_rows,
    )
    content += (
        "<p><strong>Reference limitation:</strong> the MPM bed also continues to settle. In the box and cylinder "
    )
    content += "recordings, one MPM particle escapes far below the tray: z = −114.40 m for box and −126.67 m for cylinder. "
    content += "That single particle contributes 99.40% and 99.51% of squared displacement, respectively. "
    content += "Their large RMS values therefore do not mean that the bulk MPM bed drifts more than NeRD. "
    content += "No particle is removed from any temporal metric. These runs diagnose drift and boundary leakage; "
    content += "they do not demonstrate a correct static equilibrium for either method.</p>"
    content += "<p>The original paired field evaluator exceeded its 20-million-node limit for box and cylinder; "
    content += "those paired field metrics remain unavailable. The separate temporal analysis uses the identical "
    content += "0.02 m CIC and density-surface rules with an explicit 100-million-node limit. "
    content += "It includes every particle and records failures. " + " ".join(diagnostics) + "</p>"
    content += "<p>Each case page contains its exact 50 fps video, raw forces and full temporal curves:</p><ul>"
    content += "".join(f'<li><a href="{case}/">{label}</a></li>' for case, label in HOLD_CASES.items()) + "</ul>"
    content += '<figure class="result-figure"><div class="plot-scroll"><a href="sphere-fixed-hold/spatiotemporal_consistency.svg">'
    content += '<img src="sphere-fixed-hold/spatiotemporal_consistency.svg" alt="Sphere hold: measured sand drift and material loss"></a></div>'
    content += "<figcaption><strong>Figure 7:</strong> Sphere fixed-object hold. (a) RMS particle drift; (b) signed mean vertical drift; "
    content += "(c) mean and 95th-percentile surface change; (d) normalized grid-mass change; (e) material beneath the object; "
    content += "(f) mass below the tray floor. Every curve uses the same recording as the "
    content += '<a href="sphere-fixed-hold/#hold-video">sphere video</a>; the video includes insertion, and these curves cover 3–23 s. '
    content += '<a href="sphere-fixed-hold/spatiotemporal_consistency.png">PNG</a> · '
    content += '<a href="sphere-fixed-hold/spatiotemporal_consistency.pdf">PDF</a></figcaption></figure>'
    content += '<p><a href="consistency_results.json">Exact measured summaries and settings</a>. '
    content += "Full scalar time series, input hashes, and resolution sensitivity are linked on each case page.</p>"
    return section("5. Spatio-temporal consistency with a stationary object", content)
