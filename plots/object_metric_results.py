"""Expose measured metric results and fixed-object diagnostics in the public report."""

import html
import json
import re
import shutil
from pathlib import Path

from plot_object_metric_comparisons import PLOTS, figure_html

HOLD_CASES = {
    f"{shape}-fixed-hold": f"{shape.capitalize()}: fixed-object hold"
    for shape in ("box", "sphere", "cylinder")
}


METRIC_HEADINGS = {
    int(number): f"{number}. {title}"
    for number, title in re.findall(
        r"^## ([1-6])\. (.+)$",
        (Path(__file__).resolve().parents[1] / "metrics/README.md").read_text(),
        re.MULTILINE,
    )
}


def metric_section(number, content, section, *, numbered=True):
    """Use the canonical README heading and link ONLY to its definition in metrics/README.md."""
    title = METRIC_HEADINGS[number]
    anchor = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
    definition = (
        '<p><a href="https://github.com/jnskkmhr/Particle_NeRD_benchmark/blob/main/'
        f'metrics/README.md#{anchor}">Definition in metrics/README.md</a></p>'
    )
    return section(title if numbered else title.split(". ", 1)[1], definition + content)


def metric_results(report, section):
    """Group measured plots under the corresponding README metric headings (separate object vs. motion plots)."""
    total = report["groups"]["combined"]
    introduction = (
        f"<p>{total['complete_trajectories']}/{total['trajectories']} held-out six-second trajectories completed "
        "(60 elementary-object trajectories across Box, Sphere, Cylinder: 12 Kinematic [Horizontal, Vertical, Circle, Hold] "
        "and 8 Dynamic [Drop, Slide, Throw, Tumble] per shape; plus 8 unconfined Free-fall sand trajectories on flat ground). "
        "Each dot is one trajectory summary. Boxes show median and interquartile range; diamonds mark means. "
        "Whiskers extend to 1.5 IQR and every measured value is plotted, including outliers.</p>"
    )
    groups = (
        (1, ("test_particle_errors", "test_particle_errors_by_motion"), introduction),
        (
            2,
            ("test_surface_errors", "test_surface_errors_by_motion"),
            (
                "<p>The mean and 95th-percentile symmetric surface distance measure the reconstructed sand shape "
                "on a 0.02 m measurement grid (separate from the 0.04 m MPM simulation grid).</p>"
            ),
        ),
        (
            3,
            ("test_material_distribution", "test_material_distribution_by_motion"),
            (
                "<p>Material-mass discrepancy on a common 0.02 m grid. "
                "For the local depletion diagnostic, the region is beneath the box, sphere, or cylinder.</p>"
            ),
        ),
        (
            4,
            ("test_velocity_energy", "test_velocity_energy_by_motion"),
            "<p>Velocity-field RMSE, shared occupied-cell reference-mass coverage, and particle kinetic energy.</p>",
        ),
    )
    return "".join(
        metric_section(
            number,
            description
            + "".join(
                "<h3>" + html.escape(PLOTS[stem][1]) + "</h3>" + figure_html(stem)
                for stem in stems
            ),
            section,
        )
        for number, stems, description in groups
    )


def hold_results(root, output, checkpoint, digest, table, section):
    """Validate and publish temporal consistency metrics without extra links."""
    diagnostics, summaries = [], {}
    for case, label in HOLD_CASES.items():
        folder = root / case
        summary = json.loads((folder / "consistency_summary.json").read_text())
        provenance = json.loads((folder / "consistency_provenance.json").read_text())
        video = json.loads((folder / "video_provenance.json").read_text())
        if summary["checkpoint_sha256"] != checkpoint:
            raise ValueError(f"Temporal metric checkpoint mismatch: {case}")
        for key in ("checkpoint_sha256", "reference_sha256", "prediction_sha256"):
            if summary[key] != provenance[key] or summary[key] != video[key]:
                raise ValueError(
                    f"Temporal metrics and video do not use the same states: {case}"
                )
        for name, expected in provenance["files_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"Temporal artifact changed: {folder / name}")
            shutil.copy2(folder / name, output / case / name)
        shutil.copy2(
            folder / "consistency_provenance.json",
            output / case / "consistency_provenance.json",
        )
        summaries[case] = summary
        diagnostics.append(
            f"{label}: {len(summary['surface_failures'])} temporal surface failures."
        )
    (output / "consistency_results.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    content = (
        "<p><strong>Kinematic Motion (Vertical Insertion 1–2 s + Fixed-Pose Hold 2–23 s):</strong> "
        "Box, Sphere, and Cylinder each follow a prescribed vertical insertion from 1–2 s and are then held at an "
        "exactly fixed pose through 23 s. Translation, rotation, and input linear/angular velocities are verified "
        "constant/zero throughout measurement. Each method is compared with its own state at 3 s through 23 s.</p>"
    )
    content += figure_html("hold_summary")
    content += figure_html("hold_material")
    return metric_section(6, content, section)
