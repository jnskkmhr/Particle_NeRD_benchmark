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


METRIC_HEADINGS = dict(
    (int(number), f"{number}. {title}")
    for number, title in re.findall(
        r"^## ([1-6])\. (.+)$",
        (Path(__file__).resolve().parents[1] / "metrics/README.md").read_text(),
        re.MULTILINE,
    )
)


def metric_section(number, content, section, *, numbered=True):
    """Use the canonical README heading and link to its definition."""
    title = METRIC_HEADINGS[number]
    anchor = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
    definition = (
        '<p><a href="https://github.com/jnskkmhr/Particle_NeRD_benchmark/blob/main/'
        f'metrics/README.md#{anchor}">Definition in metrics/README.md</a></p>'
    )
    return section(title if numbered else title.split(". ", 1)[1], definition + content)


def metric_results(report, section):
    """Group measured plots under the corresponding README metric headings."""
    total = report["groups"]["combined"]
    introduction = (
        f"<p>{total['complete_trajectories']}/{total['trajectories']} held-out six-second trajectories completed. "
        "Each dot is one trajectory summary. Boxes show median and interquartile range; diamonds mark means. "
        "Whiskers extend to 1.5 IQR and every measured value is plotted, including outliers. "
        "An empty particle region remains unavailable. No smoothing or time shifting is applied.</p>"
        '<p><a href="test_report.json">Exact numerical results and availability counts</a> · '
        '<a href="comparison_plot_provenance.json">Plot input and output hashes</a>. '
        "Numerical completion does not establish physical accuracy. These are errors of one frozen object model "
        "against MPM.</p>"
    )
    groups = (
        (1, ("test_particle_errors", "test_regions_baseline"), introduction),
        (
            2,
            ("test_surface_errors",),
            "<p>The mean and 95th-percentile symmetric surface distance measure the reconstructed sand shape. "
            "The plotted measurements use a 0.02 m grid for reconstructing the sand surface from recorded particles. "
            "This is a measurement setting, separate from the 0.04 m MPM simulation grid. "
            "The README also requests a reconstruction-resolution sensitivity check: its 0.03 m results remain "
            '<a href="test_report.json">in the downloadable results</a>, outside the main metric plot.</p>',
        ),
        (
            3,
            ("test_material_distribution",),
            "<p>These object tests use the same material-mass discrepancy on a common grid. "
            "For the local depletion diagnostic, the region is beneath the box, sphere, or cylinder; "
            "these tests do not contain robot feet. Each case page defines the region.</p>",
        ),
        (4, ("test_velocity_energy",), ""),
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
    """Validate and publish new temporal metrics without exporting particle states."""
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
    content = "<p><strong>New experiment:</strong> box, sphere and cylinder each follow a vertical insertion from 1–2 s "
    content += "and are then held at an exactly fixed pose through 23 s. Translation, rotation, and input linear/angular "
    content += "velocities are verified constant/zero throughout measurement. There is one paired run per shape, seed 42.</p>"
    content += "<p>Each method evolves from identical particles at 0 s. Measurements compare each method with its own "
    content += "state at 3 s, then track 20 seconds of change. The 3 s anchor neither resets particles nor establishes "
    content += "equilibrium. This is a prescribed-object material test, not freely balancing locomotion. "
    content += "The fixed pose can remain suspended if sand moves away. No retraining or stabilization was added.</p>"
    content += "<p>Particle drift is the physical-mass-weighted RMS displacement. Signed vertical drift is the "
    content += "physical-mass-weighted mean change in world z (negative downward). The late drift rate is the "
    content += "least-squares slope over 13–23 s using every 200 Hz sample. Surface change uses the same method’s "
    content += "3 s surface without alignment; mass change is the L1 difference of grid masses divided by total mass.</p>"
    content += figure_html("hold_summary")
    content += figure_html("hold_material")
    content += "<p><strong>Reference limitation:</strong> the MPM bed also continues to settle. In the box and cylinder "
    content += "recordings, one MPM particle escapes far below the tray: z = −114.40 m for box and −126.67 m for cylinder. "
    content += "That single particle contributes 99.40% and 99.51% of squared displacement, respectively. "
    content += "Their large RMS values therefore do not mean that the bulk MPM bed drifts more than NeRD. "
    content += "No particle is removed from any temporal metric. These runs diagnose drift and boundary leakage; "
    content += (
        "they do not demonstrate a correct static equilibrium for either method.</p>"
    )
    content += "<p>The original paired field evaluator exceeded its 20-million-node limit for box and cylinder; "
    content += "those paired field metrics remain unavailable. The separate temporal analysis uses the identical "
    content += (
        "0.02 m CIC and density-surface rules with an explicit 100-million-node limit. "
    )
    content += (
        "It includes every particle and records failures. "
        + " ".join(diagnostics)
        + "</p>"
    )
    content += "<p>Each case page contains its exact 50 fps video, raw forces and full temporal curves:</p><ul>"
    content += (
        "".join(
            f'<li><a href="{case}/">{label}</a></li>'
            for case, label in HOLD_CASES.items()
        )
        + "</ul>"
    )
    content += '<p><a href="consistency_results.json">Exact measured summaries and settings</a>. '
    content += "Full scalar time series, input hashes, and resolution sensitivity are linked on each case page.</p>"
    return metric_section(6, content, section)
