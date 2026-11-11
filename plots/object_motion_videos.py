"""Publish videos of the exact recorded trajectories used by the object metrics."""

import html
import json
import os
import shutil
from collections import Counter

MOTIONS = {
    "horizontal": (
        "Horizontal motion",
        "The object is lowered into the sand and follows a prescribed horizontal path.",
    ),
    "vertical": (
        "Vertical motion",
        "The object is lowered into the sand, then moves down and up along a prescribed vertical path.",
    ),
    "circle": (
        "Circular motion",
        "The object is lowered into the sand and follows one prescribed horizontal circle.",
    ),
    "hold": (
        "Holding",
        "The object is lowered into the sand and its pose is then held fixed.",
    ),
    "drop": (
        "Drop",
        "The object is released at 1 s and responds to gravity and sand contact in the MPM recording.",
    ),
    "slide": (
        "Slide",
        "The object is released at 1 s with a horizontal velocity and then responds to gravity and sand contact.",
    ),
    "throw": (
        "Throw",
        "The object is released at 1 s with horizontal and upward velocity and then responds to gravity and sand contact.",
    ),
    "tumble": (
        "Tumble",
        "The object is released at 1 s with horizontal and angular velocity and then responds to gravity and sand contact.",
    ),
}


def install_test_videos(source, output, report, digest):
    """Verify clip identities against the plotted test set before copying public assets."""
    manifest = json.loads((source / "manifest.json").read_text())
    records = {r["id"]: r for r in report["episodes"]}
    first = {}
    counts = Counter((r["shape"], r["motion"]) for r in records.values())
    for record in sorted(records.values(), key=lambda r: r["id"]):
        first.setdefault((record["shape"], record["motion"]), record["id"])
    if {c["episode_id"] for c in manifest["clips"]} != set(first.values()):
        raise ValueError(
            "Motion videos must include the first test episode in every shape/motion group"
        )
    if manifest["checkpoint_sha256"] != report["checkpoint_sha256"]:
        raise ValueError("Motion videos use a different model checkpoint")
    for clip in manifest["clips"]:
        case = clip["episode_id"]
        record = records[case]
        for key in ("shape", "motion", "motion_mode"):
            if clip[key] != record[key]:
                raise ValueError(
                    f"Motion description differs from plotted trajectory: {case}"
                )
        folder = source / case
        provenance = json.loads((folder / "video_provenance.json").read_text())
        if clip["reference_sha256"] != record["reference_sha256"]:
            raise ValueError(
                f"Motion recording differs from the plotted trajectory: {case}"
            )
        for key in ("checkpoint_sha256", "reference_sha256", "prediction_sha256"):
            if clip[key] != provenance[key]:
                raise ValueError(f"Video identity mismatch: {case}, {key}")
        if (
            provenance["native_dt_s"] != 0.005
            or provenance["fps"] != 50
            or provenance["state_indices"] != list(range(4, 1201, 4))
        ):
            raise ValueError(
                f"Video must show the full six-second trajectory at 50 fps: {case}"
            )
        destination = output / "motions" / case
        destination.mkdir(parents=True)
        for name, key in (
            ("comparison.mp4", "video_sha256"),
            ("comparison_poster.png", "poster_sha256"),
            ("video_provenance.json", "video_provenance_sha256"),
        ):
            if digest(folder / name) != clip[key]:
                raise ValueError(f"Video artifact changed: {case}/{name}")
            shutil.copy2(folder / name, destination / name)
        clip["group_trajectory_count"] = counts[(record["shape"], record["motion"])]
        clip["motion_label"], clip["motion_description"] = MOTIONS[record["motion"]]
        clip["model_label"] = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    (output / "motion_video_provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def motion_gallery(manifest):
    """Display one full-width paired video with explicit shape and motion selectors."""
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    initial = next(c for c in manifest["clips"] if c["episode_id"] == "box-0009")
    options = "".join(
        f'<option value="{key}">{html.escape(value[0])}</option>'
        for key, value in MOTIONS.items()
    )
    payload = json.dumps(manifest["clips"]).replace("</", "<\\/")
    return f"""<div id="test-motion-videos" class="motion-gallery">
<h3>Recorded motions for metrics 1–4</h3>
<p>Each shape has 20 held-out six-second trajectories: three each for horizontal motion, vertical motion, circles, and holding; two each for drops, slides, throws, and tumbles. The plots include all 60 trajectories.</p>
<p>Choose a shape and motion below. These 24 clips show the first held-out trajectory in each shape/motion group. Each clip is one of the recorded trajectories included in the plots. Both panels replay the same MPM body motion while their particles follow their respective saved states.</p>
<div class="motion-controls"><label for="motion-shape">Object <select id="motion-shape"><option value="box">Box</option><option value="sphere">Sphere</option><option value="cylinder">Cylinder</option></select></label>
<label for="motion-kind">Motion <select id="motion-kind">{options}</select></label></div>
<figure class="video-figure"><video id="motion-video" controls playsinline preload="metadata" poster="motions/{initial["episode_id"]}/comparison_poster.png"><source src="motions/{initial["episode_id"]}/comparison.mp4" type="video/mp4"></video>
<figcaption><strong>Video 1:</strong> <span id="motion-description" aria-live="polite">Box — horizontal motion, episode box-0009. {MOTIONS["horizontal"][1]} Left: implicit MPM. Right: {html.escape(model_label)}. Saved states at 50 fps, simulation time 0.02–6 s.</span> <a id="motion-provenance" href="motions/box-0009/video_provenance.json">Recording identity and timestamps</a>.</figcaption></figure>
<p><a href="motion_video_provenance.json">All clip identities and selection rule</a>. The dynamic objects respond to forces during the original MPM recording; during {html.escape(model_label)} evaluation, their recorded poses are imposed.</p>
<script type="application/json" id="motion-video-data">{payload}</script>
</div>"""


def reference_videos(cases):
    """Embed the four force-reference recordings used in metric 5."""
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    content = '<div class="reference-video-gallery"><h3>Recorded motions for contact-force accuracy</h3>'
    for index, (case, title) in enumerate(cases.items(), 2):
        description = (
            "The box is lowered into the sand and follows a prescribed horizontal path for this six-second recording."
            if case == "box-horizontal-flat"
            else "The object is released at 1 s and settles dynamically in the MPM recording. Support is measured over 20 s."
        )
        content += f'''<details class="reference-motion"><summary>{html.escape(title)}</summary><p>{description}</p>
<figure class="video-figure"><video controls playsinline preload="none" poster="{case}/comparison_poster.png"><source src="{case}/comparison.mp4" type="video/mp4"></video>
<figcaption><strong>Video {index}:</strong> {html.escape(title)}. Left: implicit MPM. Right: {html.escape(model_label)} replaying the same body motion. This exact recording supplies the corresponding force results below. <a href="{case}/">Force time series and recording conditions</a>.</figcaption></figure></details>'''
    return content + "</div>"
