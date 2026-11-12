"""Publish videos of the exact recorded trajectories used by the object and free-fall metrics."""

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
        "Released in mid-air and responds to gravity and sand/floor contact in the MPM recording.",
    ),
    "slide": (
        "Slide",
        "Released with a horizontal velocity and then responds to gravity and sand/floor contact.",
    ),
    "throw": (
        "Throw",
        "Released with horizontal and upward velocity and then responds to gravity and sand/floor contact.",
    ),
    "tumble": (
        "Tumble",
        "Released with horizontal and angular velocity and then responds to gravity and sand/floor contact.",
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
    """Display one full-width paired video with Object, Motion type (Kinematic / Dynamic / Free-fall), and Motion selectors."""
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    initial = next((c for c in manifest["clips"] if c["episode_id"] == "box-0009"), manifest["clips"][0])
    kinematic_options = "".join(
        f'<option value="{key}">{html.escape(MOTIONS[key][0])}</option>'
        for key in ("horizontal", "vertical", "circle", "hold")
    )
    has_freefall = any(c["shape"] == "freefall" for c in manifest["clips"])
    freefall_mode_opt = (
        '<option value="freefall">Free-fall sand (flat ground, no object)</option>'
        if has_freefall
        else ""
    )
    payload = json.dumps(manifest["clips"]).replace("</", "<\\/")
    return f"""<div id="test-motion-videos" class="motion-gallery">
<h3>Recorded motions for metrics 1–4</h3>
<p>Each elementary object (<strong>Box</strong>, <strong>Sphere</strong>, <strong>Cylinder</strong>) is evaluated across 20 held-out six-second trajectories: 12 <strong>Kinematic (prescribed object pose)</strong> trajectories (3 each for <em>Horizontal</em>, <em>Vertical</em>, <em>Circular</em>, and <em>Holding</em>) and 8 <strong>Dynamic (object gravity &amp; contact)</strong> trajectories (2 each for <em>Drop</em>, <em>Slide</em>, <em>Throw</em>, and <em>Tumble</em>). In addition, <strong>Free-fall sand (flat ground, no object)</strong> evaluates 8 held-out six-second trajectories of unconfined 3D mid-air sand blocks dropped onto open flat ground.</p>
<div class="motion-controls">
<label id="motion-shape-wrap" for="motion-shape">Object <select id="motion-shape"><option value="box">Box</option><option value="sphere">Sphere</option><option value="cylinder">Cylinder</option></select></label>
<label for="motion-mode">Motion type <select id="motion-mode"><option value="prescribed">Kinematic (prescribed object pose)</option><option value="dynamic">Dynamic (object gravity &amp; contact)</option>{freefall_mode_opt}</select></label>
<label for="motion-kind">Motion <select id="motion-kind">{kinematic_options}</select></label>
</div>
<figure class="video-figure"><video id="motion-video" controls playsinline preload="metadata" poster="motions/{initial["episode_id"]}/comparison_poster.png"><source src="motions/{initial["episode_id"]}/comparison.mp4" type="video/mp4"></video>
<figcaption><strong>Video 1:</strong> <span id="motion-description" aria-live="polite">Box [Kinematic: prescribed pose] — horizontal motion, episode box-0009. {MOTIONS["horizontal"][1]} Left: implicit MPM. Right: {html.escape(model_label)}. Saved states at 50 fps, simulation time 0.02–6 s.</span></figcaption></figure>
<script type="application/json" id="motion-video-data">{payload}</script>
</div>"""


def reference_videos(cases):
    """Embed the three dynamic-object force-reference recordings without extra links."""
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    content = '<div class="reference-video-gallery"><h3>Recorded dynamic motions for contact-force accuracy</h3>'
    for index, (case, title) in enumerate(cases.items(), 1):
        description = "The 2 kg dynamic object is released at 1 s and settles dynamically in the MPM recording (Motion: Dynamic Drop & Support, 0–20 s)."
        content += f'''<details class="reference-motion"><summary>{html.escape(title)}</summary><p>{description}</p>
<figure class="video-figure"><video controls playsinline preload="none" poster="{case}/comparison_poster.png"><source src="{case}/comparison.mp4" type="video/mp4"></video>
<figcaption><strong>Video {index}:</strong> {html.escape(title)}. Left: implicit MPM. Right: {html.escape(model_label)} replaying the same body motion.</figcaption></figure></details>'''
    return content + "</div>"
