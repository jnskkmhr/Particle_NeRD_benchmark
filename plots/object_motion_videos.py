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

SHAPE_LABELS = {
    "box": "Box",
    "sphere": "Sphere",
    "cylinder": "Cylinder",
    "freefall": "Free-fall sand",
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
    """Display one full-width paired video with explicit scenario, motion-type (kinematic/dynamic), and motion selectors."""
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    initial = next((c for c in manifest["clips"] if c["episode_id"] == "box-0009"), manifest["clips"][0])
    present_shapes = []
    for s in ("box", "sphere", "cylinder", "freefall"):
        if any(c["shape"] == s for c in manifest["clips"]):
            present_shapes.append(s)
    shape_options = "".join(
        f'<option value="{s}">{html.escape(SHAPE_LABELS.get(s, s.title()))}</option>'
        for s in present_shapes
    )
    kinematic_options = "".join(
        f'<option value="{key}">{html.escape( MOTIONS[key][0] )}</option>'
        for key in ("horizontal", "vertical", "circle", "hold")
    )
    payload = json.dumps(manifest["clips"]).replace("</", "<\\/")
    total_clips = len(manifest["clips"])
    has_freefall = "freefall" in present_shapes
    intro_text = (
        "Each elementary object (Box, Sphere, Cylinder) has 20 held-out six-second trajectories: "
        "12 <strong>kinematic</strong> trajectories (3 each for horizontal motion, vertical motion, circular motion, and holding, where object poses are prescribed) "
        "and 8 <strong>dynamic</strong> trajectories (2 each for drop, slide, throw, and tumble, where the 2 kg object is released at 1 s and responds to gravity and sand contact)."
    )
    if has_freefall:
        intro_text += (
            " <strong>Free-fall sand</strong> adds 8 held-out six-second dynamic trajectories (2 each for drop, slide, throw, and tumble of unconfined 3D mid-air sand blocks/columns collapsing onto the tray floor without a rigid object)."
        )
    return f"""<div id="test-motion-videos" class="motion-gallery">
<h3>Recorded motions for metrics 1–4</h3>
<p>{intro_text}</p>
<p>Select a scenario, toggle <strong>Motion type</strong> between <strong>Kinematic (prescribed)</strong> and <strong>Dynamic (gravity &amp; contact)</strong>, and choose a motion below ({total_clips} clips total, showing the first held-out trajectory in each group).</p>
<div class="motion-controls">
<label for="motion-shape">Scenario <select id="motion-shape">{shape_options}</select></label>
<label for="motion-mode">Motion type <select id="motion-mode"><option value="prescribed">Kinematic (prescribed)</option><option value="dynamic">Dynamic (gravity &amp; contact)</option></select></label>
<label for="motion-kind">Motion <select id="motion-kind">{kinematic_options}</select></label>
</div>
<figure class="video-figure"><video id="motion-video" controls playsinline preload="metadata" poster="motions/{initial["episode_id"]}/comparison_poster.png"><source src="motions/{initial["episode_id"]}/comparison.mp4" type="video/mp4"></video>
<figcaption><strong>Video 1:</strong> <span id="motion-description" aria-live="polite">Box [Kinematic: prescribed pose] — horizontal motion, episode box-0009. {MOTIONS["horizontal"][1]} Left: implicit MPM. Right: {html.escape(model_label)}. Saved states at 50 fps, simulation time 0.02–6 s.</span> <a id="motion-provenance" href="motions/box-0009/video_provenance.json">Recording identity and timestamps</a>.</figcaption></figure>
<p><a href="motion_video_provenance.json">All clip identities and selection rule</a>.</p>
<script type="application/json" id="motion-video-data">{payload}</script>
</div>"""


def reference_videos(cases):
    """Embed the three dynamic-object force-reference recordings."""
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
    content = '<div class="reference-video-gallery"><h3>Recorded dynamic motions for contact-force accuracy</h3>'
    for index, (case, title) in enumerate(cases.items(), 1):
        description = "The 2 kg dynamic object is released at 1 s and settles dynamically in the MPM recording. Support is measured over 20 s."
        content += f'''<details class="reference-motion"><summary>{html.escape(title)}</summary><p>{description}</p>
<figure class="video-figure"><video controls playsinline preload="none" poster="{case}/comparison_poster.png"><source src="{case}/comparison.mp4" type="video/mp4"></video>
<figcaption><strong>Video {index}:</strong> {html.escape(title)}. Left: implicit MPM. Right: {html.escape(model_label)} replaying the same body motion. This exact recording supplies the corresponding force results below. <a href="{case}/">Force time series and recording conditions</a>.</figcaption></figure></details>'''
    return content + "</div>"
