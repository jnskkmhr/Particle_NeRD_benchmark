"""Label object-motion constraints and publish matched fully dynamic results."""

import html
import json
import os
import shutil

from plot_object_closed_loop import caption, digest, export_curves, plot


def evaluation_modes(table, section, include_dynamic):
    rows = [
        (
            "60 held-out trajectories: horizontal, vertical, circular, holding (36)",
            "Kinematic / prescribed",
            "Kinematic replay",
            "No",
        ),
        (
            "60 held-out trajectories: drop, slide, throw, tumble (24)",
            "Dynamic after release at 1 s",
            "Kinematic replay",
            "No",
        ),
        (
            "Force references: box, sphere, cylinder drop-and-support (3)",
            "Dynamic after release at 1 s",
            "Kinematic replay",
            "No",
        ),
        (
            "Stationary-object consistency: box, sphere, cylinder (3)",
            "Kinematic insertion and fixed pose",
            "Kinematic replay",
            "No",
        ),
    ]
    if include_dynamic:
        rows.append(
            (
                "Fully dynamic comparison: box, sphere, cylinder drop-and-support (3)",
                "Dynamic after release at 1 s; same MPM support references",
                "Dynamic simulation",
                "Yes: estimated force and torque",
            )
        )
    return section(
        "Kinematic and dynamic evaluation sets",
        "<p><strong>Kinematic replay</strong> means the object follows recorded MPM poses, even when those poses originally came from a dynamic simulation. "
        "NeRD particles evolve from their own predicted history in both modes. "
        "<strong>Dynamic simulation</strong> means the object responds to gravity, rigid tray contacts, and NeRD’s estimated material force and torque.</p>"
        + "<p>Kinematic replay describes how object motion is imposed. Force reconstruction for dynamic-source recordings still uses the physical body mass and inertia.</p>"
        + table(
            "Table 2: Object motion during reference generation and NeRD evaluation. Counts denote trajectories, not training updates.",
            (
                "Evaluation set",
                "MPM reference generation",
                "NeRD object motion",
                "NeRD material force changes object motion?",
            ),
            rows,
        )
        + (
            '<p><a href="#fully-dynamic">View the three fully dynamic comparisons below.</a></p>'
            if include_dynamic
            else ""
        ),
    )


def dynamic_results(source, output, checkpoint, section):
    parts = [
        "<p>Three 20-second comparisons, one each for box, sphere, and cylinder. These use the same frozen checkpoint and the same initial states as the existing MPM drop-and-support references. "
        "The 2 kg object starts with its center at z = 0.25 m. An upward force balances gravity until 1 s, then is removed. "
        "After release, each method evolves its own object motion and particles; no recorded MPM body motion or future particle state is supplied to NeRD.</p>",
        "<p>NeRD predicts all 20,646 particles every 0.005 s. The existing analytical estimator supplies world-frame material force and center-of-mass torque. "
        "Both are applied over the two 0.0025 s MJWarp rigid substeps of the next exchange, matching the reference’s lagged coupling. "
        "Normal and tangential effective material mass factors remain 10, stabilization remains 0.01, and collider friction remains 1.0. "
        "No Top-K selection, displacement gate, particle projection, force filtering, or native MPM solve is used in the NeRD run.</p>",
        "<p>These are three individual coupled-simulation examples. Differences include particle prediction, force reconstruction, and the resulting differences in object motion. "
        "They are reported separately from metrics 1–6, which impose matched object motion.</p>",
    ]
    parts.append(
        "<p><strong>Observed result:</strong> In all three NeRD runs, the particles keep moving downward and eventually lose contact with the object. The object then rests against the rigid tray’s contact boundary, while the estimated sand force falls to zero. MPM retains sand support over the same 20 seconds. The recordings below show this behavior in full; they do not establish stable sand support by NeRD.</p>"
    )
    for number, shape in enumerate(("box", "sphere", "cylinder"), start=9):
        case = source / f"{shape}-dynamic"
        destination = output / "dynamic" / shape
        meta = json.loads((case / "evaluation.json").read_text())
        video = json.loads((case / "video_provenance.json").read_text())
        if meta["checkpoint_sha256"] != checkpoint or meta["native_mpm_steps"] != 0:
            raise ValueError("Dynamic experiment checkpoint or material solver differs")
        for key in ("reference_sha256", "prediction_sha256", "checkpoint_sha256"):
            if video[key] != meta[key]:
                raise ValueError(
                    f"Video and dynamic simulation disagree: {shape}/{key}"
                )
        if video["fps"] != 50 or video["state_indices"] != list(
            range(4, meta["completed_steps"] + 1, 4)
        ):
            raise ValueError("Expected the complete 50 fps simulation video")
        if digest(case / "comparison.mp4") != video["video_sha256"]:
            raise ValueError("Video hash differs from provenance")
        data = export_curves(case, destination)
        plot(destination, number)
        for name in (
            "comparison.mp4",
            "comparison_poster.png",
            "video_provenance.json",
        ):
            shutil.copy2(case / name, destination / name)
        public = {
            k: v
            for k, v in meta.items()
            if k not in ("reference_path", "implementation_source_sha256")
        }
        public["implementation_source_sha256"] = {
            path.rsplit("/", 1)[-1]: value
            for path, value in meta["implementation_source_sha256"].items()
        }
        (destination / "simulation_provenance.json").write_text(
            json.dumps(public, indent=2) + "\n"
        )
        model_label = os.environ.get("BENCHMARK_MODEL_LABEL", "Particle NeRD")
        prefix = f"dynamic/{shape}/"
        parts.append(
            f"<h3>{shape.title()}: dynamic drop and support</h3>"
            f'<video controls playsinline preload="metadata" poster="{prefix}comparison_poster.png" style="display:block;width:100%;height:auto">'
            f'<source src="{prefix}comparison.mp4" type="video/mp4"></video>'
            f"<p>Left: Implicit MPM. Right: {html.escape(model_label)}. Both objects move dynamically; their trajectories can diverge. Playback is real time at 50 fps.</p>"
            f'<figure><a href="{prefix}dynamic_response.png"><img src="{prefix}dynamic_response.svg" alt="{shape.title()} dynamic object position and Fx Fy Fz over time" style="width:100%;height:auto"></a>'
            f"<figcaption>{html.escape(caption(shape, number))}</figcaption></figure>"
            f'<p><a href="{prefix}dynamic_response.png">PNG</a> · <a href="{prefix}dynamic_response.pdf">PDF</a> · '
            f'<a href="{prefix}simulation_provenance.json">Simulation conditions</a> · <a href="{prefix}video_provenance.json">Video provenance</a></p>'
        )
        if data["contact_iteration_limit_count"]:
            parts.append(
                f"<p>The contact solver reached its iteration limit in {data['contact_iteration_limit_count']} of {meta['completed_steps']} exchanges; these samples remain in the plots and video.</p>"
            )
    return (
        '<div id="fully-dynamic">'
        + section("Fully dynamic simulation with NeRD force feedback", "".join(parts))
        + "</div>"
    )
