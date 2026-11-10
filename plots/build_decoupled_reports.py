"""Build the three force evaluations and two particle evaluations from measured data."""

import html
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "g1_humanoid"
FULL = G1 / "data/full-20261109"
PROFILES = [
    ("forward", "Forward"),
    ("backward", "Backward"),
    ("left", "Left"),
    ("right", "Right"),
    ("forward_yaw_positive", "Forward + positive yaw"),
    ("forward_yaw_negative", "Forward + negative yaw"),
]
read = lambda p: json.loads(p.read_text())
replay = read(G1 / "data/current-20261109/metadata.json")
force_only = read(G1 / "data/estimator-only-20261109/metadata.json")
MPM = G1 / "data/full-mpm-20261109"
COMPARISON = G1 / "data/full-comparison-20261109"
summary = read(COMPARISON / "summary.json")


def table(caption, rows, heads=None):
    header = "" if not heads else "<thead><tr>" + "".join(f'<th scope="col">{v}</th>' for v in heads) + "</tr></thead>"
    body = "".join(
        '<tr><th scope="row">' + str(r[0]) + "</th>" + "".join("<td>" + str(v) + "</td>" for v in r[1:]) + "</tr>"
        for r in rows
    )
    return f'<div class="table-wrap"><table><caption>{caption}</caption>{header}<tbody>{body}</tbody></table></div>'


FIGURE_VERSION = "20261109-mpm-reference"


def figure(number, path, caption, alt):
    return f'''<figure class="result-figure"><div class="plot-scroll" tabindex="0" aria-label="Scrollable plot"><a href="{path}.svg?v={FIGURE_VERSION}" target="_blank" rel="noopener"><img src="{path}.svg?v={FIGURE_VERSION}" alt="{alt}" loading="lazy"></a></div><figcaption><strong>Figure {number}:</strong> {caption} <a href="{path}.pdf?v={FIGURE_VERSION}">PDF</a> · <a href="{path}.png?v={FIGURE_VERSION}">PNG</a> · Tap to enlarge.</figcaption></figure>'''


def video(number, path, caption):
    return f'''<figure class="video-figure"><video controls playsinline preload="metadata" poster="{path}.jpg"><source src="{path}.mp4" type="video/mp4"><a href="{path}.mp4">Download video</a></video><figcaption><strong>Video {number}:</strong> {caption} <a href="{path}.mp4">Download MP4</a>.</figcaption></figure>'''


def page(title, lead, nav, sections):
    sections = re.sub(
        r'<a href="([^"]+\.(?:h5|npz|npy|pt))">([^<]+)</a>',
        lambda match: "<span>" + match.group(2) + " (local archive)</span>",
        sections,
    )
    navigation = "".join(f'<a href="#{key}">{text}</a>' for key, text in nav)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · Particle NeRD</title><link rel="stylesheet" href="../../assets/slides.css?v=20261109-decoupled"></head>
<body class="benchmark-report"><header><a class="brand" href="../../index.html">Particle NeRD / Benchmarks</a><span class="archive">Current solver evaluation · 9 November 2026</span></header><main><div class="report-heading"><div class="eyebrow">G1 humanoid · current Particle NeRD implementation</div><h1>{title}</h1><p>{lead}</p><nav aria-label="Report sections">{navigation}</nav></div>{sections}</main><footer class="site-footer"><span class="footer-brand">Particle NeRD</span><span>G1 humanoid benchmarks · 11/09/2026</span></footer></body></html>"""


def section(id, title, body):
    return f'<section class="report-section" id="{id}"><h2>{title}</h2>{body}</section>'


replay_conditions = table(
    "Table 1: Conditions shared by the prescribed-motion tests.",
    [
        (
            "Recorded reference",
            "One implicit-MPM G1 trajectory, episode 0002; recording time 0.300–5.300 s, shown as elapsed 0–5 s. The robot first supports its weight, then takes steps.",
        ),
        ("Sand bed", "3 m × 3 m × 0.3 m; 88,360 particles; MPM voxel size 0.04 m; particle radius 15.63 mm."),
        (
            "Material",
            "Density 1,600 kg/m³; friction 0.48; Young’s modulus 10¹⁵ Pa; Poisson ratio 0.3; yield pressure 10¹⁵ Pa.",
        ),
        (
            "Current neural checkpoint",
            "<code>particle_nerd/checkpoint/best.pt</code>, SHA-256 <code>7eee570d75d0…</code>. The force-only test does not call the neural network.",
        ),
        (
            "Active-particle selection",
            "5,312 particles selected for local evaluation; 512 background particles supply spatial context; selection refreshed every four physics steps.",
        ),
        (
            "Particle initialization",
            "Three recorded MPM particle-position frames ending at recording time 0.300 s. The prescribed-motion tests do not start from an unsettled grid.",
        ),
        (
            "Force-reconstruction settings",
            "Effective material-response factors: normal 100, tangential 40, horizontal 8, upward 2; normal exponent 3. Gradient radius 0.065 m; contact band 0.08 m; stabilization 0.01; contact offset 0 m. Complete settings are archived.",
        ),
        (
            "Sampling",
            "0.005 s solver interval (200 Hz), 1,000 force samples per foot; video 50 fps at simulation speed. No temporal smoothing.",
        ),
        (
            "Body input",
            "Recorded collider poses and velocities. Proxy free velocities are reconstructed from the advanced recorded proxy velocity and the previous recorded MPM force/torque impulse. Both prescribed-motion tests therefore use prior MPM wrench information; the current target wrench is not supplied to the estimator. Recorded foot mass, inertia and collision margins are retained.",
        ),
    ],
)

full_meta = read(FULL / "forward/metadata.json")
full_conditions = table(
    "Full-simulation conditions, stated separately from the prescribed-motion tests.",
    [
        (
            "Controller",
            "MPM-trained G1 policy <code>jqur5jhm</code>, checkpoint 19,999; deterministic inference. Same policy and seed 42. Five commands have paired MPM references; the negative-yaw case retains its NeRD-only recording.",
        ),
        (
            "Dynamics",
            "For each of the five paired commands, two independent rollouts: MJWarp + implicit MPM, and MJWarp + current Particle NeRD. Identical rigid solver, foot proxy mapping and lagged coupling. Each policy receives its own simulated observations; its actions and robot trajectory may diverge. No prescribed robot or particle trajectory.",
        ),
        (
            "Initialization",
            "Bit-identical initial particle positions and rigid-body poses, checked from saved arrays. Robot base starts at (0, 0, 0.76) m. Both start from the fresh grid without settling warm-up. NeRD runs zero native MPM steps and loads no MPM trajectory; a runtime guard rejects native-MPM solver construction. The separate MPM run evolves the sand with native implicit MPM.",
        ),
        (
            "Sand bed / resolution",
            f"6 m × 6 m × 0.3 m, widened from the task’s 6 m × 2 m bed to allow lateral motion; {full_meta['particle_count']:,} particles. Both solvers use voxel size 0.02 m and identical material parameters: density 1,600 kg/m³, friction 0.48, Young’s modulus 10¹⁵ Pa, Poisson ratio 0.3 and yield pressure 10¹⁵ Pa. All material fields are checked against the metadata.",
        ),
        (
            "Particle selection / stabilization",
            "NeRD: 7,104 active particles and 512 background samples, with the current foot-speed and particle-height displacement rule enabled. MPM: every particle is evolved by the native solver; no neural network or displacement gate. Neither run adds a force filter.",
        ),
        (
            "Timing",
            "200 Hz physics, 50 Hz control, 50 fps video. Six NeRD episodes and five MPM references (negative-yaw reference omitted), requested duration 6 s per episode; stop on termination or when |base x| or |base y| exceeds 2.6 m. Plots retain each run’s full recorded duration. Paired videos stop at the shorter recording; individual complete videos remain linked. Failures are retained.",
        ),
        (
            "Implicit MPM settings",
            "Sparse grid, zero padding; APIC transfer; P0 strain and Q1 velocity basis; pic27 collider basis; forward collider velocity; at most 25 iterations, tolerance 10⁻⁴; automatic warm start. Rigid solver uses two substeps per 0.005 s exchange. Full solver settings and source hashes are linked for each run.",
        ),
        (
            "Commands",
            "Body-frame vx, vy in m/s; wz in rad/s. Each command is constant from reset. Positive vy is left; positive wz is counterclockwise viewed from above.",
        ),
        (
            "Measurement",
            "Foot reactions in world coordinates, in newtons; +z is upward. Plot total contact reaction = sand contribution + rigid-geometry contribution. Robot velocities are measured in its body frame. Compare each solver over the same elapsed interval, including startup. Forces are responses along independent robot trajectories, so their differences are not isolated force-estimator accuracy.",
        ),
    ],
)


def command(meta):
    c = meta["command_body_frame"]
    return f"vx = {c['vx_m_s']:+.1f} m/s, vy = {c['vy_m_s']:+.1f} m/s, wz = {c['wz_rad_s']:+.1f} rad/s"


def outcome_text(stats):
    text = {
        "duration": "Completed requested duration",
        "reached_sand_bed_edge": "Stopped at bed-boundary threshold",
        "terminated": "Environment termination",
    }[stats["outcome"]]
    if stats.get("terminal_terms"):
        text += ": " + html.escape(", ".join(stats["terminal_terms"]))
    return text


def profiles_summary():
    rows = []
    for name, label in PROFILES:
        if name not in summary:
            continue
        r = summary[name]
        for backend, solver in [("mpm", "Implicit MPM"), ("nerd", "Particle NeRD")]:
            stats = r["profiles"][backend]
            rows.append(
                (
                    label,
                    solver,
                    f"0–{r['common_measurement_end_s']:.2f} s",
                    f"{stats['duration_s']:.2f} s; {outcome_text(stats)}",
                    " / ".join(f"{x:+.3f}" for x in stats["mean_body_velocity"]),
                    " / ".join(f"{x:.3f}" for x in stats["tracking_rmse"]),
                )
            )
    return table(
        "One independent episode per solver and command. For each pair, means and command-tracking RMSE use the same elapsed interval, from the first 0.005 s sample to the shorter rollout’s endpoint, including startup. vx / vy use m/s; wz uses rad/s. This is a behavioral comparison, not force-prediction RMSE.",
        rows,
        [
            "Case",
            "Solver",
            "Measurement interval",
            "Complete rollout",
            "Mean vx / vy / wz",
            "Tracking RMSE vx / vy / wz",
        ],
    )


def profile_blocks(kind):
    blocks = []
    for i, (name, label) in enumerate(PROFILES):
        meta = read(FULL / name / "metadata.json")
        heading = f"{label}: {command(meta)}"
        if name not in summary:
            body = "<p><strong>NeRD only; MPM reference omitted.</strong> The existing six-second Particle NeRD recording is retained. This case is excluded from the paired summary table.</p>"
            if kind == "force":
                body += figure(
                    3 + 2 * i,
                    f"figures/full_{name}_force",
                    f"{heading}. NeRD-only contact forces: (a,b) Fx; (c,d) Fy; (e,f) Fz. Left/right columns: left/right foot. Solid green: total contact force; dashed light green: sand contribution; dotted dark blue: rigid-geometry contribution. No MPM curve is present. Raw world-frame forces at 200 Hz; no smoothing.",
                    "Negative-yaw NeRD-only foot forces",
                )
                body += figure(
                    4 + 2 * i,
                    f"figures/full_{name}_tracking",
                    f"{heading}. NeRD-only tracking: (a) vx; (b) vy; (c) wz. Solid green: measured robot velocity; dashed dark blue: command. Body-frame measurements at 200 Hz.",
                    "Negative-yaw NeRD-only command tracking",
                )
            else:
                body += figure(
                    2 + i,
                    f"figures/full_{name}_displacement",
                    f"{heading}. NeRD-only deformation: (a) RMS 3D displacement from initial particle positions; (b) mean signed z displacement from those positions. Bed-wide statistics at 50 Hz, including unchanged particles. No MPM comparison is present.",
                    "Negative-yaw NeRD-only particle deformation",
                )
            body += video(
                (3 if kind == "force" else 2) + i,
                f"../data/full-20261109/{name}/simulation",
                f"{heading}. Existing full closed-loop Particle NeRD recording used for these plots. 300 frames at 50 fps; simulation states from 0 to 5.98 s. MPM reference omitted.",
            )
            body += f'<p class="sources"><a href="../data/full-20261109/{name}/metadata.json">NeRD settings and source hashes</a> · <a href="../data/full-20261109/{name}/verification.json">Recording checks</a></p>'
            blocks.append(
                f'<details class="profile" id="{name}"><summary>{heading} — NeRD only</summary>{body}</details>'
            )
            continue
        stats = summary[name]
        body = "<p>Independent closed-loop runs from identical initial states. "
        for backend, solver in [("mpm", "Implicit MPM"), ("nerd", "Particle NeRD")]:
            r = stats["profiles"][backend]
            body += f"{solver}: {r['duration_s']:.2f} s; {outcome_text(r)}. "
        body += "No resets are joined into either recording. Dark blue: implicit MPM; dashed green: Particle NeRD.</p>"
        if kind == "force":
            body += figure(
                3 + 2 * i,
                f"figures/full_{name}_force",
                f"{heading}. World-frame contact reaction on each foot. (a,b) Fx; (c,d) Fy; (e,f) Fz. Left/right columns: left/right foot. Solid dark blue: implicit MPM; dashed green: Particle NeRD. Each curve includes sand and rigid-geometry contributions, sampled at 200 Hz without smoothing. Each curve ends at its own recorded endpoint. Robot motions are independently controlled, so curve differences include differences in gait and contact timing.",
                f"{label}: implicit MPM and Particle NeRD closed-loop Fx, Fy, Fz on both feet",
            )
            body += figure(
                4 + 2 * i,
                f"figures/full_{name}_tracking",
                f"{heading}. (a) vx; (b) vy; (c) wz, measured in the robot body frame. Solid dark blue: implicit MPM; dashed green: Particle NeRD; dotted gray: command. Same episodes and simulation clock as the force plot and paired video; traces sample at 200 Hz and video at 50 fps.",
                f"{label}: command tracking in implicit MPM and Particle NeRD",
            )
            rows = []
            for foot, footname in enumerate(("Left foot", "Right foot")):
                for component in range(3):
                    vals = []
                    for backend in ("mpm", "nerd"):
                        r = stats["profiles"][backend]
                        vals.append(
                            f"{r['mean_foot_force_N'][foot][component]:+.2f} ± {r['std_foot_force_N'][foot][component]:.2f}"
                        )
                    rows.append((footname + " F" + "xyz"[component], *vals))
            body += table(
                f"Mean ± temporal standard deviation of foot contact force (N), same elapsed interval 0–{stats['common_measurement_end_s']:.2f} s in both runs. Standard deviation includes gait cycles and startup; it is not a noise estimate or uncertainty across trials.",
                rows,
                ["Force component", "Implicit MPM", "Particle NeRD"],
            )
            body += (
                "<p>Maximum absolute rigid-geometry force component in either foot over each complete rollout: "
                + "; ".join(
                    f"{label}: {stats['profiles'][backend]['max_abs_rigid_force_N']:.2f} N"
                    for backend, label in [("mpm", "implicit MPM"), ("nerd", "Particle NeRD")]
                )
                + ".</p>"
            )
        else:
            body += figure(
                2 + i,
                f"figures/full_{name}_displacement",
                f"{heading}. (a) RMS 3D particle displacement from that run’s initial positions. (b) Mean signed z displacement from those positions. Solid dark blue: implicit MPM; dashed green: Particle NeRD. Each spatial average includes all {meta['particle_count']:,} particles, including unchanged particles, and is sampled at 50 Hz. These bed-wide statistics include settling of the fresh grid as well as robot-induced motion. MPM evolves every particle; NeRD retains its active selection and displacement rule. Initial positions are identical, but robot trajectories evolve independently: these are deformation statistics within each run, not pointwise prediction errors against MPM.",
                f"{label}: particle deformation in independent MPM and NeRD runs",
            )
        body += video(
            (3 if kind == "force" else 2) + i,
            f"../data/full-comparison-20261109/{name}/comparison",
            f"{heading}. Left: full implicit MPM. Right: full Particle NeRD. Both use the same policy; each robot responds to its own simulated forces. These recordings supplied the plotted measurements. Panels share identical simulation timestamps, from 0 to {stats['last_video_state_s']:.2f} s ({stats['video_frames']} frames, 50 fps), stopping at the shorter recording without padding. Complete individual videos are linked below.",
        )
        for backend, folder, solver in [
            ("mpm", "full-mpm-20261109", "Implicit MPM"),
            ("nerd", "full-20261109", "Particle NeRD"),
        ]:
            body += f'<p class="sources">{solver}: <a href="../data/{folder}/{name}/simulation.mp4">Complete video</a> · <a href="../data/{folder}/{name}/metadata.json">Settings and source hashes</a> · <a href="../data/{folder}/{name}/trace.npz">200 Hz force and robot trace</a> · <a href="../data/{folder}/{name}/particle_metrics.npz">Particle statistics and video timestamps</a></p>'
        body += f'<p class="sources"><a href="../data/full-comparison-20261109/{name}/verification.json">Initial-state, timing and artifact checks</a></p>'
        blocks.append(
            f'<details class="profile" id="{name}" {"open" if i == 0 else ""}><summary>{heading}</summary>{body}</details>'
        )
    return "".join(blocks)


matching_replay_video = lambda number: video(
    number,
    "../data/current-20261109/comparison",
    "Prescribed robot motion in both panels. Left: recorded MPM particles. Right: current NeRD particle predictions, including the foot-speed displacement rule. NeRD force estimates do not control the robot. These are the exact states used by the prescribed-motion plots; 250 frames at 50 fps, elapsed 0–4.98 s. The final 5.00 s state remains in the data.",
)
force_caption = "World-frame sand forces on the feet. (a,b) Fx; (c,d) Fy; (e,f) Fz. Left/right columns: left/right foot. Solid dark blue: recorded implicit MPM; dashed green: current NeRD reconstruction. One recorded trajectory; 1,000 samples per foot at 200 Hz, no smoothing."

force_overview = table(
    "Three tests of different parts of the same current method.",
    [
        (
            "1. Force estimator only",
            "MPM robot motion, particle positions and prior wrench for free-velocity reconstruction",
            "Sand forces",
            "Force-estimator agreement with MPM, without neural particle-prediction error.",
        ),
        (
            "2. Predicted particles + force estimator",
            "MPM robot motion and prior wrench; MPM particles only initialize the three-frame history",
            "Recurrent particle motion, then sand forces",
            "Combined particle-prediction and force-estimation error under identical robot motion.",
        ),
        (
            "3. Full simulation",
            "Constant velocity command to policy jqur5jhm",
            "Robot motion, particle motion and reaction forces",
            "Closed-loop forces, behavior and command tracking against a separate native-MPM rollout. No imposed trajectory; no isolated force-accuracy RMSE.",
        ),
    ],
    ["Test", "What is prescribed", "What is computed", "What it measures"],
)

f1 = "<p>Supply recorded robot motion and recorded MPM particle positions at every step. Particle velocities are computed from consecutive recorded positions divided by 0.005 s, matching the current reconstruction’s position-to-velocity convention. Apply the current active-particle selector and force estimator. <strong>No neural particle prediction or displacement stabilization is applied.</strong> The estimated forces do not alter the recorded robot motion. As specified above, the pre-contact proxy velocity is reconstructed using the prior recorded MPM wrench; this is a conditional reconstruction test.</p>"
f1 += "<p>For each timestep and foot, subtract the recorded MPM force vector. Force-vector RMSE = √mean(ΔFx² + ΔFy² + ΔFz²), averaged over both feet and all 1,000 timesteps. This test also retains the current contact selection and finite-difference velocity convention.</p>"
f1 += figure(
    1,
    "figures/estimator_only_force",
    force_caption + " Here the estimator receives prescribed MPM particles.",
    "Force-only reconstruction versus MPM, with identical prescribed robot and particle motion",
)
f1 += table(
    "Force-only errors over the complete plotted replay.",
    [
        ("Force-vector RMSE", f"{force_only['force_vector_rmse_N']:.2f} N"),
        ("Fx / Fy / Fz RMSE", " / ".join(f"{v:.2f}" for v in force_only["metrics"]["component_rmse_N"]) + " N"),
    ],
)
f1 += video(
    1,
    "../data/estimator-only-20261109/prescribed-input",
    "The prescribed MPM robot and particle motion supplied to the force estimator for Figure 1. This is the matching reference panel from the recorded replay, not an independently simulated NeRD rollout. There is no predicted particle trajectory in this test. Elapsed 0–4.98 s, 250 frames at 50 fps.",
)
f1 += '<p class="sources"><a href="../data/estimator-only-20261109/metadata.json">Force-only settings</a> · <a href="../data/estimator-only-20261109/trace.npz">Raw forces</a> · <a href="../data/estimator-only-20261109/verification.json">State alignment checks</a></p>'

f2 = "<p>Keep the same recorded robot motion, but let the current NeRD solver predict particle positions recurrently for five seconds. MPM particle positions initialize only the first three history frames. Subsequent NeRD particle states are carried forward without replacement by MPM. The force estimator reads these predicted particles. <strong>NeRD forces do not move the robot in this test.</strong></p><p>The current foot-speed and particle-height displacement rule is enabled. These results evaluate the integrated predictor, active selection, displacement scaling, boundary handling and force reconstruction together.</p>"
f2 += figure(
    2,
    "figures/current_force_components",
    force_caption + " Here particle motion is predicted recurrently by NeRD.",
    "Combined NeRD particle prediction and force estimation under prescribed robot motion",
)
f2 += f"<p>Force-vector RMSE: <strong>{replay['force_vector_rmse_N']:.2f} N</strong>. The same estimator gives {force_only['force_vector_rmse_N']:.2f} N with prescribed particles. Their difference is not an additive decomposition of error; particle geometry changes contact admission and force response nonlinearly.</p>"
f2 += matching_replay_video(2)
f2 += '<h3>Why left and right support differ in this replay</h3><p>At elapsed 1.50 s, MPM gives Fz = 155.7 N left / 170.7 N right. NeRD-predicted particles give 0 / 165.1 N; feeding MPM particle motion to the same estimator gives 153.9 / 168.2 N. The nearest particle center directly below the left sole is 26.5 mm from the sole in NeRD, versus 1.8 mm in MPM, measured along the foot’s local normal. These are particle-center distances, not surface clearances.</p><p>The displacement rule can stop further motion beneath a slow foot, but it does not refill an existing gap. The early left-support loss is a failure in this prescribed-motion test. It does not identify the network, particle selection and stabilization rule as separate causes. <a href="../data/current-20261109/force-diagnostic.json">Detailed diagnostic measurements</a>.</p>'

f3 = (
    "<p>Run the same policy independently with implicit MPM and Particle NeRD, starting from identical robot and particle states. Each solver’s forces change its own robot motion, which changes subsequent contacts. Five command pairs below compare the complete simulations; the negative-yaw NeRD-only result is retained separately. MPM states are never fed into the NeRD rollout.</p>"
    + full_conditions
    + profiles_summary()
    + "<p>Open a command to inspect its force plot, tracking plot and matching video.</p>"
    + profile_blocks("force")
)

sources = '<p>Raw arrays and particle-state datasets remain in the local archive. The <a href="../../local_artifacts.json">local artifact manifest</a> lists their paths, sizes and checksums; these files are not public downloads.</p><p>All results use the installed checkpoint and solver source recorded in their metadata. No training was performed for this report. Prescribed-motion accuracy is measured on one archived MPM trajectory; full simulation is measured on one episode per solver and command. The different bed dimensions, resolutions and initial states are documented above and are not pooled into a single accuracy number.</p><p><a href="../data/current-20261109/metadata.json">Prescribed-motion settings and source hashes</a> · <a href="../data/current-20261109/reference-body-properties.json">Reference foot mass and inertia</a> · <a href="../data/current-20261109/trace.npz">Prescribed-motion measurements</a> · <a href="../data/current-20261109/states.h5">Matched replay states</a> · <a href="../data/full-comparison-20261109/summary.json">Full-simulation summaries</a> · <a href="../../plots/DECOUPLED_EVALUATION.md">Reproduction instructions</a></p>'
force_sections = (
    section("conditions", "Evaluation design and simulation conditions", force_overview + replay_conditions)
    + section("force-only", "1. Force estimator with prescribed particles", f1)
    + section("predicted-particles", "2. Force estimator with NeRD-predicted particles", f2)
    + section("full-simulation", "3. Full closed-loop simulation", f3)
    + section("sources", "Data and reproduction", sources)
)
(G1 / "force_estimation/index.html").write_text(
    page(
        "G1 foot-force evaluation",
        "Separate force reconstruction from particle prediction, then test the complete coupled simulation. One current method; three evaluation protocols.",
        [
            ("conditions", "Conditions"),
            ("force-only", "Force only"),
            ("predicted-particles", "Predicted particles + forces"),
            ("full-simulation", "Six closed-loop commands"),
            ("sources", "Data"),
        ],
        force_sections,
    )
)

particle_overview = table(
    "Two particle evaluations with different reference requirements.",
    [
        (
            "1. Prescribed robot motion",
            "Same recorded robot motion and initial particle history as MPM",
            "Same-particle-ID errors against MPM; the policy is not part of this test.",
        ),
        (
            "2. Full simulation",
            "Velocity commands to policy jqur5jhm; no reference trajectory",
            "Particle deformation and robot behavior. Independent trajectories do not provide pointwise prediction accuracy.",
        ),
    ],
    ["Test", "Input", "Measurement"],
)
p1 = "<p>At every 0.005 s step, prescribe the recorded robot motion and predict particle positions recurrently with the current NeRD solver. Compare each predicted particle with the same particle ID in the MPM recording. The 3D position RMSE is √mean(‖predicted position − MPM position‖²) over 88,360 particles. Mean signed z error is mean(predicted z − MPM z); positive means higher than the corresponding MPM particle.</p><p>Argon’s foot-speed and particle-height displacement rule is enabled. Active selection, background motion and boundary handling are included. This measures the complete particle-update method, rather than the raw network output. A bed-wide mean can hide local errors beneath the feet.</p>"
p1 += figure(
    1,
    "figures/current_particle_error",
    "(a) RMS 3D position error against MPM. (b) Mean signed z error against MPM. Same 88,360 particle IDs and recorded robot trajectory; 1,000 timesteps, no smoothing. Current foot-speed displacement scaling is enabled.",
    "Same-particle-ID position errors against MPM under prescribed robot motion",
)
p1 += table(
    "Errors at elapsed 5.00 s.",
    [
        ("3D particle-position RMSE", f"{replay['final_particle_position_rmse_m'] * 1000:.2f} mm"),
        ("Mean signed z error", f"{np.load(G1 / 'data/current-20261109/trace.npz')['metrics'][-1, 2] * 1000:+.2f} mm"),
    ],
)
p1 += matching_replay_video(1)
p2 = (
    "<p>Here the same policy controls two independent robots: one coupled to native implicit MPM and the other to Particle NeRD. Both start from identical particle and robot states. The plots compare deformation relative to each run’s own initial particles. Because the robot trajectories can diverge, this is a full-simulation comparison, not isolated particle-prediction accuracy.</p>"
    + full_conditions
    + profiles_summary()
    + "<p>Open a command to inspect particle displacement and its matching simulation video. Each spatial average includes unchanged particles, so localized deformation can be much larger than the reported bed-wide value.</p>"
    + profile_blocks("particles")
)
particle_sections = (
    section("conditions", "Evaluation design and simulation conditions", particle_overview + replay_conditions)
    + section("prescribed-motion", "1. Particle prediction under prescribed robot motion", p1)
    + section("full-simulation", "2. Full closed-loop particle simulation", p2)
    + section("sources", "Data and reproduction", sources)
)
(G1 / "particle_prediction/index.html").write_text(
    page(
        "G1 sand-particle evaluation",
        "Measure particle-prediction error under prescribed robot motion, then inspect deformation in six freely controlled simulations.",
        [
            ("conditions", "Conditions"),
            ("prescribed-motion", "Prediction accuracy"),
            ("full-simulation", "Six closed-loop commands"),
            ("sources", "Data"),
        ],
        particle_sections,
    )
)

for target, prefix in [(ROOT / "index.html", "g1_humanoid/"), (G1 / "index.html", "")]:
    assets = "assets/" if target.parent == ROOT else "../assets/"
    lead = "Separate force-only accuracy, combined particle-and-force accuracy, and full closed-loop behavior. Six Particle NeRD command profiles use policy jqur5jhm. Five have matched independent implicit-MPM references, overlaid plots and synchronized videos; negative yaw remains NeRD only."
    text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 benchmark results · Particle NeRD</title><link rel="stylesheet" href="{assets}slides.css?v=20261109-decoupled"></head><body class="benchmark-report"><header><a class="brand" href="index.html">Particle NeRD / Benchmarks</a><span class="archive">Current solver evaluation · 9 November 2026</span></header><main><div class="landing"><div class="eyebrow">G1 humanoid · current solver evaluation</div><h1>Particle motion and foot forces</h1><p>{lead}</p><div class="grid"><a class="card" href="{prefix}force_estimation/index.html"><h2>Foot-force evaluation</h2><img class="preview" src="{prefix}force_estimation/figures/estimator_only_force.svg" alt="Force-only comparison with prescribed MPM particle and robot motion"><p>Three protocols: prescribed particles, predicted particles, and full simulation. Fx, Fy and Fz on each foot.</p></a><a class="card" href="{prefix}particle_prediction/index.html"><h2>Particle evaluation</h2><img class="preview" src="{prefix}particle_prediction/figures/current_particle_error.svg" alt="Particle-position error under prescribed robot motion"><p>Prediction error against MPM under prescribed motion, plus five MPM–NeRD deformation comparisons and the retained NeRD-only negative-yaw run.</p></a></div><p class="caption">200 Hz physics · 50 Hz policy control · 50 fps videos. Forward, backward and lateral commands at 0.5 m/s; forward motion with ±1 rad/s yaw.</p></div></main><footer class="site-footer"><span class="footer-brand">Particle NeRD</span><span>G1 humanoid benchmarks · 11/09/2026</span></footer></body></html>'''
    if target.parent == ROOT:
        text = text.replace(
            '<p class="caption">',
            '<p><a class="card" href="benchmark_environment/"><strong>Native MPM benchmark environments</strong> · 14 object and robot cases with matched videos, force plots, and datasets.</a></p><p class="caption">',
            1,
        )
    target.write_text(text)
css = ROOT / "assets/slides.css"
if ".profile > summary" not in css.read_text():
    css.write_text(
        css.read_text()
        + """\n/* Each command opens its own matched plots and recording. */
.profile { border-top: 1px solid #ccc; padding: 22px 0; }
.profile > summary { cursor: pointer; font-size: 23px; font-weight: 700; line-height: 1.4; padding: 8px 0; }
.profile > summary::marker { color: #76b900; }
.profile[open] > summary { margin-bottom: 24px; }
.profile .sources { margin-top: 25px; }
@media (max-width: 700px) { .profile > summary { font-size: 18px; } }
"""
    )
print("Built separated reports locally; publication requires a GitHub push and live-site verification")
