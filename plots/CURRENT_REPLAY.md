# Current-solver replay, 9 November 2026

## Reproduce the figures

From this benchmark repository:

```bash
uv run --no-project --with numpy --with matplotlib python plots/current_force.py
uv run --no-project --with numpy --with matplotlib python plots/current_particles.py
```

Both scripts read `g1_humanoid/data/current-20261109/trace.npz` and validate the summary against `metadata.json`. They produce SVG, vector PDF, and 300 dpi PNG. No smoothing, sample removal, or historical checkpoint comparisons are applied.

## Replay and rendering

`record_current_replay.py` records current integrated-solver predictions. `render_current_replay.py` renders the saved states without advancing physics. Run these in the IsaacLab-particle-nerd uv environment; they currently require the absolute reference-data paths recorded in the scripts and metadata:

```bash
uv run --no-sync python /path/to/plots/record_current_replay.py --output /tmp/replay --steps 1000
uv run --no-sync --extra ovrtx python /path/to/plots/render_current_replay.py --run /tmp/replay --output /tmp/replay/comparison.mp4
```

For exact reproduction, use the recorded checkpoint and solver source hashes. The rendering scene is loaded from the preserved reference-era source tree to reproduce the recorded geometry. It is used only to draw recorded body states and particles; no MPM steps are run during rendering.

## Data and alignment

`trace.npz`: `time` is elapsed replay time from 0.005 to 5.000 s; `forces` and `reference_forces` have shape [1000, 2, 6], ordered left/right foot, with Fx/Fy/Fz followed by torques. Forces are sand-on-foot forces in the world frame.

`metrics[:,0]` is 3D position RMSE over 88,360 particles. `metrics[:,1]` is signed mean z error for reference particles horizontally within 0.15 m of either recorded foot body origin at the beginning of the step. `metrics[:,2]` is signed mean z error across the whole particle bed. The website plots columns 0 and 2.

`states.h5` contains 251 snapshots at 0.02 s intervals, including the initial state and final 5.00 s state: `particle_position`, `reference_position`, `body_pose`, and `time_s`. The two video panels use identical `body_pose` values. Video frames use snapshots 0–249 at 50 fps, giving five seconds without duplicating the endpoint. Source recording time is elapsed replay time + 0.300 s.

The reference recording is episode 0002, using a 3 × 3 × 0.3 m sand bed, 0.04 m voxel size and 88,360 particles. The current checkpoint and integrated solver are evaluated on that bed. This is not the live fixed-bed task’s default 0.02 m resolution. The NeRD force estimate never drives the prescribed robot motion.

Argon’s displacement scaling is enabled in Top-K inference. Effective material-response factors are normal 100, tangential 40, horizontal 8 and upward 2. Compilation and CUDA-graph capture are disabled for this offline diagnostic, which is not a speed benchmark.

See `verification.json` beside the data for snapshot/trace alignment, video metadata, and artifact hashes.

`reference-body-properties.json` preserves the reference foot proxy masses, inertia tensors, centers of mass, collision margins, and the original metadata-file hash. These recorded-reference properties are used for the free-velocity calculation and contact model; they are separate from the current material-response factors. The replay script also requires that archived metadata source.
