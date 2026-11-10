# G1 evaluation protocols — 9 November 2026

This report evaluates one installed Particle NeRD checkpoint and implementation. It contains no new training and no historical model comparison.

## Data and interpretation

1. **Force estimator with prescribed particles.** `g1_humanoid/data/estimator-only-20261109/` supplies recorded MPM robot poses/velocities and particle positions every step. Particle velocities are consecutive-position differences divided by 0.005 s. Current active selection and force reconstruction are retained. The network is not evaluated, and its displacement rule is not applied. The MP4 shows the exact prescribed input motion. Its reference particle states are shared with `current-20261109/states.h5`, rather than duplicated.
2. **Predicted particles with prescribed robot motion.** `g1_humanoid/data/current-20261109/` runs the current integrated solver recurrently, including active selection and the foot-speed/particle-height displacement rule. Only the initial three particle-history frames come from MPM. Robot motion remains prescribed, so reconstructed forces do not change that motion. Particle errors match particle IDs with the reference.
3. **Full simulation.** `g1_humanoid/data/full-20261109/` contains six fresh closed-loop NeRD G1 episodes using the MPM-trained policy `jqur5jhm`, checkpoint 19,999. Policy actions, rigid dynamics, NeRD particle dynamics and reconstructed contact reactions all interact. The newly estimated sand reaction feeds the next rigid step through lagged coupling. No reference robot or particle trajectory is loaded. Native MPM construction is rejected by a guard in the recorder. Five matched native implicit-MPM episodes (forward, backward, left, right, forward + positive yaw) are in `g1_humanoid/data/full-mpm-20261109/`. The negative-yaw MPM reference was canceled at the user’s request; its existing NeRD-only figures/video remain available and are excluded from paired summaries. Initial particle coordinates and every rigid-body pose are bit-identical to the corresponding NeRD run. Material parameters, policy checkpoint, commands, seed, timestep, rigid solver, collision pipeline and foot proxy mapping are checked to match. MPM evolves every particle; NeRD uses its existing active selection and displacement stabilization. Neither solver is driven by the other solver’s states. Particle displacement from initialization is a deformation measurement, not an accuracy metric against MPM.

Both prescribed-motion protocols reconstruct the proxy free velocity by subtracting the previous recorded MPM wrench impulse from the advanced recorded body velocity, using the reference foot mass and inertia. They therefore condition on the prior MPM wrench; they are not predictions from robot kinematics and particle positions alone. The current target wrench is used only for comparison. This reference dependency does not exist in the full-simulation runs.

## Simulation conditions

The prescribed-motion reference is episode 0002, recording time 0.300–5.300 s, with a 3×3×0.3 m bed, 88,360 particles, 0.04 m voxel size, 5,312 selected particles and 512 background samples. Its three initial particle-position frames end at frame 60. The full runs use a fresh 6×6×0.3 m bed, 2,671,875 particles, current task voxel size 0.02 m, 7,104 selected particles and 512 background samples. The task's bed is widened from 6×2 m to accommodate lateral commands. These different conditions are not pooled into a single accuracy statistic.

All runs use 200 Hz physics. Full simulation uses 50 Hz policy control. Videos are 50 fps and show states at 0,.02,.04,… seconds. Force traces start at .005s. No samples are smoothed or removed. Full runs stop at six seconds, environment termination, or a base-position boundary of |x| or|y|>2.6m. A boundary stop is reported separately from a fall. No automatic resets are concatenated into a video.

Full-run force plots show total contact reaction = sand contribution + rigid-geometry contribution, not the sum of gravity and joint/actuator forces. All foot forces are world-frame forces exerted on the foot; +z is up. Velocity commands and tracking are body-frame vx,vy,wz.

The six commands are(+0.5,0,0),(-0.5,0,0),(0,+0.5,0),(0,-0.5,0),(+0.5,0,+1),(+0.5,0,-1), in m/s,m/s,rad/s. Each uses seed 42, the same initial particles, the same policy, and the same checkpoint/source hashes. One episode per solver and command is not a repeated-seed evaluation. Summary statistics compare the same elapsed interval within each pair, including startup. The force curves retain both complete rollouts, which can have different end times. Force differences include different robot motions and gait phases; they are not isolated force-estimator errors.

`g1_humanoid/data/full-comparison-20261109/` contains paired videos (MPM left, NeRD right), verification reports and summary statistics. Videos align frames by simulation timestamps and stop at the shorter recording; they do not pad either side with frozen frames. Each original complete video remains available. Mean ± standard deviation of force describes temporal variation including gait; it is not uncertainty across seeds.

## Reproduction

Run from `/home/jkamohara/isaac/IsaacLab-particle-nerd`, using its uv environment. Scripts live in `/home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/`.

```bash
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/record_estimator_only.py --steps 1000 --output /tmp/force-only-reproduction

uv run --no-sync --extra ovrtx python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/record_full_profiles.py \
  --backend nerd \
  --checkpoint /tmp/nerd-foot-force-runs/checkpoints/jqur5jhm/model_19999.pt \
  --speed 0.5 --vy 0.0 --wz 0.0 --duration 6 --seed 42 \
  --output /tmp/full-forward-reproduction
```

Use `--backend mpm` for the matched native-MPM baseline, keeping all other arguments the same. Change `--speed`, `--vy`, and `--wz` for the other commands. `run_profiles.py --backend mpm --checkpoint PATH --output NEW_DIRECTORY` runs all six serially. Copy completed recordings into the documented data directories before regenerating comparisons. Keep `training_config.json` alongside the policy checkpoint: the recorder restores actor/history settings and checks the action contract. The current NeRD checkpoint is resolved by the task configuration. Exact policy/checkpoint/source hashes, material values and force settings are in each run's metadata. The recorded MPM dataset and reference proxy mass/inertia file paths are embedded in the prescribed-motion recorder and their provenance is archived with the report.

The RTX viewer draws the authoritative live simulation arrays. It does not advance the physics. Foot reactions and robot velocities are sampled in the physics-step callback; every fourth step records the matching particle metrics and video frame. Full particle/body snapshots are stored at 1 Hz for independent metric and pose checks. Prescribed-motion videos use saved50 Hz states.

Verify matched states, recompute displacement statistics from snapshots, check timestamps and assemble paired videos, then generate figures from the archived arrays:

```bash
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/compare_full.py
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/estimator_only_force.py
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/current_force.py
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/current_particles.py
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/full_force.py
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/full_tracking.py
uv run --no-sync python /home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/plots/full_particles.py
```

Each full plot script also accepts `--profile forward`, `backward`, `left`, `right`, `forward_yaw_positive`, or `forward_yaw_negative`. Outputs are SVG, vector PDF and 300 dpi PNG. Each paired video's `verification.json` records initial-state equality, timing, same-run state checks and artifact hashes. `full-comparison-20261109/summary.json` records the common measurement interval for each pair. Run `build_decoupled_reports.py` to regenerate HTML, then publish and verify the live site as required by the repository README. See `CURRENT_REPLAY.md` for the prescribed-motion replay renderer and source-data provenance.
