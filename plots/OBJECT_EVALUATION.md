# Shared object-model evaluation report

`build_object_report.py` builds the object-model report under
`benchmark_environment/object_prediction/`. It reuses `assets/slides.css`.
The builder requires the full 600-trajectory training audit, a completed training
log or an explicitly frozen in-progress checkpoint, all 60 held-out test IDs,
and four matching reference-case evaluations:
Box horizontal-flat plus 20-second Box, Sphere, and Cylinder support tests.

From the IsaacLab-particle-nerd checkout, create each reference case's plots and
50 fps video from its saved evaluation directory:

```bash
PYTHONPATH=examples uv run --no-sync python -m particle_nerd_benchmark.learning.plot CASE_DIRECTORY
PYTHONPATH=examples uv run --no-sync python -m particle_nerd_benchmark.learning.video CASE_DIRECTORY
```

Then stage the report using private downloaded training outputs and the same
trained checkpoint for every evaluation:

```bash
uv run --no-project --with matplotlib --with numpy python /PATH/Particle_NeRD_benchmark/plots/build_object_report.py \
  --training PRIVATE_TRAINING_OUTPUT \
  --test-report PRIVATE_TRAINING_OUTPUT/test_evaluation/report.json \
  --references PRIVATE_REFERENCE_EVALUATIONS \
  --holds PRIVATE_FIXED_HOLD_EVALUATIONS \
  --test-videos PRIVATE_TEST_MOTION_VIDEOS \
  --dynamics PRIVATE_DYNAMIC_EVALUATIONS \
  --output NEW_STAGING_DIRECTORY
```

`PRIVATE_TRAINING_OUTPUT` contains `data_audit.json`, `model/config.json`,
`model/metrics.jsonl`, and `model/best.pt`. Reference directories contain
`evaluation.json`, `prediction.h5`, metric CSVs, generated figures, the paired
video, and both plot/video provenance files. These inputs remain private.
The reference root also needs `report.json`, generated with
`python -m particle_nerd_benchmark.learning.summarize PRIVATE_REFERENCE_EVALUATIONS --expected 4`.
Its one-second force-window means and temporal standard deviations populate
each case's force-summary plot. The six figures include reference population counts,
so missing regional errors are distinguishable from zero error.

To evaluate before the planned training run finishes, copy the selected
`best.pt`, configuration, and metrics log into a separate snapshot directory.
Confirm that the checkpoint step is the minimum-loss validation checkpoint in
that frozen log; retry the copy if training updated the selection during copying.
Add `training_snapshot.json` with schema
`particle_nerd_object_evaluation_snapshot_v1`, `checkpoint_sha256`,
`checkpoint_step`, `training_step_at_snapshot`, `selection`, `source_workflow`,
`snapshot_utc`, and `training_complete`. The latter is false for an in-progress
run. All evaluations must use these frozen weights. The report shows the actual
selected checkpoint step and its maximum training-window length, separately
from the planned training duration.

The builder checks checkpoint, prediction, reference, figure, and video hashes;
video state indices; and common evaluation/network source hashes. It rejects
partial test sets, unfinished training without a verified snapshot, stale
figures, and mismatched videos.
It never copies HDF5, NPZ, CSV state measurements, or model weights into the
public report. Public files contain plots, videos, metric summaries, settings,
and provenance. `evaluation_settings.json` omits the local reference path;
`plot_provenance.json` retains the hash of the original private `evaluation.json`.

Inspect staged pages at desktop and mobile widths, check video playback and
figure captions, then copy the verified staging contents to
`benchmark_environment/object_prediction/` and add a link from the environment
gallery. Do not publish local engineering checks as final model results.


## Fixed-object consistency extension

The report now includes measured core-metric plots and three 23-second fixed-object recordings.
The object stops at 2 s. Temporal metrics compare each method to its own state at 3 s through 23 s,
without resetting particles. The benchmark source provides:

```bash
PYTHONPATH=examples uv run --no-sync python -m particle_nerd_benchmark.learning.collect \
  --shape box --mode prescribed --reference fixed-hold --output /private/reference/box
PYTHONPATH=examples uv run --no-sync --with scikit-image==0.27.0rc1 python \
  -m particle_nerd_benchmark.learning.evaluate --data /private/reference \
  --checkpoint /private/model/best.pt --output /private/evaluation
PYTHONPATH=examples uv run --no-sync --with scikit-image==0.27.0rc1 python \
  -m particle_nerd_benchmark.learning.consistency /private/evaluation/box-fixed-hold
```

Repeat collection for sphere and cylinder. Run the existing plot, video, and summarize modules on the
completed evaluations. Build the public report with the new required `--holds /private/evaluation`
argument in addition to `--training`, `--test-report`, `--references`, and `--output`.
`object_metric_results.py` embeds plots of actual measured JSON; it does not contain substituted results.
Use the frozen evaluation source and checkpoint hashes recorded in the report; a newer model or evaluator
is a different experiment. Do not expose private trajectories or checkpoints when reproducing the site.

The MPM references are not perfect equilibrium targets: settling and floor leakage are retained and
reported. All particles are included in the temporal metrics. The enlarged offline grid does not repair
the original paired-field failures. See `metrics/README.md` and `consistency_results.json` for definitions.

The public `plot_object_consistency.py CASE_PAGE_DIRECTORY --output NEW_PLOT_DIRECTORY` regenerates Figure 7 from published scalar time series without private particle trajectories.


## Graphical presentation of measured metrics

The overview contains eight comparison figures instead of numerical result tables.
Each detailed case also replaces force-error and selected-window tables with plots.
Only simulation-condition tables remain. Exact numerical values remain in the same public JSON files.

```bash
uv run --no-project --with matplotlib --with numpy python plots/plot_object_metric_comparisons.py \
  benchmark_environment/object_prediction
```

This command reads only the public test summaries, consistency summaries, evaluation settings, and
force-window data. It creates SVG/PDF/PNG figures and `comparison_plot_provenance.json` with their
source/input/output hashes. It does not train a model, run physics, smooth force traces, drop
trajectory outliers, or substitute missing particle-region errors with zeros.

The distribution plots show every available trajectory summary, median/IQR boxes, 1.5-IQR whiskers,
and mean diamonds. Counts are explicit per shape. Fixed-object bars represent one paired recording
per shape; no repeated-trial uncertainty is implied. Reference MPM leakage and the effect of escaped
particles on RMS remain disclosed.

The report reads its six numbered metric section titles directly from `metrics/README.md` and links each section to its definition. Figure 2 shows mean and 95th-percentile symmetric surface distances on the primary 0.02 m reconstruction grid. The auxiliary 0.03 m reconstruction-resolution check remains in `test_report.json`; these offline measurement grids do not change the simulation resolution.


## Videos and panel labels

The motion gallery shows the first held-out episode in each shape/motion group (24 clips). Its manifest verifies the exact reference hash against `test_report.json`, the model checkpoint, prediction hash, and every video artifact. It copies only videos, posters, and provenance. The four force-reference videos are embedded in metric 5. No simulation or inference is rerun to make these videos.

Render each downloaded saved evaluation with `uv run --no-sync python plots/render_object_motion_video.py PRIVATE_CASE --reference PRIVATE_REFERENCE.h5`; the reference's `manifest.json` must be beside that HDF5 file. Keep its original `evaluation.json`, `prediction.h5`, and reference file private. The gallery uses `motion_video_provenance.json` to record its deterministic selection rule and exact clip identities.

Figures use descriptive labels without lettered panels. The particle-position, sand-surface, and velocity-field comparison columns are Time-mean (0–6 s) and Terminal state (6 s). The overview regional-population/unchanged-position-baseline figure is removed. Numerical inputs and the 60 evaluated trajectories are unchanged. Detailed plots can be regenerated with `plots/plot_recorded_object_case.py PRIVATE_CASE`, while temporal plots use `plot_object_consistency.py`.


## Kinematic replay and fully dynamic object comparisons

Table 2 separates reference generation from evaluation. All original metrics
1–6 impose the recorded object motion during NeRD evaluation, including records
whose MPM source was dynamic. Force reconstruction for those dynamic-source
records still uses finite physical body mass/inertia.

The additional fully dynamic section uses one 20-second drop-and-support run per
shape. `PRIVATE_DYNAMIC_EVALUATIONS` contains `box-dynamic`, `sphere-dynamic`, and
`cylinder-dynamic`, each produced with the same step-9,000 checkpoint by the
IsaacLab example `particle_nerd_benchmark.learning.closed_loop`. For example:

```bash
PYTHONPATH=examples uv run --no-sync python -m particle_nerd_benchmark.learning.closed_loop \
  --reference PRIVATE_REFERENCES/box-static-support/box-static-support.h5 \
  --checkpoint PRIVATE_TRAINING_OUTPUT/model/best.pt \
  --output PRIVATE_DYNAMIC_EVALUATIONS/box-dynamic
uv run --no-sync python /PATH/Particle_NeRD_benchmark/plots/render_object_motion_video.py \
  PRIVATE_DYNAMIC_EVALUATIONS/box-dynamic \
  --reference PRIVATE_REFERENCES/box-static-support/box-static-support.h5
```

The runner builds only rigid-body dynamics and the neural/contact estimators.
It uses no native MPM steps, no future reference body/particle inputs, and no
additional stabilization or force filter. It matches the reference's lagged
force exchange and two rigid substeps. The object remains dynamic even before
release: an external upward force balances gravity until 1 s.

The report builder verifies complete trajectories, shared initial states,
checkpoint identities, exact lagged force application, and video/state hashes.
It exports only media, plotted scalar curves, and provenance; full HDF5 states
and model weights stay private. To regenerate a published dynamic plot:

```bash
uv run --no-project --with matplotlib --with numpy python plots/plot_object_closed_loop.py \
  benchmark_environment/object_prediction/dynamic/box --figure-number 9
```

Figures 9–11 correspond to box, sphere, cylinder. These individual coupled runs
are kept separate from the 60 recorded-motion test trajectories. All three
completed 4,000 steps; their NeRD particles continued sinking, and the rigid tray
eventually supported the objects while material force fell to zero. The full
20-second videos and curves retain this failure to maintain sand support.
