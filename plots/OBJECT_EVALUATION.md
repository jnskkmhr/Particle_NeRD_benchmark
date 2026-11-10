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
uv run --no-project python /PATH/Particle_NeRD_benchmark/plots/build_object_report.py \
  --training PRIVATE_TRAINING_OUTPUT \
  --test-report PRIVATE_TRAINING_OUTPUT/test_evaluation/report.json \
  --references PRIVATE_REFERENCE_EVALUATIONS \
  --output NEW_STAGING_DIRECTORY
```

`PRIVATE_TRAINING_OUTPUT` contains `data_audit.json`, `model/config.json`,
`model/metrics.jsonl`, and `model/best.pt`. Reference directories contain
`evaluation.json`, `prediction.h5`, metric CSVs, generated figures, the paired
video, and both plot/video provenance files. These inputs remain private.
The reference root also needs `report.json`, generated with
`python -m particle_nerd_benchmark.learning.summarize PRIVATE_REFERENCE_EVALUATIONS --expected 4`.
Its one-second force-window means and temporal standard deviations populate
each case's force table. The six figures include reference population counts,
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
