# G1 force estimation

Archived October 28, 2026 validation results: 50 windows, each 100 steps / 0.5 seconds. The analytical estimator is identified as `rft_body_velocity` in the original provenance.

Forces are sand-on-foot forces compared with MPM reference forces. Force-vector RMSE is aggregated across individual feet and timesteps. Upward force is summed over both feet and then averaged; it is not the same quantity as per-foot force error.

| Particle motion supplied to estimator | Force-vector RMSE per foot (N) | Mean total upward force (N) | MPM mean total upward force (N) |
|---|---:|---:|---:|
| Reference MPM particle motion | 40.92 | 319.99 | 330.19 |
| Initial Transolver | 81.45 | 255.07 | 330.19 |
| Contact-loss fine-tuned Transolver (local pilot) | 46.31 | 329.50 | 330.19 |

[Raw measurements and per-trajectory records](validation-force.json)

The reference-particle row isolates force reconstruction error. The learned-particle rows also include particle prediction error under recorded rigid-body motion. These measurements do not establish closed-loop locomotion accuracy. Torque accuracy is not reported in this artifact.

The pilot improves force error but is not the current installed checkpoint. Checkpoint hashes and the shared validation IDs are in [particle prediction](../particle_prediction/README.md); training details are in [provenance](../provenance.json).
