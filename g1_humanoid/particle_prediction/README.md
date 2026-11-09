# G1 particle prediction

Archived October 28, 2026 results against recorded MPM particle positions, using recorded rigid-body motion. Both checkpoints use the same 50 validation trajectory IDs and sampling seed 42027. Each window uses 100 recurrent prediction steps at 0.005 seconds per step (0.5 seconds). The reported contact band is 0.02 m.

| Metric | Initial checkpoint (mm) | Local fine-tuned pilot (mm) |
|---|---:|---:|
| All-particle position RMSE | 3.533 | 3.907 |
| Moving-particle position RMSE | 3.772 | 4.171 |
| Contact-region position RMSE | 21.620 | 21.730 |

The pilot improves force estimation without improving these aggregate particle-position errors. Do not infer force accuracy from position RMSE alone.

## Included artifacts

- [validation-initial-direct.json](validation-initial-direct.json): initial checkpoint SHA-256, validation IDs, evaluation settings and aggregate prediction errors.
- [validation-local-pilot-direct.json](validation-local-pilot-direct.json): corresponding local fine-tuned checkpoint results.
- [validation-geometry.json](validation-geometry.json): world-axis contact-region errors, under-sole normal errors and geometric diagnostics, including per-trajectory records. Fields ending in `_m` use metres; fractions are dimensionless.

The original `selection_score_rmse_m` and other diagnostic fields are preserved verbatim. Their evaluation implementation is not imported or redefined here; these files are archived results, not a new metric specification.

See [force estimation](../force_estimation/README.md) for sand-on-foot force accuracy and [provenance](../provenance.json) for pilot training details. This import does not certify the current installed model or closed-loop dynamics.
