# G1 particle-position prediction

[Open the current report](index.html). It evaluates the currently installed Particle NeRD solver on the same five-second prescribed-motion replay as the foot-force report. Argon’s foot-speed displacement scaling is enabled. The matching video is rendered from the saved states used to calculate the plotted errors.

Final 3D particle-position RMSE is **19.29 mm** across 88,360 particles. The solver uses 5,312 active particles and 512 background samples on the recorded 0.04 m bed. It starts with three recorded MPM history frames; it is not an initially unsettled-bed test.

See [run metadata](../data/current-20261109/metadata.json), [alignment verification](../data/current-20261109/verification.json), and [reproduction instructions](../../plots/CURRENT_REPLAY.md).

The `validation-*.json` files are separate historical October 28 archives, not the data displayed in the current report.
