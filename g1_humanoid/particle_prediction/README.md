# G1 particle evaluation

[Open the report](index.html). Two protocols distinguish prediction accuracy from freely evolving deformation.

1. **Prescribed robot motion:** compare identical particle IDs against the recorded MPM trajectory. Final 3D position RMSE is 19.29 mm over 88,360 particles. The current active selection and foot-speed displacement rule are included; only the initial three history frames come from MPM.
2. **Full simulation:** six constant body-frame velocity commands under policy jqur5jhm. The robot and particles evolve through coupled rigid–NeRD dynamics without a reference trajectory. Plots show particle displacement relative to initialization, not pointwise accuracy against MPM. Each video and plot comes from the same run.

The prescribed and full protocols use different bed dimensions, resolution and initial states; their conditions are stated separately. See [conditions and reproduction](../../plots/DECOUPLED_EVALUATION.md). Historical validation JSON files are not the active report’s data.
