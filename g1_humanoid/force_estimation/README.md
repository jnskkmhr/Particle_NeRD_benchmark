# G1 foot-force evaluation

[Open the report](index.html). Three protocols separate current force reconstruction from particle-prediction error and full closed-loop behavior.

- **Prescribed MPM particles and robot motion:** force-vector RMSE 32.66 N; no neural particle prediction.
- **Prescribed robot motion with recurrent NeRD particles:** force-vector RMSE 83.60 N; current displacement stabilization enabled.
- **Full simulation:** policy jqur5jhm with six constant body-frame velocity commands; measured Fx, Fy, Fz and command tracking, with each plot paired to its own 50 fps video. No reference trajectory enters the simulation.

The two replay protocols reconstruct proxy free velocities using prior MPM wrench information. Their forces do not move the recorded robot. Full simulation uses lagged force feedback to the next rigid step. Plots show world-frame contact reactions on each foot, including rigid-floor contributions explicitly in the full runs.

See [conditions and reproduction](../../plots/DECOUPLED_EVALUATION.md). One episode per command does not establish robustness across seeds or long-duration stability. Historical validation JSON files remain separate from the active report.
