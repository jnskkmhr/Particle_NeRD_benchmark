# Particle NeRD benchmark

The local G1 reports separate three evaluations of the installed Particle NeRD method:

1. Prescribed MPM robot and particle motion → force reconstruction: force-vector RMSE **32.66 N**.
2. Prescribed MPM robot motion → recurrent NeRD particles and force reconstruction: force-vector RMSE **83.60 N**, final particle-position RMSE **19.29 mm**.
3. Full closed-loop policy + rigid dynamics + NeRD: six command profiles with policy **jqur5jhm**, matched force/tracking/deformation plots and 50 fps videos. No MPM or robot reference is supplied to the full simulation.

Both replay tests use the previous recorded MPM wrench to reconstruct the proxy free velocity. They share one five-second recorded trajectory; they are conditional replay tests. Full runs start from a fresh grid with zero native MPM steps. Their different bed size, resolution and initialization are documented separately. Full-run particle displacement is not an MPM prediction-error metric.

- [Force report](g1_humanoid/force_estimation/index.html)
- [Particle report](g1_humanoid/particle_prediction/index.html)
- [Conditions and reproduction](plots/DECOUPLED_EVALUATION.md)
- [Full-run summaries](g1_humanoid/data/full-20261109/summary.json)

Serve with `python3 -m http.server 8765 --bind 127.0.0.1`, then open http://localhost:8765/. Forward port 8765 when using remote SSH. Large raw arrays are local artifacts and remain excluded from published Git history.

Historical October 28 data remain on disk for provenance and are not displayed as current results. Independently maintained benchmark-environment and metrics files are preserved.

## Native MPM benchmark environments

[Open the 14-case gallery](https://jnskkmhr.github.io/Particle_NeRD_benchmark/benchmark_environment/).
The website source is maintained in `benchmark_environment/`, including its catalog
publisher. Simulation scripts remain in IsaacLab-particle-nerd's
`examples/particle_nerd_benchmark`; executed source snapshots accompany each run.

Videos and plots are paired with the same recorded rollout. The large HDF5 datasets
and G1 policy checkpoint are retained locally and are excluded from this publication,
with paths, byte sizes, and SHA-256 hashes in `local_artifacts.json`.
The `.h5`, `.npz`, `.npy`, and `.pt` files stay outside the published Git history. The manifest records local artifact paths and checksums; no public dataset downloads are provided.

GitHub Pages serves the repository root from `main`, with `.nojekyll` enabled.
[Public website](https://jnskkmhr.github.io/Particle_NeRD_benchmark/).


## Publishing benchmark website updates

Completed and verified website updates should be published to the existing GitHub Pages site as part of the task, following the user’s standing request. Commit on a feature branch, publish with a normal fast-forward update to `main`, and verify the deployed pages and matching media before reporting completion. Preserve the local-only dataset exclusions and the independently maintained benchmark gallery.
