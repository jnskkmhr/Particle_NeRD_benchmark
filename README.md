# Particle NeRD benchmark

The website presents a **9 November 2026 evaluation of the installed Particle NeRD solver**, compared with a recorded implicit-MPM reference. Both pages use one five-second prescribed-motion replay and the exact matching video.

- [Foot forces](g1_humanoid/force_estimation/index.html): force-vector RMSE **83.60 N**.
- [Particle positions](g1_humanoid/particle_prediction/index.html): final 3D position RMSE **19.29 mm**.
- [Current settings and hashes](g1_humanoid/data/current-20261109/metadata.json).
- [Reproduction instructions](plots/CURRENT_REPLAY.md).

Argon’s foot-speed displacement scaling is enabled. The replay uses 88,360 particles from the 0.04 m reference recording, with 5,312 active particles and 512 background samples. Its three-frame initial history comes from recorded MPM states. It is not an initially unsettled-bed test or a closed-loop policy test.

Serve with `python3 -m http.server 8765 --bind 127.0.0.1`, then open http://localhost:8765/. Forward port 8765 when using remote SSH.

Historical October 28 JSON files, figures and videos remain on disk for provenance; they are not used by the current report. Their original hashes are in `import_manifest.json`. Independent `benchmark_environment/README.md` and `metrics/README.md` are preserved.


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
