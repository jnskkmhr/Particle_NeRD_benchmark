# Particle NeRD benchmark

Archived benchmark results for Particle NeRD.

```text
.
├── g1_humanoid/
│   ├── force_estimation/
│   ├── particle_prediction/
│   └── provenance.json
├── benchmark_environment/
└── metrics/
```

- [G1 force estimation](g1_humanoid/force_estimation/README.md): estimated sand-on-foot force compared with MPM reference force.
- [G1 particle prediction](g1_humanoid/particle_prediction/README.md): particle position and contact-region errors against MPM reference particles.
- `benchmark_environment/` and `metrics/` are reserved for later implementation; only `.gitkeep` files are present.

## Result provenance

The imported results are from **2026-10-28**, covering 50 validation trajectory windows of 100 recurrent prediction steps (0.5 seconds at 200 Hz). These are archived measurements, not a fresh evaluation of the current implementation or currently installed checkpoint.

The initial and fine-tuned checkpoint hashes and validation IDs are preserved in the particle-prediction JSON files. The fine-tuned checkpoint was a local pilot trained on one trajectory for 100 optimizer updates; it is not a full-data training result. See [original provenance](g1_humanoid/provenance.json).

[import_manifest.json](import_manifest.json) records the original local source directory, filenames, byte counts and SHA-256 hashes. Imported JSON files are byte-for-byte copies. Original absolute paths in provenance are historical references, not repository dependencies.

This repository contains result JSON files and summaries. Checkpoint weights, raw trajectory datasets and benchmark execution code are not included in this initial import. No new benchmark or simulation was run.
