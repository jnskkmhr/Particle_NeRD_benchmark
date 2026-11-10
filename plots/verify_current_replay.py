"""Verify saved state, trace, reference and video alignment for the current replay."""

import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "g1_humanoid/data/current-20261109"
metadata = json.loads((RUN / "metadata.json").read_text())
trace = np.load(RUN / "trace.npz", allow_pickle=False)
maximum_rmse_difference = 0.0
maximum_mean_z_difference = 0.0
with h5py.File(RUN / "states.h5") as states, h5py.File(metadata["source_recording"]) as reference:
    np.testing.assert_allclose(states["time_s"][:], np.arange(251) / 50, atol=1e-10)
    np.testing.assert_array_equal(states["particle_position"][0], states["reference_position"][0])
    np.testing.assert_array_equal(
        trace["reference_forces"],
        reference["foot_material_wrench"][metadata["start_frame"] + 1 : metadata["start_frame"] + 1001],
    )
    for frame in range(251):
        source_frame = metadata["start_frame"] + 4 * frame
        np.testing.assert_array_equal(states["body_pose"][frame], reference["body_pose"][source_frame])
        np.testing.assert_array_equal(states["reference_position"][frame], reference["particle_position"][source_frame])
        if frame == 0:
            continue
        error = states["particle_position"][frame] - states["reference_position"][frame]
        rmse = np.sqrt(np.mean(np.sum(error**2, axis=-1)))
        mean_z = error[:, 2].mean()
        trace_index = 4 * frame - 1
        maximum_rmse_difference = max(maximum_rmse_difference, abs(float(rmse) - trace["metrics"][trace_index, 0]))
        maximum_mean_z_difference = max(maximum_mean_z_difference, abs(float(mean_z) - trace["metrics"][trace_index, 2]))
        np.testing.assert_allclose(rmse, trace["metrics"][trace_index, 0], atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(mean_z, trace["metrics"][trace_index, 2], atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(states["time_s"][frame], trace["time"][trace_index], atol=1e-10)

error = trace["forces"][..., :3] - trace["reference_forces"][..., :3]
force_rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))
np.testing.assert_allclose(force_rmse, metadata["force_vector_rmse_N"], rtol=1e-6)
probe = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries", "stream=codec_name,r_frame_rate,nb_frames,width,height",
    "-show_entries", "format=duration", "-of", "json", str(RUN / "comparison.mp4"),
]))
assert probe["streams"][0]["r_frame_rate"] == "50/1"
assert int(probe["streams"][0]["nb_frames"]) == 250
assert float(probe["format"]["duration"]) == 5.0

artifacts = [p for p in RUN.iterdir() if p.name != "verification.json"]
artifacts += list((ROOT / "plots").glob("*current*.py"))
for category in ("force_estimation", "particle_prediction"):
    artifacts += list((ROOT / "g1_humanoid" / category / "figures").glob("current_*"))
result = {
    "status": "passed",
    "reference_body_and_particle_snapshots_checked": 251,
    "reference_force_samples_per_foot_checked": 1000,
    "particle_snapshots_matched_to_trace": 250,
    "maximum_snapshot_rmse_difference_m": maximum_rmse_difference,
    "maximum_snapshot_mean_z_difference_m": maximum_mean_z_difference,
    "force_vector_rmse_N": force_rmse,
    "video": probe,
    "video_scope": "Renderer uses saved particles and identical recorded body poses for both panels; no simulation advancement.",
    "artifact_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifacts)},
}
(RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "artifact_sha256"}, indent=2))
