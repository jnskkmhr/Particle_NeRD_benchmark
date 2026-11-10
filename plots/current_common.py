"""Load the November 9 current-solver replay and apply the shared figure style."""

from common import G1, np, plt, read, save

DATA = G1 / "data/current-20261109"
MPM = "#17365D"
NERD = "#187A3B"


def load():
    metadata = read(DATA / "metadata.json")
    with np.load(DATA / "trace.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    assert data["forces"].shape == data["reference_forces"].shape == (metadata["steps"], 2, 6)
    assert np.allclose(data["time"], np.arange(1, metadata["steps"] + 1) * metadata["dt_s"])
    for values in data.values():
        assert np.isfinite(values).all()
    error = data["forces"][..., :3] - data["reference_forces"][..., :3]
    assert np.isclose(np.sqrt(np.mean(np.sum(error**2, axis=-1))), metadata["force_vector_rmse_N"])
    assert np.isclose(data["metrics"][-1, 0], metadata["final_particle_position_rmse_m"])
    return data, metadata
