"""Shared plotting style and archived-data loading; no simulation execution."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "g1_humanoid"
COLORS = {"reference": "#17365D", "initial": "#65758B", "pilot": "#187A3B"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    }
)


def read(path):
    return json.loads(path.read_text())


def replay():
    provenance = read(G1 / "data/recorded-local-direct-provenance.json")
    archive = G1 / "data/recorded-local-direct.npz"
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    assert arrays["forces"].shape == arrays["reference_forces"].shape == (1000, 2, 6)
    assert arrays["metrics"].shape == (1000, 3)
    assert np.allclose(np.diff(arrays["time"]), provenance["dt"], atol=1e-6)
    for key in ("forces", "reference_forces", "metrics", "time"):
        assert np.isfinite(arrays[key]).all(), key
    # First prediction is at 0.305 s; history starts at frame 60 = 0.300 s.
    elapsed = arrays["time"] - provenance["start_frame"] * provenance["dt"]
    assert np.isclose(elapsed[-1], 5.0)
    summary = read(G1 / "data/recorded-local-direct-summary.json")["recorded-local-direct"]
    error = arrays["forces"][..., :3] - arrays["reference_forces"][..., :3]
    assert np.isclose(np.sqrt(np.mean(np.sum(error**2, axis=-1))), summary["force_vector_rmse_N"], rtol=1e-6)
    assert np.isclose(arrays["metrics"][-1, 0], summary["final_position_rmse_m"])
    return arrays, elapsed


def save(fig, category, name):
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", fig.axes):
        ax.text(0.0, 1.03, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    folder = G1 / category / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(folder / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(category + "/" + name)
