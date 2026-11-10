"""Plot particle displacement from initialization; these are not errors against MPM."""

from full_common import DATA, GREEN, load_profile, np, plt, save, selected

for name in selected():
    _, meta = load_profile(name)
    metrics = np.load(DATA / name / "particle_metrics.npz")["metrics"]
    assert np.isfinite(metrics).all()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
    for c, ylabel in enumerate(
        ("RMS 3D displacement\nfrom initial positions (mm)", "Mean z displacement\nfrom initial positions (mm)"), 1
    ):
        ax = axes[c - 1]
        ax.plot(metrics[:, 0], metrics[:, c] * 1000, color=GREEN, lw=1.6)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)
        ax.set_xlim(0, meta["recorded_seconds"])
    fig.subplots_adjust(top=0.89, bottom=0.16, left=0.08, right=0.99, wspace=0.27)
    save(fig, "particle_prediction", f"full_{name}_displacement")
