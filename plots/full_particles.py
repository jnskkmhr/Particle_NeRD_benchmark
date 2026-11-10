"""Plot particle displacement from initialization; these are not errors against MPM."""

from full_common import DATA, MPM_DATA, BLUE, GREEN, load_profile, np, plt, save, selected

for name in selected():
    _, meta = load_profile(name)
    metrics = np.load(DATA / name / "particle_metrics.npz")["metrics"]
    mpm_metrics = np.load(MPM_DATA / name / "particle_metrics.npz")["metrics"]
    assert np.isfinite(metrics).all() and np.isfinite(mpm_metrics).all()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
    for c, ylabel in enumerate(
        ("RMS 3D displacement\nfrom initial positions (mm)", "Mean z displacement\nfrom initial positions (mm)"), 1
    ):
        ax = axes[c - 1]
        ax.plot(mpm_metrics[:, 0], mpm_metrics[:, c] * 1000, color=BLUE, lw=1.6, label="Implicit MPM")
        ax.plot(metrics[:, 0], metrics[:, c] * 1000, color=GREEN, lw=1.6, ls="--", label="Particle NeRD")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)
        ax.set_xlim(0, max(meta["recorded_seconds"], mpm_metrics[-1, 0]))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.99, wspace=0.27)
    save(fig, "particle_prediction", f"full_{name}_displacement")
