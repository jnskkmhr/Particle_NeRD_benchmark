"""Compare world-frame foot reactions in independent closed-loop MPM and NeRD runs."""

from full_common import BLUE, GREEN, load_profile, plt, save, selected

for name in selected():
    runs = [
        (load_profile(name, backend)[0], color, label, style)
        for backend, color, label, style in [("mpm", BLUE, "Implicit MPM", "-"), ("nerd", GREEN, "Particle NeRD", "--")]
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey="row")
    for c in range(3):
        for f in range(2):
            ax = axes[c, f]
            for data, color, label, style in runs:
                ax.plot(data["time"], data["force"][:, f, c], color=color, ls=style, lw=1.2, label=label)
            ax.set_ylabel(f"$F_{'xyz'[c]}$ (N)")
            ax.set_xlim(0, max(d[0]["time"][-1] for d in runs))
            ax.grid(alpha=0.15)
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (s)")
    for foot, ax in enumerate(axes[0]):
        ax.text(0.5, 1.03, ("Left foot", "Right foot")[foot], transform=ax.transAxes, ha="center", va="bottom")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.99, hspace=0.25, wspace=0.16)
    save(fig, "force_estimation", f"full_{name}_force")
