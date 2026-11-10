"""Plot forces from each closed-loop rollout, including rigid-bed contact."""

from full_common import BLUE, GREEN, LIGHT_GREEN, load_profile, plt, save, selected

for name in selected():
    data, meta = load_profile(name)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey="row")
    for c in range(3):
        for f in range(2):
            ax = axes[c, f]
            for key, color, style, label in [
                ("force", GREEN, "-", "Total contact force on foot"),
                ("sand_force", LIGHT_GREEN, "--", "Sand contribution"),
                ("rigid_force", BLUE, ":", "Rigid-geometry contribution"),
            ]:
                ax.plot(data["time"], data[key][:, f, c], color=color, ls=style, lw=1.2, label=label)
            ax.set_ylabel(f"$F_{'xyz'[c]}$ (N)")
            ax.set_xlim(0, data["time"][-1])
            ax.grid(alpha=0.15)
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (s)")
    for foot, ax in enumerate(axes[0]):
        ax.text(0.5, 1.03, ("Left foot", "Right foot")[foot], transform=ax.transAxes, ha="center", va="bottom")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.99, hspace=0.25, wspace=0.16)
    save(fig, "force_estimation", f"full_{name}_force")
