"""Plot the current solver's foot forces and the matching recorded MPM forces."""

from current_common import MPM, NERD, load, plt, save

data, metadata = load()
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey="row")
for component in range(3):
    for foot in range(2):
        ax = axes[component, foot]
        ax.plot(data["time"], data["reference_forces"][:, foot, component], color=MPM, lw=1.4, label="Implicit MPM")
        ax.plot(
            data["time"], data["forces"][:, foot, component], color=NERD, lw=1.2, linestyle="--", label="Particle NeRD"
        )
        ax.set_ylabel(f"$F_{'xyz'[component]}$ (N)")
        ax.grid(alpha=0.15)
        ax.set_xlim(0, data["time"][-1])
for ax in axes[-1]:
    ax.set_xlabel("Elapsed replay time (s)")
for foot, ax in enumerate(axes[0]):
    ax.text(0.5, 1.03, ("Left foot", "Right foot")[foot], transform=ax.transAxes, ha="center", va="bottom")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.99, hspace=0.25, wspace=0.16)
save(fig, "force_estimation", "current_force_components")
