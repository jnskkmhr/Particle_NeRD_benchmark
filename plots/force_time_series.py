"""All recorded force samples, compared on shared recorded body motion."""

from common import COLORS, plt, replay, save

a, time = replay()
fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True, sharey="row", layout="constrained")
for component, letter in enumerate("xyz"):
    for foot, name in enumerate(("Left", "Right")):
        ax = axes[component, foot]
        ax.plot(
            time, a["reference_forces"][:, foot, component], color=COLORS["reference"], lw=1.4, label="MPM reference"
        )
        ax.plot(time, a["forces"][:, foot, component], color=COLORS["pilot"], lw=1.1, ls="--", label="Contact-trained model")
        ax.set_ylabel(f"{name} foot F{letter} (N)")
        ax.set_xlim(0, 5)
        ax.axhline(0, color="0.8", lw=0.6, zorder=0)
        if component == 2:
            ax.set_xlabel("Time since replay start (s)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
save(fig, "force_estimation", "force_components")
