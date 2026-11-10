"""Plot current-solver particle errors against the matching recorded MPM states."""

from current_common import NERD, load, plt, save

data, metadata = load()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(data["time"], 1000 * data["metrics"][:, 0], color=NERD, lw=1.5)
axes[0].set_ylabel("Particle-position RMSE (mm)")
axes[0].set_ylim(bottom=0)
axes[1].plot(data["time"], 1000 * data["metrics"][:, 2], color=NERD, lw=1.5)
axes[1].axhline(0, color="#666666", lw=0.8, linestyle=":")
axes[1].set_ylabel("Mean signed particle z error (mm)")
for ax in axes:
    ax.set_xlabel("Elapsed replay time (s)")
    ax.set_xlim(0, data["time"][-1])
    ax.grid(alpha=0.15)
fig.subplots_adjust(top=0.91, bottom=0.15, left=0.07, right=0.99, wspace=0.27)
save(fig, "particle_prediction", "current_particle_error")
