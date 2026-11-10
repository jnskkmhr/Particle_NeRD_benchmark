"""Recorded-motion particle errors; all 1,000 samples, without smoothing."""

from common import COLORS, plt, replay, save

a, time = replay()
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
axes[0].plot(time, a["metrics"][:, 0] * 1000, color=COLORS["pilot"], lw=1.7)
axes[0].set(ylabel="Every bed particle: position RMSE (mm)", ylim=(0, None))
axes[1].plot(
    time, a["metrics"][:, 1] * 1000, color=COLORS["pilot"], lw=1.5, label="Within 15 cm horizontally of a body origin"
)
axes[1].plot(time, a["metrics"][:, 2] * 1000, color=COLORS["initial"], lw=1.3, ls="--", label="Every bed particle")
axes[1].axhline(0, color="0.5", lw=0.8, ls=":")
axes[1].set(xlabel="Time since replay start (s)", ylabel="Mean signed vertical error (mm)", xlim=(0, 5))
axes[1].legend(frameon=False, loc="best")
save(fig, "particle_prediction", "prediction_over_time")
