"""Dataset aggregate position errors with separate contact-region scale."""

from common import COLORS, G1, np, plt, read, save

i = read(G1 / "particle_prediction/validation-initial-direct.json")
p = read(G1 / "particle_prediction/validation-local-pilot-direct.json")
assert i["validation_ids"] == p["validation_ids"] and len(i["validation_ids"]) == 50
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
for ax, metrics, names in [
    (axes[0], ("position_rmse_m", "moving_rmse_m"), ("Every bed particle", "Displaced from\nwindow-start position")),
    (axes[1], ("contact_region_rmse_m",), ("Within 2 cm of foot boxes",)),
]:
    x = np.arange(len(metrics))
    for offset, data, key, label, hatch in [(-0.18, i, "initial", "Original model", ""), (0.18, p, "pilot", "Contact-trained model", "//")]:
        values = [data["metrics"][metric] * 1000 for metric in metrics]
        bars = ax.bar(x + offset, values, width=0.34, color=COLORS[key], label=label, hatch=hatch)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_xticks(x, names)
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_ylim(0, max(data["metrics"][metric] * 1000 for data in (i, p) for metric in metrics) * 1.3)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2)
save(fig, "particle_prediction", "validation_regions")
