"""Paired force errors and empirical distributions over all 50 validation windows."""

from common import COLORS, G1, np, plt, read, save

source = read(G1 / "force_estimation/validation-force.json")
keys = ("reference", "initial", "pilot")
labels = {"reference": "Estimator with MPM particles", "initial": "Original model", "pilot": "Contact-trained model"}
records = {key: {(row["trial_id"], row["frame"]): row for row in source[key]["per_episode"]} for key in keys}
ids = sorted(records["reference"])
assert len(ids) == 50 and all(set(records[k]) == set(ids) for k in keys)
errors = {}
for key in keys:
    rows = [records[key][idx] for idx in ids]
    errors[key] = np.sqrt([row["force_squared_sum"] / row["foot_time_count"] for row in rows])
    aggregate = np.sqrt(sum(row["force_squared_sum"] for row in rows) / sum(row["foot_time_count"] for row in rows))
    assert np.isclose(aggregate, source[key]["force_vector_rmse_N"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
ax = axes[0]
limit = 1.06 * max(errors["initial"].max(), errors["pilot"].max())
ax.plot([0, limit], [0, limit], color="0.6", lw=1, ls=":", label="Equal error")
ax.scatter(
    errors["initial"],
    errors["pilot"],
    c=COLORS["pilot"],
    s=34,
    edgecolors="white",
    linewidths=0.5,
    label="One validation window",
)
ax.set(
    xlim=(0, limit), ylim=(0, limit), xlabel="Original-model force RMSE (N)", ylabel="Contact-trained force RMSE (N)"
)
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper left")
for key, style in zip(keys, (":", "-", "--")):
    axes[1].step(
        np.r_[0, np.sort(errors[key])],
        np.r_[0, np.arange(1, 51) / 50],
        where="post",
        color=COLORS[key],
        ls=style,
        lw=1.8,
        label=labels[key],
    )
axes[1].set(
    xlabel="Per-window, per-foot force RMSE (N)",
    ylabel="Fraction of validation windows",
    xlim=(0, None),
    ylim=(0, 1.03),
)
axes[1].legend(frameon=False, loc="lower right")
save(fig, "force_estimation", "validation_distribution")
