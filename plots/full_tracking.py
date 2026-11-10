"""Plot commanded and actual body-frame robot velocities in the paired independent closed-loop runs."""

from full_common import BLUE, GREEN, load_profile, plt, save, selected

for name in selected():
    data, meta = load_profile(name)
    mpm, _ = load_profile(name, "mpm")
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for k, (key, c, ylabel) in enumerate(
        [
            ("root_velocity_body", 0, "$v_x$ (m/s)"),
            ("root_velocity_body", 1, "$v_y$ (m/s)"),
            ("root_angular_velocity_body", 2, "$\\omega_z$ (rad/s)"),
        ]
    ):
        axes[k].plot(data["time"], data[key][:, c], color=GREEN, lw=1.2, label="Particle NeRD", ls="--")
        axes[k].plot(mpm["time"], mpm[key][:, c], color=BLUE, lw=1.2, label="Implicit MPM")
        axes[k].plot(
            [0, max(data["time"][-1], mpm["time"][-1])],
            [data["command"][0, k]] * 2,
            color="#555555",
            ls=":",
            lw=1.4,
            label="Command",
        )
        axes[k].set_ylabel(ylabel)
        axes[k].grid(alpha=0.15)
        axes[k].set_xlim(0, max(data["time"][-1], mpm["time"][-1]))
    axes[-1].set_xlabel("Simulation time (s)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.9, bottom=0.09, left=0.09, right=0.99, hspace=0.35)
    save(fig, "force_estimation", f"full_{name}_tracking")
