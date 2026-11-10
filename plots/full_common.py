"""Read and validate current closed-loop command-profile recordings."""

import argparse
from common import G1, np, plt, read, save

GREEN = "#187A3B"
BLUE = "#17365D"
LIGHT_GREEN = "#6C9F3F"
DATA = G1 / "data/full-20261109"
PROFILES = ("forward", "backward", "left", "right", "forward_yaw_positive", "forward_yaw_negative")


def load_profile(name):
    folder = DATA / name
    meta = read(folder / "metadata.json")
    data = np.load(folder / "trace.npz")
    assert not meta["simulation_uses_mpm_reference"] and not meta["simulation_uses_robot_reference"]
    assert np.allclose(data["time"], np.arange(1, len(data["time"]) + 1) * 0.005)
    np.testing.assert_allclose(data["force"], data["sand_force"] + data["rigid_force"], atol=1e-5)
    cmd = meta["command_body_frame"]
    expected = np.array([cmd["vx_m_s"], cmd["vy_m_s"], cmd["wz_rad_s"]])
    assert np.allclose(data["command"][:, :3], expected)
    for key in data.files:
        assert np.isfinite(data[key]).all(), key
    return data, meta


def selected():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=PROFILES)
    a = p.parse_args()
    return (a.profile,) if a.profile else PROFILES
