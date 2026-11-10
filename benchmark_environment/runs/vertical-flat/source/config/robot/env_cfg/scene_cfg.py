# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka sand-pushing scene, with a fixed base beside the tray."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from particle_nerd_benchmark.core.terrain import configure_terrain


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    robot = replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.variants["Physics"] = "mujoco"
    robot.init_state.pos = (-0.85, 0.0, 0.10)
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.50,
        "panda_joint3": 0.0,
        "panda_joint4": -2.0,
        "panda_joint5": 0.0,
        "panda_joint6": 1.5,
        "panda_joint7": 0.7854,
        "panda_finger_joint.*": 0.0,
    }
    robot.actuators["panda_arm"].stiffness = 800.0
    robot.actuators["panda_arm"].damping = 60.0
    robot.actuators["panda_hand"].stiffness = 400.0
    robot.actuators["panda_hand"].damping = 40.0

    def __post_init__(self):
        configure_terrain(self, "flat", (1.2, 1.0, 0.25))
