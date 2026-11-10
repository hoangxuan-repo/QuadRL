"""Isaac Lab 2.3 Manager-Based rewards for Rapid Locomotion via RL (Table VI).

Place this file in your task's mdp package. In env_cfg.py use
`from .mdp.rewards import RapidLocomotionRewardsCfg` and set
`rewards: RapidLocomotionRewardsCfg = RapidLocomotionRewardsCfg()`.

Scene setup:
- "robot": articulation with activate_contact_sensors=True.
- "contact_forces": ContactSensor covering the robot's links. Set
  track_air_time=True, update_period=0.0, and history_length at least equal
  to decimation so the collision proxy can see each physics substep.
- "base_velocity": command with [vx, vy, wz] in the robot base frame.
- Adjust body-name patterns and base_height.params["target_height"] to your
  robot. For rough terrain set base_height.params["sensor_cfg"] to a
  SceneEntityCfg for a downward terrain RayCaster (e.g. "height_scanner").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# At the reference 50 Hz policy rate, RewardManager multiplies these
# weights by 0.02 s to recover the coefficients printed in Table VI.
REFERENCE_STEP_DT = 0.02
FOOT_PATTERN = r"(?i).*(?:foot|feet).*"
COLLISION_PATTERN = r"(?i).*(?:base|trunk|torso|thigh|calf|shank).*"


def joint_limit_violation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count joints outside the nominal hard limits, one indicator per joint."""
    robot = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.default_joint_pos_limits[:, asset_cfg.joint_ids]
    return ((pos < limits[..., 0]) | (pos > limits[..., 1])).sum(dim=1).float()


def collision_indicator(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    self_contact_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Return one collision indicator per environment.

    With filtered sensors, use one ContactSensor per monitored robot link
    and filter each to other links of the same robot in that environment.
    Enable self-collisions in the articulation and exclude terrain from the
    filters. Filtered forces are sampled at the reward step.

    The default uses recent net forces on selected non-foot bodies as a
    proxy and also counts terrain contact on those bodies.
    """
    if self_contact_sensor_names:
        collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in self_contact_sensor_names:
            forces = env.scene.sensors[name].data.force_matrix_w
            if forces is None:
                raise ValueError(f"{name} requires robot-link contact filters.")
            collision |= (torch.linalg.vector_norm(forces, dim=-1) > threshold).any(dim=(1, 2))
        return collision.float()

    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).any(dim=(1, 2)).float()


def foot_airtime_on_touchdown(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sum the completed swing durations for feet touching down this step."""
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return (air_time * first_contact).sum(dim=1)


@configclass
class RapidLocomotionRewardsCfg:
    """Table VI coefficients converted to Isaac Lab reward rates."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.02 / REFERENCE_STEP_DT,
        # Isaac Lab divides by std**2, so std=0.5 gives sigma=0.25.
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_yaw_rate = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.01 / REFERENCE_STEP_DT,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.04 / REFERENCE_STEP_DT)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.001 / REFERENCE_STEP_DT)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.6 / REFERENCE_STEP_DT,
        # Example target for a small quadruped on a z=0 plane.
        params={"target_height": 0.30, "sensor_cfg": None},
    )
    base_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.002 / REFERENCE_STEP_DT,
    )
    self_collision = RewTerm(
        func=collision_indicator,
        weight=-0.02 / REFERENCE_STEP_DT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=COLLISION_PATTERN),
            "threshold": 1.0,
            "self_contact_sensor_names": (),
        },
    )
    joint_limits = RewTerm(
        func=joint_limit_violation,
        weight=-0.2 / REFERENCE_STEP_DT,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-7 / REFERENCE_STEP_DT)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-9 / REFERENCE_STEP_DT)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2.0e-4 / REFERENCE_STEP_DT)
    foot_airtime = RewTerm(
        func=foot_airtime_on_touchdown,
        weight=0.02 / REFERENCE_STEP_DT,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_PATTERN)},
    )

