"""Isaac Lab 2.3 Manager-Based observations for Rapid Locomotion via RL.

Use RapidLocomotionObservationsCfg() in env_cfg.py for a standard PPO
policy with one frame per step. The scene needs a 12-joint "robot",
a 12-D joint-position action, and the same "base_velocity" command
used by mdp/rewards.py. Keep joint and action ordering consistent
with your robot configuration and deployment code.

For a custom student runner use RapidLocomotionStudentObservationsCfg()
and split_student_observation(obs["policy"]) to get x_t and the flattened
15-frame history. The privileged teacher encoder additionally needs the
actual parameters sampled by your domain-randomization events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

if TYPE_CHECKING:
    import torch

HISTORY_LENGTH = 15


@configclass
class RapidLocomotionPolicyObsCfg(ObsGroup):
    """One 42-D frame in the order used by the authors' implementation."""

    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        scale=(2.0, 2.0, 0.25),
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        # Isaac Lab adds noise before applying the observation scale.
        noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        scale=0.05,
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class RapidLocomotionObservationsCfg:
    """Observation config for a single-frame PPO baseline."""

    policy: RapidLocomotionPolicyObsCfg = RapidLocomotionPolicyObsCfg()


@configclass
class RapidLocomotionStudentPolicyObsCfg(RapidLocomotionPolicyObsCfg):
    """Store x[t-15:t+1] as (num_envs, 16, 42), oldest frame first."""

    def __post_init__(self):
        super().__post_init__()
        self.history_length = HISTORY_LENGTH + 1
        self.flatten_history_dim = False
        self.concatenate_dim = -1


@configclass
class RapidLocomotionStudentObservationsCfg:
    """Observation config for a runner with a separate adaptation network."""

    policy: RapidLocomotionStudentPolicyObsCfg = RapidLocomotionStudentPolicyObsCfg()


def split_student_observation(
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the current 42-D frame and the previous 15 frames as 630 features."""
    if frames.ndim != 3 or frames.shape[1:] != (HISTORY_LENGTH + 1, 42):
        raise ValueError("Expected policy observations of shape (num_envs, 16, 42).")
    current_frame = frames[:, -1, :]
    adaptation_history = frames[:, :-1, :].flatten(start_dim=1)
    return current_frame, adaptation_history
