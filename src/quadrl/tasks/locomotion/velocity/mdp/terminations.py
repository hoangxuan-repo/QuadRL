"""Episode termination terms for the QuadRL Isaac Lab 2.3 task.

In env_cfg.py, set terminations = RapidLocomotionTerminationsCfg() and
episode_length_s = 20.0. Reuse the "contact_forces" sensor from rewards.py
and make sure its body selection includes the robot's trunk.
"""

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# Set this to the exact trunk link name if the robot uses a different name.
BASE_BODY_PATTERN = r"(?i)(?:base|base_link|body|trunk|torso)"


@configclass
class RapidLocomotionTerminationsCfg:
    """Separate the episode time limit from a physical failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # The original environment checks current force on configured links.
    # Isaac Lab's illegal_contact checks the contact sensor's force history.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY_PATTERN),
            "threshold": 1.0,  # N
        },
    )
