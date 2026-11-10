"""Flat-ground QuadRL baseline for Isaac Lab 2.3 and Unitree Go2.

Place this module beside mdp/rewards.py, mdp/observations.py, and
mdp/terminations.py in your task package. Use QuadRLFlatEnvCfg as the
env_cfg_entry_point for isaaclab.envs:ManagerBasedRLEnv.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from .mdp.observations import RapidLocomotionObservationsCfg
from .mdp.rewards import RapidLocomotionRewardsCfg
from .mdp.terminations import RapidLocomotionTerminationsCfg


@configclass
class QuadRLSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=4,
        track_air_time=True,
        force_threshold=1.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/sky_light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )


@configclass
class QuadRLCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.05,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class QuadRLActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class QuadRLEventsCfg:
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.10, 0.10), "velocity_range": (0.0, 0.0)},
    )


@configclass
class QuadRLFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: QuadRLSceneCfg = QuadRLSceneCfg(num_envs=256, env_spacing=2.5)
    observations: RapidLocomotionObservationsCfg = RapidLocomotionObservationsCfg()
    rewards: RapidLocomotionRewardsCfg = RapidLocomotionRewardsCfg()
    terminations: RapidLocomotionTerminationsCfg = RapidLocomotionTerminationsCfg()
    commands: QuadRLCommandsCfg = QuadRLCommandsCfg()
    actions: QuadRLActionsCfg = QuadRLActionsCfg()
    events: QuadRLEventsCfg = QuadRLEventsCfg()

    def __post_init__(self):
        self.episode_length_s = 20.0
        self.is_finite_horizon = False

        # 200 Hz physics, 50 Hz policy and reward.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.history_length = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.rewards.base_height.params["target_height"] = self.scene.robot.init_state.pos[2]
        self.rewards.foot_airtime.params["sensor_cfg"].body_names = ".*_foot"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
