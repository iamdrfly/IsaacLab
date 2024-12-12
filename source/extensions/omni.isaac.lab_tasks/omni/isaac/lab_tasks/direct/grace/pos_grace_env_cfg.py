# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.mdp import UniformPose2dCommandCfg, TerrainBasedPose2dCommandCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.terrain_generator import FlatPatchSamplingCfg
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.grace import GRACE_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.terrains.config.supsi_rough import SUPSI_ROUGH_TERRAINS_CFG, CUBES_SUPSI_TERRAINS_CFG, SUPSI_FLAT_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import random
import math
import torch
from omni.isaac.lab.envs import ViewerCfg

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #             "force_range": (0.0, 0.0), "torque_range": (-0.0, 0.0),
    #             }, )


@configclass
class PosGraceFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.0
    decimation = 4
    action_scale = 0.5
    action_space = 24
    observation_space = 48
    state_space = 0
    is_finite_horizon = True

    # camera
    viewer = ViewerCfg()
    viewer.eye = (-85, -30, 10)
    viewer.lookat = (40, 95, 0)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SUPSI_FLAT_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = GRACE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # ranges = {"pos_x": (1.0, 5.0), "pos_y": (1.0, 5.0), "heading":(-math.pi, math.pi)}
    pose_command : TerrainBasedPose2dCommandCfg = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot", #dict key which is associated the robot articulation (ours: robot)
        simple_heading=True,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(5.0, 5.0), pos_y=(0, 0), heading=(0, 0)),
    )

    vacuum_visualizer : VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/vacuumMarker",
        markers={
            "cylinder_no_contact": sim_utils.CylinderCfg(
                radius=0.02,
                height=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)), #0-grey
            ),
            "cylinder_contact": sim_utils.CylinderCfg(
                radius=0.02,
                height=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0., 1.)), #1-blu
            ),
            "cylinder_vacuum": sim_utils.CylinderCfg(
                radius=0.02,
                height=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 1., 0.)), #2-green
            )
        }
    )

    # reward scales
    position_tracking_reward_scale  = 13.7
    heading_tracking_reward_scale   = 5.77
    joint_vel_reward_scale          = -0.000787
    joint_torque_reward_scale       = -2.49/10**6
    joint_vel_limit_reward_scale    = -8.134
    joint_torque_limit_reward_scale = -0.1767
    base_acc_reward_scale           = -0.00102
    base_lin_acc_weight             = 1.090
    base_ang_acc_weight             = 0.0206
    feet_acc_reward_scale           = -0.000172
    action_rate_reward_scale        = -0.01
    max_feet_contact_force          = 600.
    feet_contact_force_reward_scale = -8.851/10**6
    wait_time                       = 0.268
    dont_wait_reward_scale          = -1.317
    move_in_direction_reward_scale  = 2.
    stand_min_dist                  = 0.20
    stand_min_ang                   = 0.58
    stand_at_target_reward_scale    = -0.5
    undesired_contact_reward_scale  = -1.56
    stumble_reward_scale            = -2.93
    feet_termination_force          = 1455.
    termination_reward_scale        = -230.
    theta_marg_sum_reward_scale     = 1.

    show_flat_patches = False # da passare come args
    color_scheme = "height" #["height", "random", None]

    if color_scheme in ["height", "random"]:
        terrain.visual_material = None


@configclass
class PosGraceRoughEnvCfg(PosGraceFlatEnvCfg):
    # env
    observation_space = 236 #235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CUBES_SUPSI_TERRAINS_CFG, #SUPSI_ROUGH_TERRAINS_CFG, #CUBES_SUPSI_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
    feet_air_time_reward_scale = 0.5*1.1
    lin_vel_reward_scale = 1.0*4

    if PosGraceFlatEnvCfg().color_scheme in ["height", "random"]:
        terrain.visual_material = None