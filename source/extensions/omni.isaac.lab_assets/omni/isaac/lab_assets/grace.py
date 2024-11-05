# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the GRACE robots.

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetLSTMCfg, DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .velodyne import VELODYNE_VLP_16_RAYCASTER_CFG
from omni.isaac.lab.utils.pathfinder.pathfinder import find_absolute_path

##
# Configuration - Actuators.
##

# GRACE_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
#     joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
#     saturation_effort=120.0,
#     effort_limit=80.0,
#     velocity_limit=7.5,
#     stiffness={".*": 40.0},
#     damping={".*": 5.0},
# )

# GRACE_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
#     joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
#     network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
#     saturation_effort=120.0,
#     effort_limit=80.0,
#     velocity_limit=7.5,
# )
# """Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""

GRACE_HAA_CFG = DCMotorCfg(
    # joint_names_expr=[".*HAA"],
    joint_names_expr=["RR_HAA", "RF_HAA", "LR_HAA", "LF_HAA"],
    saturation_effort=160.0,
    effort_limit=140.0,
    velocity_limit=5.0,
    stiffness={".*": 25.0},
    damping={".*": .5},
    # saturation_effort=120.0,
    # effort_limit=80.0,
    # velocity_limit=7.5,
    # stiffness={".*": 40.0},
    # damping={".*": 5.0},
)

GRACE_HFE_CFG = DCMotorCfg(
    # joint_names_expr=[".*HFE", ".*KFE"],
    joint_names_expr=["RR_HFE", "RF_HFE", "LR_HFE", "LF_HFE"],
    saturation_effort=60.0,
    effort_limit=38.0,
    velocity_limit=14.0,
    stiffness={".*": 25.0},
    damping={".*": .5},
    # saturation_effort=120.0,
    # effort_limit=80.0,
    # velocity_limit=7.5,
    # stiffness={".*": 40.0},
    # damping={".*": 5.0},
)

GRACE_KFE_CFG = DCMotorCfg(
    # joint_names_expr=[".*HFE", ".*KFE"],
    joint_names_expr=["RR_KFE", "RF_KFE", "LR_KFE", "LF_KFE"],
    saturation_effort=60.0,
    effort_limit=38.0,
    velocity_limit=14.0,
    stiffness={".*": 25.0},
    damping={".*": .5},
    # saturation_effort=120.0,
    # effort_limit=80.0,
    # velocity_limit=7.5,
    # stiffness={".*": 40.0},
    # damping={".*": 5.0},
)

GRACE_SPHERICAL_CFG = DCMotorCfg(
    joint_names_expr=[".*_FOOT:.*"],
    saturation_effort=1.0,
    effort_limit=1.,
    velocity_limit=1.,
    stiffness={".*": 0.5}, #0.5 gira bene
    damping={".*": .0},
)


##
# Configuration - Articulation.
##
GRACE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= find_absolute_path("grace_one_ball_z.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "LF_HAA": -0.7854,  #
            "LF_HFE":  1.5708,  #
            "LF_KFE":  1.5708,  #

            "LR_HAA":  0.7854,  #
            "LR_HFE":  -1.5708, #
            "LR_KFE":  -1.5708, #

            "RF_HAA":  0.7854,  #0.7854
            "RF_HFE":  1.5708,  #1.57
            "RF_KFE":  1.5708,  #1.57

            "RR_HAA": -0.7854,  #-0.7854
            "RR_HFE": -0.7854,  #-0.7854
            "RR_KFE":  0,       #0
        },
    ),
    actuators={"HAA": GRACE_HAA_CFG, "HFE": GRACE_HFE_CFG,"KFE": GRACE_KFE_CFG}, #"sphericals": GRACE_SPHERICAL_CFG
    # actuators={"legs": GRACE_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)


##
# Configuration - Sensors.
##

GRACE_CFG = VELODYNE_VLP_16_RAYCASTER_CFG.replace(
    offset=RayCasterCfg.OffsetCfg(pos=(-0.310, 0.000, 0.159), rot=(0.0, 0.0, 0.0, 1.0))
)
"""Configuration for the Velodyne VLP-16 sensor mounted on the ANYmal robot's base."""
