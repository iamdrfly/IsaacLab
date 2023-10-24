# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the common functions that can be used to enable different randomizations.

Randomization includes anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`omni.isaac.orbit.managers.RandomizationTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from omni.isaac.orbit.envs.rl_env import RLEnv


def randomize_rigid_body_material(
    env: RLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  ``(total_body_count, num_shapes)`` where ``total_body_count``
    is the number of assets spawned times the number of bodies per asset and ``num_shapes`` is the number of
    shapes per body. The integer values are used as indices to select the material properties from the
    material buckets.

    .. tip::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)

    # sample material properties from the given ranges
    material_buckets = torch.zeros(num_buckets, 3)
    material_buckets[:, 0].uniform_(*static_friction_range)
    material_buckets[:, 1].uniform_(*dynamic_friction_range)
    material_buckets[:, 2].uniform_(*restitution_range)
    # create random material assignments based on the total number of shapes: num_assets x num_bodies x num_shapes
    material_ids = torch.randint(0, num_buckets, (asset.body_view.count, asset.body_view.num_shapes))
    materials = material_buckets[material_ids]
    # resolve the global body indices from the env_ids and the env_body_ids
    bodies_per_env = asset.body_view.count // num_envs  # - number of bodies per spawned asset
    indices = torch.tensor(asset_cfg.body_ids, dtype=torch.int).repeat(len(env_ids), 1)
    indices += env_ids.unsqueeze(1) * bodies_per_env

    # set the material properties into the physics simulation
    # TODO: Need to use CPU tensors for now. Check if this changes in the new release
    asset.body_physx_view.set_material_properties(materials, indices)


def add_body_mass(env: RLEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, mass_range: tuple[float, float]):
    """Randomize the mass of the bodies by adding a random value sampled from the given range.

    .. tip::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)

    # get the current masses of the bodies (num_assets x num_bodies)
    masses = asset.body_physx_view.get_masses()
    masses += sample_uniform(*mass_range, masses.shape, device=masses.device)
    # resolve the global body indices from the env_ids and the env_body_ids
    bodies_per_env = asset.body_view.count // env.num_envs
    indices = torch.tensor(asset_cfg.body_ids, dtype=torch.int).repeat(len(env_ids), 1)
    indices += env_ids.unsqueeze(1) * bodies_per_env

    # set the mass into the physics simulation
    # TODO: Need to use CPU tensors for now. Check if this changes in the new release
    asset.body_physx_view.set_masses(masses, indices)


def apply_external_force_torque(
    env: RLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)

    # sample random forces and torques
    size = (len(env_ids), len(asset_cfg.body_ids), 3)
    forces = sample_uniform(*force_range, size, asset.device)
    torques = sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def push_by_setting_velocity(
    env: RLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, velocity_range: dict[str, tuple[float, float]]
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    vel_w[:, 0].uniform_(*velocity_range.get("x", (0.0, 0.0)))
    vel_w[:, 1].uniform_(*velocity_range.get("y", (0.0, 0.0)))
    vel_w[:, 2].uniform_(*velocity_range.get("z", (0.0, 0.0)))
    vel_w[:, 3].uniform_(*velocity_range.get("roll", (0.0, 0.0)))
    vel_w[:, 4].uniform_(*velocity_range.get("pitch", (0.0, 0.0)))
    vel_w[:, 5].uniform_(*velocity_range.get("yaw", (0.0, 0.0)))
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state(
    env: RLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
):
    """Reset the asset root state to a random position and velocity within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state_w[env_ids].clone()

    # positions
    pos_offset = torch.zeros_like(root_states[:, 0:3])
    pos_offset[:, 0].uniform_(*pose_range.get("x", (0.0, 0.0)))
    pos_offset[:, 1].uniform_(*pose_range.get("y", (0.0, 0.0)))
    pos_offset[:, 2].uniform_(*pose_range.get("z", (0.0, 0.0)))
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pos_offset
    # orientations
    euler_angles = torch.zeros_like(positions)
    euler_angles[:, 0].uniform_(*pose_range.get("roll", (0.0, 0.0)))
    euler_angles[:, 1].uniform_(*pose_range.get("pitch", (0.0, 0.0)))
    euler_angles[:, 2].uniform_(*pose_range.get("yaw", (0.0, 0.0)))
    orientations = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    # velocities
    velocities = root_states[:, 7:13]
    velocities[:, 0].uniform_(*velocity_range.get("x", (0.0, 0.0)))
    velocities[:, 1].uniform_(*velocity_range.get("y", (0.0, 0.0)))
    velocities[:, 2].uniform_(*velocity_range.get("z", (0.0, 0.0)))
    velocities[:, 3].uniform_(*velocity_range.get("roll", (0.0, 0.0)))
    velocities[:, 4].uniform_(*velocity_range.get("pitch", (0.0, 0.0)))
    velocities[:, 5].uniform_(*velocity_range.get("yaw", (0.0, 0.0)))

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
    env: RLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # scale these values randomly
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
