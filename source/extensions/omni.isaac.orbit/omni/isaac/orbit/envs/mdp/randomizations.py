# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different randomizations.

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
from omni.isaac.orbit.managers.manager_base import ManagerTermBase
from omni.isaac.orbit.managers.manager_term_cfg import RandomizationTermCfg
from omni.isaac.orbit.terrains import TerrainImporter
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, random_orientation, sample_uniform

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


def randomize_rigid_body_material(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    asset_cfg: SceneEntityCfg,
):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (total_body_count, num_shapes) where ``total_body_count``
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
        env_ids = torch.arange(num_envs, device="cpu")
    # resolve body indices
    if isinstance(asset_cfg.body_ids, slice):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")[asset_cfg.body_ids]
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample material properties from the given ranges
    material_buckets = torch.zeros(num_buckets, 3)
    material_buckets[:, 0].uniform_(*static_friction_range)
    material_buckets[:, 1].uniform_(*dynamic_friction_range)
    material_buckets[:, 2].uniform_(*restitution_range)
    # create random material assignments based on the total number of shapes: num_assets x num_bodies x num_shapes
    material_ids = torch.randint(0, num_buckets, (asset.body_physx_view.count, asset.body_physx_view.max_shapes))
    materials = material_buckets[material_ids]
    # resolve the global body indices from the env_ids and the env_body_ids
    bodies_per_env = asset.body_physx_view.count // num_envs  # - number of bodies per spawned asset
    indices = body_ids.repeat(len(env_ids), 1)
    indices += env_ids.unsqueeze(1) * bodies_per_env

    # set the material properties into the physics simulation
    asset.body_physx_view.set_material_properties(materials, indices)


def add_body_mass(
    env: BaseEnv, env_ids: torch.Tensor | None, mass_range: tuple[float, float], asset_cfg: SceneEntityCfg
):
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
        env_ids = torch.arange(num_envs, device="cpu")
    # resolve body indices
    if isinstance(asset_cfg.body_ids, slice):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")[asset_cfg.body_ids]
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets x num_bodies)
    masses = asset.body_physx_view.get_masses()
    masses += sample_uniform(*mass_range, masses.shape, device=masses.device)
    # resolve the global body indices from the env_ids and the env_body_ids
    bodies_per_env = asset.body_physx_view.count // env.num_envs
    indices = body_ids.repeat(len(env_ids), 1)
    indices += env_ids.unsqueeze(1) * bodies_per_env

    # set the mass into the physics simulation
    asset.body_physx_view.set_masses(masses, indices)


def apply_external_force_torque(
    env: BaseEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = sample_uniform(*force_range, size, asset.device)
    torques = sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def push_by_setting_velocity(
    env: BaseEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w[:] = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_uniform(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

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
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_root_from_terrain(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the robot to this pose. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_poses: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_poses is None:
        raise ValueError(
            "The randomization term 'reset_robot_root_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_poses.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_poses[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
    env: BaseEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_joints_by_offset(
    env: BaseEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # bias these values randomly
    joint_pos += sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class reset_joints_within_range(ManagerTermBase):
    """Reset an articulation's joints to a random position in the given ranges.

    This function samples random values for the joint position and velocities from the given ranges.
    The values are then set into the physics simulation.

    The parameters to the function are:

    * :attr:`position_range` - a dictionary of position ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`use_default_offset` - a boolean flag to indicate if the ranges are offset by the default joint state.
      Defaults to False.
    * :attr:`asset_cfg` - the configuration of the asset to reset. Defaults to the entity named "robot" in the scene.

    The dictionary values are a tuple of the form ``(min, max)``, where ``min`` and ``max`` are the minimum and
    maximum values. If the dictionary does not contain a key, the joint position or joint velocity is set to
    the default value for that joint. If the ``min`` or the ``max`` value is ``None``, the joint limits are used
    instead.
    """

    def __init__(self, cfg: RandomizationTermCfg, env: BaseEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # check if the cfg has the required parameters
        if "position_range" not in cfg.params or "velocity_range" not in cfg.params:
            raise ValueError(
                f"The term 'reset_joints_within_range' requires parameters: 'position_range' and 'velocity_range'."
                f" Received: {list(cfg.params.keys())}."
            )

        # parse the parameters
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        use_default_offset = cfg.params.get("use_default_offset", False)

        # extract the used quantities (to enable type-hinting)
        self._asset: Articulation = env.scene[asset_cfg.name]
        default_joint_pos = self._asset.data.default_joint_pos[0]
        default_joint_vel = self._asset.data.default_joint_vel[0]

        # create buffers to store the joint position and velocity ranges
        self._pos_ranges = self._asset.data.soft_joint_pos_limits[0].clone()
        self._vel_ranges = torch.stack(
            [-self._asset.data.soft_joint_vel_limits[0], self._asset.data.soft_joint_vel_limits[0]], dim=1
        )

        # parse joint position ranges
        pos_joint_ids = []
        for joint_name, joint_range in cfg.params["position_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            pos_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if joint_range[0] is not None:
                self._pos_ranges[joint_ids, 0] = joint_range[0] + use_default_offset * default_joint_pos[joint_ids]
            if joint_range[1] is not None:
                self._pos_ranges[joint_ids, 1] = joint_range[1] + use_default_offset * default_joint_pos[joint_ids]

        # store the joint pos ids (used later to sample the joint positions)
        self._pos_joint_ids = torch.tensor(pos_joint_ids, device=self._pos_ranges.device)
        # clamp sampling range to the joint position limits
        joint_pos_limits = self._asset.data.soft_joint_pos_limits[0]
        self._pos_ranges = self._pos_ranges.clamp(min=joint_pos_limits[:, 0], max=joint_pos_limits[:, 1])
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

        # parse joint velocity ranges
        vel_joint_ids = []
        for joint_name, joint_range in cfg.params["velocity_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            vel_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if joint_range[0] is not None:
                self._vel_ranges[joint_ids, 0] = joint_range[0] + use_default_offset * default_joint_vel[joint_ids]
            if joint_range[1] is not None:
                self._vel_ranges[joint_ids, 1] = joint_range[1] + use_default_offset * default_joint_vel[joint_ids]

        # store the joint vel ids (used later to sample the joint positions)
        self._vel_joint_ids = torch.tensor(vel_joint_ids, device=self._vel_ranges.device)
        # clamp sampling range to the joint velocity limits
        joint_vel_limits = self._asset.data.soft_joint_vel_limits[0]
        self._vel_ranges = self._vel_ranges.clamp(min=-joint_vel_limits[:, None], max=joint_vel_limits[:, None])
        self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    def __call__(
        self,
        env: BaseEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float | None, float | None]],
        velocity_range: dict[str, tuple[float | None, float | None]],
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        # get default joint state
        joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
        joint_vel = self._asset.data.default_joint_vel[env_ids].clone()

        # sample random joint positions for each joint
        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (len(env_ids), len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = sample_uniform(
                self._pos_ranges[:, 0], self._pos_ranges[:, 1], joint_pos_shape, device=joint_pos.device
            )
        # sample random joint velocities for each joint
        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = sample_uniform(
                self._vel_ranges[:, 0], self._vel_ranges[:, 1], joint_vel_shape, device=joint_vel.device
            )

        # set into the physics simulation
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_scene_to_default(env: BaseEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
