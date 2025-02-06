# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from trimesh.creation import cylinder

import omni.isaac.lab.sim as sim_utils
from hid import device
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.envs.mdp import UniformPose2dCommand, SupsiTerrainBasedPose2dCommand
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .pos_grace_env_cfg import PosGraceFlatEnvCfg, PosGraceRoughEnvCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, wrap_to_pi, quat_rotate_inverse, yaw_quat
from collections.abc import Sequence

from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import random
import omni.isaac.lab.utils.math as math_utils
import itertools

# Se usi pc 4 
from vacuum.LSTM_Helper import *

# Se usi pc 3 
# import sys
# sys.path.append("/home/amosca/IsaacLab/vacuum/")
# from LSTM_Helper import *


cnt = 0
cnt_tracktime = 0

import time
from functools import wraps

# Dizionario globale per registrare i tempi
execution_times = {}
call_counts = {}

def track_time(func):
    """Decoratore per tracciare il tempo di esecuzione di una funzione."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # Nome funzione
        func_name = func.__name__

        # Aggiorna tempi medi e conteggio
        if func_name not in execution_times:
            execution_times[func_name] = 0
            call_counts[func_name] = 0

        execution_times[func_name] += elapsed_time
        call_counts[func_name] += 1

        return result

    return wrapper

# Funzione per esportare i tempi medi
def export_execution_times(filename="execution_times.txt"):
    with open(filename, "w") as f:
        for func_name, total_time in execution_times.items():
            avg_time = total_time / call_counts[func_name]
            f.write(f"{func_name}: chiamate={call_counts[func_name]}, tempo medio={avg_time:.6f} s\n")

class GraceEnv(DirectRLEnv):
    cfg: PosGraceFlatEnvCfg | PosGraceRoughEnvCfg

    def __init__(self, cfg: PosGraceFlatEnvCfg | PosGraceRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands ------------------------------------------------------> da cambiare per POS
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # self._lstm_vacuum =
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position_tracking_xy",
                "heading_tracking_xy",
                "dof_vel_l2",
                "dof_torques_l2",
                "dof_vel_limit",
                "dof_torques_limit",
                "base_acc",
                "feet_acc",
                "action_rate_l2",
                "feet_contact_force",
                "dont_wait",
                "move_in_direction",
                "stand_at_target",
                "undesired_contacts",
                "stumble",
                "termination",
                "three_finger",
                "theta_marg_sum",
                # "a_marg"

            ]
        }
        # Get specific body indices
        self._cs_base_id, _ = self._contact_sensor.find_bodies("base")
        self._robot_base_id, _ = self._robot.find_bodies("base")

        self._cs_foot_ids_center = {'rl': self._contact_sensor.find_bodies("LR_FOOT_FINGER_00")[0],
                          'fr': self._contact_sensor.find_bodies("RF_FOOT_FINGER_00")[0],
                          'fl': self._contact_sensor.find_bodies("LF_FOOT_FINGER_00")[0],
                          'rr': self._contact_sensor.find_bodies("RR_FOOT_FINGER_00")[0]}

        self._robot_foot_ids_center = {'rl': self._robot.find_bodies("LR_FOOT_FINGER_00")[0],
                          'fr': self._robot.find_bodies("RF_FOOT_FINGER_00")[0],
                          'fl': self._robot.find_bodies("LF_FOOT_FINGER_00")[0],
                          'rr': self._robot.find_bodies("RR_FOOT_FINGER_00")[0]}

        self._cs_foot_ids = {
            'rl': self._contact_sensor.find_bodies(r"^(?!LR_FOOT_FINGER_00).*LR_FOOT_FINGER.*")[0],  # Excludes LR_FOOT_FINGER_00
            'fr': self._contact_sensor.find_bodies(r"^(?!RF_FOOT_FINGER_00).*RF_FOOT_FINGER.*")[0],  # Excludes RF_FOOT_FINGER_00
            'fl': self._contact_sensor.find_bodies(r"^(?!LF_FOOT_FINGER_00).*LF_FOOT_FINGER.*")[0],  # Excludes LF_FOOT_FINGER_00
            'rr': self._contact_sensor.find_bodies(r"^(?!RR_FOOT_FINGER_00).*RR_FOOT_FINGER.*")[0]  # Excludes RR_FOOT_FINGER_00
        }

        self._robot_foot_ids = {
            'rl': self._robot.find_bodies(r"^(?!LR_FOOT_FINGER_00).*LR_FOOT_FINGER.*")[0],  # Excludes LR_FOOT_FINGER_00
            'fr': self._robot.find_bodies(r"^(?!RF_FOOT_FINGER_00).*RF_FOOT_FINGER.*")[0],  # Excludes RF_FOOT_FINGER_00
            'fl': self._robot.find_bodies(r"^(?!LF_FOOT_FINGER_00).*LF_FOOT_FINGER.*")[0],  # Excludes LF_FOOT_FINGER_00
            'rr': self._robot.find_bodies(r"^(?!RR_FOOT_FINGER_00).*RR_FOOT_FINGER.*")[0]  # Excludes RR_FOOT_FINGER_00
        }


        self._cs_vacuum_ids = [self._cs_foot_ids[idx] for idx in self._cs_foot_ids.keys()]
        self._cs_vacuum_name = [idx for idx in self._cs_foot_ids.keys()]
        self._cs_vacuum_ids = list(itertools.chain.from_iterable(self._cs_vacuum_ids))


        self._robot_vacuum_ids = [self._robot_foot_ids[idx] for idx in self._robot_foot_ids.keys()]
        self._robot_vacuum_name = [idx for idx in self._robot_foot_ids.keys()]
        self._robot_vacuum_ids = list(itertools.chain.from_iterable(self._robot_vacuum_ids ))


        self._cs_id_acc_foot = self._cs_foot_ids_center
        self._robot_id_acc_foot = self._robot_foot_ids_center

        self._cs_foot_ids_center_list = [self._cs_foot_ids_center[idx] for idx in self._cs_foot_ids_center.keys()]
        self._cs_foot_ids_center_list = list(itertools.chain.from_iterable(self._cs_foot_ids_center_list))


        self._robot_foot_ids_center_list = [self._robot_foot_ids_center[idx] for idx in self._robot_foot_ids_center.keys()]
        self._robot_foot_ids_center_list = list(itertools.chain.from_iterable(self._robot_foot_ids_center_list ))

        # zero_force_finger = torch.tensor(self.num_envs, 3)
        # self._vacuum_force = {  "rl": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        #                         "fl": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        #                         "rr": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        #                         "fr": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        # }


        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")

        self._min_finger_contacts = 3

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*HFE", ".*KFE"])
        self._all_joints, _ = self._robot.find_joints(['^(?!.*(_FOOT|ankle).*).*$'])


        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        self.error_pos = torch.zeros(self.num_envs, device=self.device)
        self.error_pos_xy = torch.zeros(self.num_envs, device=self.device)

        self.error_heading = torch.zeros(self.num_envs, device=self.device)
        self.remaining_time = torch.zeros(self.num_envs, device=self.device)

        self.joint_vel_limit = torch.zeros((self.num_envs,len(self._all_joints)), device=self.device)
        self.joint_effort_limit = torch.zeros_like(self.joint_vel_limit )

        self.tot_mass = self._robot.data.default_mass.sum(dim=1).unsqueeze(-1).to(device=self.device)

        self.pos_foot_w = dict()
        self.foot_in_contact = dict()
        self.force_w = dict()
        self.n_gab_w = {}
        self.check_face = {
            "fl-fr": ["rr", "rl"],
            "fr-rr": ["rl", "fl"],
            "rr-rl": ["fl", "fr"],
            "rl-fl": ["fr", "rr"],
            "fl-rr": ["fr", "rl"],
            "fr-rl": ["fl", "rr"]
        }
        self.foot_faces = {
            "fl-fr": ["fl", "fr"],
            "fr-rr": ["fr", "rr"],
            "rr-rl": ["rr", "rl"],
            "rl-fl": ["rl", "fl"],
            "fl-rr": ["fl", "rr"],
            "fr-rl": ["fr", "rl"]
        }
        self.mass_times_agilim_dot_n_agab_w = {}
        self.theta_marg = {}
        self.a_marg = {}
        self.is_inside_poly = {}

        self._amarg = 0
        self._thetamarg = 0
        self._sumthetamarg = 0

        self.a_gilim_w = 0

        for act in self._robot.actuators.keys():
            self.joint_vel_limit[:,self._robot.actuators[act]._joint_indices] = self._robot.actuators[act].velocity_limit
            self.joint_effort_limit[:, self._robot.actuators[act]._joint_indices] = self._robot.actuators[act].effort_limit

        #definizione degli attributi per le vacuum force
        self._num_bodies_vacuum = len(self._cs_vacuum_ids)
        self._forces_vacuum = torch.zeros((self.num_envs,  self._num_bodies_vacuum, 3), device=self.device)
        self._torques_vacuum = torch.zeros((self.num_envs,  self._num_bodies_vacuum, 3), device=self.device)

        self._lstm_vacuum = LSTM_Helper()
        self._vacuum_time = None
        self._vacuum_old = None
        # self._robot.set_external_force_and_torque(self._forces_vacuum, self._torques_vacuum, env_ids=torch.arange(self.num_envs, device=self.device), body_ids=self._vacuum_ids)

    # @track_time
    def pose_command(self) -> torch.Tensor:
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    # @track_time
    def _update_pose_metrics(self):
        self.error_pos = torch.norm(self.pos_command_w - self._robot.data.root_pos_w, dim=1)
        self.error_pos_xy = torch.norm(self.pos_command_w[:,:2] - self._robot.data.root_pos_w[:,:2] , dim=1)
        self.error_heading = torch.abs(wrap_to_pi(self.heading_command_w - self._robot.data.heading_w))

    # @track_time
    def _resample_pose_command(self, env_ids: Sequence[int]):
        if cnt == 0:
            return
        # obtain env origins for the environments

        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self.pos_command_w[env_ids] = default_root_state[:, :3]
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)

        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.pose_command.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.pose_command.ranges.pos_y)

        #setto il commando nuovo per visualizzazione
        self._pos_command_visualizer.pos_command_w = self.pos_command_w

        # self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2] #da mettere altezza qui

        if (self.cfg.pose_command.simple_heading):
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self._robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self._robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self._robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.pose_command.ranges.heading)
        self._pos_command_visualizer.heading_command_w[env_ids] = self.heading_command_w[env_ids]

    # @track_time
    def _resample_command_terrain_based(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)

        self.pos_command_w[env_ids] = self.valid_targets[
            self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids], ids
        ]
        # offset the position command by the current root height
        self.pos_command_w[env_ids, 2] += self._robot.data.default_root_state[env_ids, 2]/1.8

        if self.cfg.pose_command.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self._robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self._robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self._robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.pose_command.ranges.heading)

        self._pos_command_visualizer.pos_command_w[env_ids] = self.pos_command_w[env_ids]
        self._pos_command_visualizer.heading_command_w[env_ids] = self.heading_command_w[env_ids]

    # @track_time
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        if isinstance(self.cfg, PosGraceRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
            self._pos_command_visualizer = SupsiTerrainBasedPose2dCommand(self.cfg.pose_command, self, self._terrain )
        elif isinstance(self.cfg, PosGraceFlatEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._pos_command_visualizer = SupsiTerrainBasedPose2dCommand(self.cfg.pose_command, self, self._terrain )

        self._vacuum_visualizer = VisualizationMarkers(self.cfg.vacuum_visualizer)

        if self.cfg.show_flat_patches:
            # Configure the flat patches
            vis_cfg = VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers={})
            for name in self._terrain.flat_patches:
                vis_cfg.markers[name] = sim_utils.CylinderCfg(
                    radius=0.5,  # note: manually set to the patch radius for visualization
                    height=0.1,
                    visual_material=sim_utils.GlassMdlCfg(glass_color=(random.random(), random.random(), random.random())),
                )
            flat_patches_visualizer = VisualizationMarkers(vis_cfg)

            # Visualize the flat patches
            all_patch_locations = []
            all_patch_indices = []
            for i, patch_locations in enumerate(self._terrain.flat_patches.values()):
                num_patch_locations = patch_locations.view(-1, 3).shape[0]
                # store the patch locations and indices
                all_patch_locations.append(patch_locations.view(-1, 3))
                all_patch_indices += [i] * num_patch_locations
            # combine the patch locations and indices
            flat_patches_visualizer.visualize(torch.cat(all_patch_locations), marker_indices=all_patch_indices)

        if "target" not in self._terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self._terrain.flat_patches.keys())}"
            )
        self.valid_targets: torch.Tensor = self._terrain.flat_patches["target"]

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # @track_time
    def _pre_physics_step(self, actions: torch.Tensor):
        global cnt_tracktime
        # cnt_tracktime += 1
        # print(cnt_tracktime)
        # if cnt_tracktime == 48*5: #48 each epoch? iteration*decimation*steps?
        #     import sys
        #     export_execution_times(filename="/home/lab/IsaacLab/execution_times.txt")
        #     sys.exit()

        self._actions = actions.clone()

        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:,self._all_joints]

        # self._actions_pos = self._actions[:,:-4*3]
        # self._processed_actions_pos = self.cfg.action_scale * self._actions_pos + self._robot.data.default_joint_pos[:, self._all_joints]


        # self._action_vacuum = self._actions[:,-4*3:]
        # self._processed_action_vacuum = self.cfg.action_scale * self._action_vacuum
        # self._processed_action_vacuum = torch.abs(self._processed_action_vacuum )
        # self._processed_action_vacuum = torch.clamp(self._processed_action_vacuum,min=0.,max=1.)
        # self._processed_action_vacuum = torch.where(self._processed_action_vacuum<3/5, 0., self._processed_action_vacuum) # voltage
        # contact_time = self._contact_sensor.data.current_contact_time[:, self._vacuum_ids]

        ##vedo le forze di rezione nel W
        # self._finger_reaction_forces_w = self._contact_sensor.data.net_forces_w[:, self._vacuum_ids]
        # #converto forze nel body piedi
        # self._finger_reaction_forces_b  = quat_rotate_inverse(self._robot.data.body_quat_w[:,self._vacuum_ids], self._finger_reaction_forces_w )
        # #verifico che sono all interno del cono del giunto sferico
        # spherical_joint_limit = 20.0
        # theta_xz_w = torch.atan2(self._finger_reaction_forces_w[:,:,0],self._finger_reaction_forces_w[:,:,2])*180.0/torch.pi
        # theta_yz_w = torch.atan2(self._finger_reaction_forces_w[:,:,1],self._finger_reaction_forces_w[:,:,2])*180.0/torch.pi
        # theta_xz = torch.atan2(self._finger_reaction_forces_b[:,:,0],self._finger_reaction_forces_b[:,:,2])*180.0/torch.pi
        # theta_yz = torch.atan2(self._finger_reaction_forces_b[:,:,1],self._finger_reaction_forces_b[:,:,2])*180.0/torch.pi
        #
        # mask_xz = theta_xz < spherical_joint_limit
        # mask_yz = theta_yz < spherical_joint_limit
        #
        # self._mask_inside_joint_limit = torch.logical_and(mask_xz, mask_yz)
        #
        # contact_time[torch.logical_not(self._mask_inside_joint_limit)] = 0.
        ## FINE

        # if self._vacuum_time is None:
        #     self._vacuum_time = contact_time
        # if self._vacuum_old is None:
        #     self._vacuum_old = contact_time
        #
        # mask = torch.logical_and(self._processed_action_vacuum>0., contact_time>0.)
        # self._vacuum_old = torch.where(mask,self._vacuum_old, contact_time)
        # self._vacuum_time = contact_time - self._vacuum_old
        # self._forces_vacuum = torch.zeros_like(self._forces_vacuum, device=self.device)
        # self._forces_vacuum[:, :, 2][mask] = -self._lstm_vacuum.predict(self._vacuum_time, self._processed_action_vacuum)[mask]
        #
        #
        # if self.sim.has_gui():
        #     scales = torch.ones_like(self._forces_vacuum, device=self.device)
        #     scales[:, :, 2][mask] = self._forces_vacuum[:, :, 2][mask] / 380 # 380 --> max force from LSTM
        #     translations = self._robot.data.body_pos_w[:, self._vacuum_ids, :]
        #     translations[:, :, 2][torch.logical_not(mask)] += self.cfg.vacuum_visualizer.markers["cylinder_no_contact"].height / 2
        #     translations[:, :, 2][mask] += -scales[:, :, 2][mask] * self.cfg.vacuum_visualizer.markers["cylinder_no_contact"].height / 2
        #     scales = scales.reshape((-1, 3))
        #     translations = translations.reshape((-1, 3))
        #
        #     no_contact_mask = (contact_time==0).flatten()
        #     contact_mask = torch.logical_and(contact_time>0, mask==False).flatten()
        #     vacuum_mask = mask.flatten()
        #     vacuum_indices = torch.ones_like(vacuum_mask, device=self.device).int()
        #     vacuum_indices[no_contact_mask] = 0
        #     vacuum_indices[contact_mask] = 1
        #     vacuum_indices[vacuum_mask] = 2
        #
        #     self._vacuum_visualizer.visualize(translations=translations, scales=scales, marker_indices=vacuum_indices)

    # @track_time
    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, self._all_joints)

        # self._robot.set_joint_position_target(self._processed_actions_pos, self._all_joints)
        # self._robot.set_external_force_and_torque(self._forces_vacuum, self._torques_vacuum, env_ids=torch.arange(self.num_envs, device=self.device), body_ids=self._vacuum_ids)
        # applico forza su piede se a contatto  GUARDA METODO IN ARTICULATION root_physx_view

    # @track_time
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, PosGraceRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)

        self._update_pose_command()

        # obs = torch.cat(
        #     [
        #         tensor
        #         for tensor in (
        #             self._robot.data.root_lin_vel_b, #3
        #             self._robot.data.root_ang_vel_b, #3
        #             self._robot.data.projected_gravity_b, #3
        #             self.pose_command(), #3
        #             self._robot.data.joint_pos[:,self._all_joints] - self._robot.data.default_joint_pos[:,self._all_joints], #12
        #             self._robot.data.joint_vel[:,self._all_joints], #12
        #             height_data,#187
        #             self._actions,#12
        #         )
        #         if tensor is not None
        #     ],
        #     dim=-1,
        # )
        #
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b, #3
                    self._robot.data.root_ang_vel_b, #3
                    self._robot.data.projected_gravity_b, #3
                    self._robot.data.joint_pos[:,self._all_joints], #12
                    self._robot.data.joint_vel[:,self._all_joints], #12
                    self.pose_command(),  # 4
                    self._remaining_time(),  # 1
                    height_data,#187 -->196
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    # @track_time
    def _compute_foot_contact(self, contact_sensor, step_dt, foot_ids, min_contacts=2):
        """
        Calcola se ciascun piede è in contatto in base a un numero minimo di punti di contatto attivi.

        Args:
        - contact_sensor: il sensore di contatto che fornisce informazioni sul primo contatto e sul tempo in aria.
        - step_dt: intervallo di tempo tra i passi della simulazione.
        - foot_ids: dizionario con ID delle dita per ciascun piede. Es: {'lr': [0,1,2], 'rf': [3,4,5], ...}
        - min_contacts: numero minimo di punti di contatto per considerare un piede "in contatto".

        Returns:
        - first_contacts: dizionario con `first_contact` per ciascun piede.
        - last_air_times: dizionario con `last_air_time` per ciascun piede.
        """

        first_contacts = {}
        last_air_times = {}

        for foot, toe_ids in foot_ids.items():
            # Calcola il primo contatto per ciascun dito del piede
            first_contact_per_toe = contact_sensor.compute_first_contact(step_dt)[:, toe_ids]  # [n_envs, num_toes]

            # Conta il numero di dita in contatto per ciascun piede e verifica se supera il minimo richiesto
            first_contacts[foot] = (torch.sum(first_contact_per_toe, dim=1) >= min_contacts).float()  # [n_envs]

            # Calcola l'ultimo tempo in aria tra le dita del piede (massimo tra le dita)
            last_air_time_per_toe = contact_sensor.data.last_air_time[:, toe_ids]  # [n_envs, num_toes]
            last_air_times[foot] = torch.max(last_air_time_per_toe, dim=1).values  # [n_envs]

        return first_contacts, last_air_times

    # @track_time
    def _remaining_time(self):
        self.remaining_time = self.max_episode_length_s - (self.episode_length_buf * (self.cfg.sim.dt * self.cfg.decimation)).squeeze(dim=-1)

    # @track_time
    def safe_normalize(self, vectors, epsilon=1e-6):
        norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
        return vectors / (norms + epsilon)

    # @track_time
    def compute_foot_properties(self, name):

        #SE VUOI USARE VERSIONE SENZA CENTRO
        # pos_fingers = self._robot.data.body_pos_w[:, self._foot_ids[name], :]
        # self.pos_foot_w[name] = pos_fingers.mean(dim=1)  # Media delle posizioni delle dita
        # self.foot_in_contact[name] = self._contact_sensor.data.current_contact_time[:, self._foot_ids[name]].sum(dim=1) > 0

        #SE VUOI USAE VERSIONE CON CENTRO
        self.pos_foot_w[name] = self._robot.data.body_pos_w[:, self._robot_foot_ids_center[name], :].squeeze()
        self.foot_in_contact[name] = self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center[name]] > 1.

        # #ordinate _forces_vacuum in accordo a vacuum_ids e vacuum_names
        vacuum = torch.zeros_like(self._forces_vacuum[:, :3, :], device=self.device)
        if name in "rl":
            vacuum = self._forces_vacuum[:, :3, :]#[17,18,19]
        if name in "fr":
            vacuum = self._forces_vacuum[:,3:6,:] #[26,27,28]
        if name in "fl":
            vacuum = self._forces_vacuum[:,6:9,:] #[23,24,25]
        if name in "rr":
            vacuum = self._forces_vacuum[:,9:12,:] #[20,21,22]

        """gripping force into tumble stability, it can be considered a force to resist an external tearing-off force at the contact point of the gripper"""
        # Fj_b = math_utils.quat_rotate(self._robot.data.body_quat_w[:, self._robot_foot_ids[name]], vacuum)

        # SOTTO IPOTESI: che l xform delle dita dei piedi abbia la z rivolta verso l'alto. La variabile vacuum contiene gia le forze di reazione dovute alla vaccum
        # quindi: quando ho adesione della ventosa la forza di reazione delle vacuum sara entrante nel punto di appoggio (-)
        Fj_b = vacuum

        """reaction force"""
        forces_foot_w = self._contact_sensor.data.net_forces_w[:, self._cs_foot_ids[name], :] #PERCHE SONO QUELLE RILEVATE DAL SENSORE --> REAZIONE. sono forze uscenti dal terreno (+)

        # verifico dove sto usando la forza di vacuum
        mask = torch.norm(Fj_b, dim = -1)>1.

        #Esprimo le forze delle ventose nel frame W. NB le ventose sono espresse nel frame relativo all xform delle dita della zampa
        Fj_w = math_utils.quat_rotate(self._robot.data.body_quat_w[:, self._robot_foot_ids[name]], Fj_b)

        #Inserisco solo forze di reazione delle ventose che sono in contatto nel vettore delle forze di reazione.
        #NB le forces_foot_w teoricamente non devono essere usate perche contengono sia le forze di reazione date dal puro contatto che quelle dovute al vacuum.
        #in accordo con https://doi.org/10.13180/clawar.2020.24-26.08.18 SI DOVREBBE TENERE CONTO DELLE SOLE FORZE DI VACUUM
        forces_foot_w[mask] = Fj_w[mask]

        #Salvo in force_w la somma dele forze di rezione  per usarle successivamente nel calcolo delle metriche di stabilita
        # self.force_w[name] = forces_foot_w.sum(dim=1)
        self.force_w[name] = Fj_w.sum(dim=1)



    # @track_time
    def _theta_marg_and_a_marg(self):
        # Gravito-inertial acceleration
        acc_mass_w  = self._robot.data.body_lin_acc_w * self._robot.data.default_mass.unsqueeze(-1).to(device=self.device)
        ag_total_w  = acc_mass_w.sum(dim=1) / self.tot_mass
        self.a_gi_w = (self._robot.data.GRAVITY_VEC_W *  9.81) - ag_total_w

        # Center of mass
        self.com_w = torch.sum(
            self._robot.data.body_pos_w * self._robot.data.default_mass.unsqueeze(-1).to(device=self.device), dim=1
        ) / self.tot_mass

        # Foot properties and contacts
        for name in self._cs_foot_ids.keys():
            self.compute_foot_properties(name)

        # Cross products for tumbling axes --> Ottieni i versori che definiscono i lati del poliedro di stabilita
        self.n_gab_w = {
            "fl-fr": torch.cross(self.com_w - self.pos_foot_w["fl"], self.com_w - self.pos_foot_w["fr"], dim=1),
            "fr-rr": torch.cross(self.com_w - self.pos_foot_w["fr"], self.com_w - self.pos_foot_w["rr"], dim=1),
            "rr-rl": torch.cross(self.com_w - self.pos_foot_w["rr"], self.com_w - self.pos_foot_w["rl"], dim=1),
            "rl-fl": torch.cross(self.com_w - self.pos_foot_w["rl"], self.com_w - self.pos_foot_w["fl"], dim=1),
            "fl-rr": torch.cross(self.com_w - self.pos_foot_w["fl"], self.com_w - self.pos_foot_w["rr"], dim=1),
            "fr-rl": torch.cross(self.com_w - self.pos_foot_w["fr"], self.com_w - self.pos_foot_w["rl"], dim=1),
        }

        # A four legged robot has six possible tumbling axes. This function simply computes a bitmap if either the axis is active (leg is in contact) or not
        self.bitmap_contatc = {
                "fl": self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fl"]].sum(dim=1) > 0,
                "fr": self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fr"]].sum(dim=1) > 0,
                "rr": self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rr"]].sum(dim=1) > 0,
                "rl": self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rl"]].sum(dim=1) > 0,
                "fl-fr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fl"]].sum(dim=1) > 0, self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fr"]].sum(dim=1) > 0),
                "fr-rr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fr"]].sum(dim=1) > 0, self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rr"]].sum(dim=1) > 0),
                "rr-rl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rr"]].sum(dim=1) > 0, self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rl"]].sum(dim=1) > 0),
                "rl-fl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rl"]].sum(dim=1) > 0, self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fl"]].sum(dim=1) > 0),
                "fl-rr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fl"]].sum(dim=1) > 0, self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rr"]].sum(dim=1) > 0),
                "fr-rl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["fr"]].sum(dim=1) > 0, self._contact_sensor.data.current_contact_time[:, self._cs_foot_ids_center["rl"]].sum(dim=1) > 0),
        }

        #Tensore costruito dal dizionario bitmap con uno specifico ordine che viene utilizzato per calcolare le metriche
        is_active           = torch.stack([self.bitmap_contatc["fl-fr"], self.bitmap_contatc["fr-rr"], self.bitmap_contatc["rr-rl"], self.bitmap_contatc["rl-fl"], self.bitmap_contatc["fl-rr"], self.bitmap_contatc["fr-rl"]], dim=0)

        # Normalize tumbling axis vectors
        for key, value in self.n_gab_w.items():
            self.n_gab_w[key] = self.safe_normalize(value)

        # Compute mass_times_agilim_dot_n_agab vedi Eq.(5) di https://doi.org/10.13180/clawar.2020.24-26.08.18
        for key, value in self.foot_faces.items():
            foot_j1, foot_j2 = self.check_face[key][0], self.check_face[key][1]
            foot_a,  foot_b  = self.foot_faces[key][0], self.foot_faces[key][1]
            temp_j1 = torch.cross(self.pos_foot_w[foot_b] - self.pos_foot_w[foot_j1], self.pos_foot_w[foot_a] - self.pos_foot_w[foot_j1], dim=1) #cross dopo = dell'eq5 per j=1
            temp_j2 = torch.cross(self.pos_foot_w[foot_b] - self.pos_foot_w[foot_j2], self.pos_foot_w[foot_a] - self.pos_foot_w[foot_j2], dim=1) #cross dopo = dell'eq5 per j=2
            #NB self.force_w[foot_j1] contiene la forza di reazione dovuta al vacuum, non ci sono M0 e F0 (M0 and F0 are external components of the tumbling moment)
            self.mass_times_agilim_dot_n_agab_w[key] = torch.sum(self.force_w[foot_j1] * temp_j1, dim=1) + torch.sum(self.force_w[foot_j2] * temp_j2, dim=1) # parte dopo = del Eq.5 https://doi.org/10.13180/clawar.2020.24-26.08.18

        # Devo trovare la a_{gi,lim} risolvendo m*a_{gi,lim} \cdot n_{gab} = mass_times_agilim_dot_n_agab_w. NB: e' un sistema composto da 6 equzioni
        for key in self.n_gab_w.keys():
            #metto a 0 le n_{gab} che non sono attive (i.e. se lato del poliedro non e' attivo significa che uno o entrambi i piedi che definiscono il lato non sono a contatto)
            mask = torch.logical_not(self.bitmap_contatc[key])
            self.n_gab_w[key][mask] = torch.zeros((3), device=self.device)
            self.mass_times_agilim_dot_n_agab_w[key][mask] = 0.

        #CONVERTO IL SISTEMA IN MATRICI
        #A*x = b dove A e' una matrice di 6x3, x e' una 3x1 e b e' 6x1.
        A = torch.stack([self.n_gab_w[key] for key in self.n_gab_w.keys()], dim=1).to(self.device)
        b = torch.stack([self.mass_times_agilim_dot_n_agab_w[key] for key in self.n_gab_w.keys()], dim=1).to(self.device)
        b = b.unsqueeze(2)  # (num_envs, num_faces, 1)

        if A.shape[1] >= 3:  # Assicura che ci siano almeno 3 vincoli
            # A^{#} * A * x = A^{#} * b --> x = A^{#} * b  dove x e' la a_{gi,lim}
            A_pseudo_inv = torch.linalg.pinv(A)
            self.a_gilim_w = torch.matmul(A_pseudo_inv, b).squeeze(-1)
        else:
            raise ValueError("Numero insufficiente di vincoli per calcolare a_{gi,lim}.")


        rew_eth = torch.zeros(self.num_envs, device=self.device)
        rew_amarg = torch.zeros(self.num_envs, device=self.device)
        epsilon = 1e-8

        for key in self.foot_faces.keys():

            #versore del lato del poliedro
            n_agb = self.n_gab_w[key]

            # Calcolo delle norme con aggiunta di epsilon per evitare divisioni per zero
            norm_n_agb = torch.linalg.norm(n_agb, dim=1) + epsilon
            norm_a_gi_w = torch.linalg.norm(self.a_gi_w, dim=1) + epsilon
            norm_a_gilim_w = torch.linalg.norm(self.a_gilim_w, dim=1) + epsilon

            # Calcolo di cos_theta_agi e cos_theta_gilim con denominatore corretto
            # Our first proposition for quantitative analysis is the inclination margin for gravito-inertial acceleration, which is the angle between the GIA vector and the limit plane for a tumbling
            # axis. The minimum value among all tumbling axes is the inclination margin θ_{marg} vedi Eq.(7) https://doi.org/10.13180/clawar.2020.24-26.08.18
            cos_theta_agi = torch.clip(torch.sum(n_agb * self.a_gi_w, dim=1) / (norm_n_agb * norm_a_gi_w), -1.0, 1.0)
            cos_theta_gilim = torch.clip(norm_a_gilim_w / norm_a_gi_w, -1.0, 1.0)

            # Calcolo di theta_marg per ogni lato. The value is normalized with  − π/2 to ensure negative angle if the GIA vector points out of the polyhedron. Eq.2 DOI: 10.1109/IROS55552.2023.10341665
            self.theta_marg[key] = (torch.arccos(cos_theta_agi) - torch.arccos(cos_theta_gilim)) - torch.pi / 2

            # Calcolo di a_marg per ogni lato. The acceleration margin represents the maximum acceleration increment that can be applied in any direction that does not cause the robot to tumble. Eq 8 https://doi.org/10.13180/clawar.2020.24-26.08.18
            self.a_marg[key] = norm_a_gilim_w - torch.sum(n_agb * self.a_gi_w, dim=1) / norm_n_agb

            # Calcolo di is_inside_poly
            # a destra del <= puo essere 0 se non usi le vacuum oppuere devi verificare con la a_{gi,lim}
            self.is_inside_poly[key] = torch.sum(n_agb * self.a_gi_w, dim=1) <= torch.sum(n_agb * self.a_gilim_w, dim=1)


            # Aggiornamento di rew_eth e rew_amarg
            rew_eth[:] += self.theta_marg[key]
            rew_amarg[:] += self.a_marg[key]

        a_marg_stack        = torch.stack([self.a_marg["fl-fr"], self.a_marg["fr-rr"], self.a_marg["rr-rl"], self.a_marg["rl-fl"], self.a_marg["fl-rr"], self.a_marg["fr-rl"]], dim=0)
        theta_marg_stack    = torch.stack([self.theta_marg["fl-fr"], self.theta_marg["fr-rr"], self.theta_marg["rr-rl"], self.theta_marg["rl-fl"], self.theta_marg["fl-rr"], self.theta_marg["fr-rl"]], dim=0)
        is_in_poly          = torch.stack([self.is_inside_poly["fl-fr"], self.is_inside_poly["fr-rr"], self.is_inside_poly["rr-rl"], self.is_inside_poly["rl-fl"], self.is_inside_poly["fl-rr"], self.is_inside_poly["fr-rl"]], dim=0)

        mask_active_in_poly = is_in_poly * is_active # se lato stabile e se lato attivo (tutti e due piedi a contatto)
        zeros = torch.zeros(self.num_envs, device=self.device)

        # CONSIDERO LE METRICHE SOLO SE: at least three legs are contacting the ground and the respective GIA vector points inside the stability polyhedron ALTRIMENTI 0
        amin        = torch.where(mask_active_in_poly.sum(dim=0) >= 3, a_marg_stack.min(dim=0).values, 0)
        theta_min   = torch.where(mask_active_in_poly.sum(dim=0) >= 3, theta_marg_stack.min(dim=0).values, 0)

        # #IN ACCORDO CON THESIS CEWEILBEL
        self._amarg         = torch.max(zeros, amin).to(device=self.device)
        # #IN ACCORDO ARTICOLO VALSECCHI
        self._sumthetamarg  = theta_marg_stack.sum(dim=0).to(device=self.device)


    def get_amarg(self):
        return self._amarg
    def get_thetamarg(self):
        return self._thetamarg
    def get_sumthetamarg(self):
        return self._sumthetamarg

    # @track_time
    def _get_rewards(self) -> torch.Tensor:
        cnt = 1
        #compute remaining time
        self._remaining_time()

        #XY-Position Tracking
        self._update_pose_metrics()

        self._theta_marg_and_a_marg()

        position_tracking_mapped = torch.where(self.remaining_time < 1, (1 - 0.5 * self.error_pos_xy), 0.0)
        # Heading Tracking
        heading_tracking_mapped = torch.where(self.remaining_time < 1, (1 - 0.5 * self.error_heading), 0.0)
        # joint velocity
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel[:,self._all_joints]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque[:,self._all_joints]), dim=1)
        # Joint velocity limit
        joint_vel_limit = torch.sum(torch.clamp(torch.abs(self._robot.data.joint_vel[:,self._all_joints])-self.joint_vel_limit,min=0), dim=1)
        # Torque limit
        joint_eff_limit = torch.sum(torch.clamp(torch.abs(self._robot.data.applied_torque[:,self._all_joints])-self.joint_effort_limit,min=0), dim=1)
        # Base acc
        base_acc = (self.cfg.base_lin_acc_weight * torch.square(torch.norm(self._robot.data.body_lin_acc_w[:, self._robot_base_id, :], dim=-1)) +
                    self.cfg.base_ang_acc_weight * torch.square(torch.norm(self._robot.data.body_ang_acc_w[:, self._robot_base_id, :], dim=-1))).squeeze(dim=1)
        # Feet acc and Feet Force
        feet_acc    = torch.zeros(self.num_envs, device=self.device)
        feet_force  = torch.zeros(self.num_envs, self._contact_sensor.data.net_forces_w_history.shape[1], device=self.device)
        stumble     = torch.zeros(self.num_envs, device=self.device)
        combined_mask = torch.zeros(self.num_envs, device=self.device)
        norm_feet_force_dict = dict()
        # good_foot = torch.zeros(self.num_envs, device=self.device)
        good_foot = torch.ones(self.num_envs, device=self.device) *- 1 / 3 * 12.

        for foot in self._robot_id_acc_foot.keys():
            #FEET ACC
            feet_acc    = feet_acc + torch.norm(self._robot.data.body_lin_acc_w[:, self._robot_id_acc_foot[foot], :], dim=-1).squeeze(dim=-1)
            #CONTACT FORCE
            norm_feet_force_dict[foot] = torch.norm(torch.sum(self._contact_sensor.data.net_forces_w_history[:, :, self._cs_foot_ids_center[foot]], dim=2), dim=-1)
            feet_force  = feet_force + torch.clamp(norm_feet_force_dict[foot] - self.cfg.max_feet_contact_force, min=0)** 2
            #STUMBLE
            net_forces_w = self._contact_sensor.data.net_forces_w[:, self._cs_foot_ids_center[foot], :]
            net_forces_b = quat_rotate_inverse(self._robot.data.body_quat_w[:,self._robot_foot_ids_center[foot]], net_forces_w)
            fxy = torch.norm(net_forces_b[:,:, :2], dim=-1)
            fz = torch.norm(net_forces_b[:, :, 2:], dim=-1)

            stumble = stumble + torch.sum(torch.where(fxy>2*fz,1,0),dim=-1)
            #TERMINATION FEET CONTACT
            combined_mask = torch.logical_or(torch.max(norm_feet_force_dict[foot], dim=1)[0] > self.cfg.feet_termination_force,combined_mask)

            #GOOD FOOT 3
            fz_mask = torch.norm(net_forces_b[:, :, 2:], dim=-1) > 1.
            n_finger_in_contact = fz_mask.float().sum(dim=1)
            good_foot = good_foot + 1/3 * n_finger_in_contact

            # #
            # mask_contact_no_three = torch.logical_and(fz_mask.float().sum(dim=1) >=1, fz_mask.float().sum(dim=1) <3)
            # penalty = -4 #(fz_mask.float().sum(dim=1)-3)
            # good_foot = torch.where(mask_contact_no_three,good_foot+penalty,good_foot+0.)
            #
            # mask_contact_three = fz_mask.float().sum(dim=1) == 3
            # good_foot = torch.where(mask_contact_three, good_foot+1., good_foot+0.)



        # if torch.any(feet_force>0.):
        #     print("feet_force>0")
        # if torch.any(combined_mask):
        #     print("combined_mask>0")
        feet_force = torch.max(feet_force, dim=-1)[0]

            # Action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # Don't wait
        dont_wait = torch.where(torch.norm(self._robot.data.root_lin_vel_b, dim=-1) < self.cfg.wait_time, 1., 0.)
        # Move in direction
        target_vec = self.pos_command_w  - self._robot.data.root_pos_w
        move_in_direction = torch.sum(self._robot.data.root_lin_vel_b * target_vec, dim=-1) / (torch.norm(self._robot.data.root_lin_vel_b, dim=-1) * torch.norm(target_vec, dim=-1)  + 1e-6)
        # Stand at target
        mask = torch.logical_and(torch.where(self.error_pos_xy <self.cfg.stand_min_dist,1,0),torch.where(self.error_heading < self.cfg.stand_min_ang, 1, 0))
        stand_at_target = torch.where(mask, torch.norm(self._robot.data.default_joint_pos[:,self._all_joints] - self._robot.data.joint_pos[:,self._all_joints], dim=-1),0)
        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # Termination
        mask_base_collision = torch.max(torch.norm(net_contact_forces[:, :, self._cs_base_id], dim=-1), dim=1)[0] > 1.0
        termination = torch.where(mask_base_collision.squeeze() | combined_mask, 1, 0)

        theta_marg_sum = self.get_sumthetamarg()

        a_marg = self.get_amarg()

        mask_moving = torch.norm(self._robot.data.root_lin_vel_b, dim=-1) >= 0.2
        net_forces_w = self._contact_sensor.data.net_forces_w[:, self._cs_vacuum_ids, :]
        net_forces_b = quat_rotate_inverse(self._robot.data.body_quat_w[:, self._robot_vacuum_ids], net_forces_w)
        fz = torch.norm(net_forces_b[:, :, 2:], dim=-1)
        fz_mask = torch.norm(net_forces_b[:, :, 2:], dim=-1) > 1.
        n_finger_in_contact = fz_mask.float().sum(dim=1)
        #
        mask_not_moving_and_no_four_contact_feet = torch.logical_and(torch.norm(self._robot.data.root_lin_vel_b, dim=-1) < 0.2, n_finger_in_contact  != 12)
        if torch.any(mask_not_moving_and_no_four_contact_feet):
            pippo = 1
        three_finger = good_foot*mask_moving.float()
        three_finger[mask_not_moving_and_no_four_contact_feet] = -1.

        air_time = -self._contact_sensor._data.current_air_time[:, self._cs_foot_ids_center_list].sum(dim=-1)
        std = 0.25

        norm_airtime = torch.abs(air_time)  # Euclidean norm (default)
        square_airtime = air_time**2

        # Normed Exponential Kernel: exp(-||x|| / std^2)
        epsilon = 1e-8  # Small value to prevent artifacts
        normed_exponential = -torch.exp(-torch.clamp(norm_airtime, min=epsilon) / (std ** 2))

        # Squared Exponential Kernel: exp(-||x||^2 / (2 * std^2))
        squared_exponential = -torch.exp(-(norm_airtime ** 2) / (2 * std ** 2))



        rewards = {
            "position_tracking_xy":     position_tracking_mapped    * self.cfg.position_tracking_reward_scale   * self.step_dt,
            "heading_tracking_xy":      heading_tracking_mapped     * self.cfg.heading_tracking_reward_scale    * self.step_dt,
            "dof_vel_l2":               joint_vel                   * self.cfg.joint_vel_reward_scale           * self.step_dt,
            "dof_torques_l2":           joint_torques               * self.cfg.joint_torque_reward_scale        * self.step_dt,
            "dof_vel_limit":            joint_vel_limit             * self.cfg.joint_vel_limit_reward_scale     * self.step_dt,
            "dof_torques_limit":        joint_eff_limit             * self.cfg.joint_torque_limit_reward_scale  * self.step_dt,
            "base_acc":                 base_acc                    * self.cfg.base_acc_reward_scale            * self.step_dt,
            "feet_acc":                 feet_acc                    * self.cfg.feet_acc_reward_scale            * self.step_dt,
            "action_rate_l2":           action_rate                 * self.cfg.action_rate_reward_scale         * self.step_dt,
            "feet_contact_force":       feet_force                  * self.cfg.feet_contact_force_reward_scale  * self.step_dt,
            "dont_wait":                dont_wait                   * self.cfg.dont_wait_reward_scale           * self.step_dt,
            "move_in_direction":        move_in_direction           * self.cfg.move_in_direction_reward_scale   * self.step_dt,
            "stand_at_target":          stand_at_target             * self.cfg.stand_at_target_reward_scale     * self.step_dt,
            "undesired_contacts":       contacts                    * self.cfg.undesired_contact_reward_scale   * self.step_dt,
            "stumble":                  stumble                     * self.cfg.stumble_reward_scale             * self.step_dt,
            "termination":              termination                 * self.cfg.termination_reward_scale         * self.step_dt,
            "three_finger":             normed_exponential          * self.cfg.three_finger_reward_scale        * self.step_dt,
            "theta_marg_sum":           theta_marg_sum              * self.cfg.theta_marg_sum_reward_scale      * self.step_dt,
            # "a_marg":                   a_marg                      * self.cfg.a_marg_reward_scale              * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # @track_time
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._cs_base_id], dim=-1), dim=1)[0] > 1.0, dim=1)

        tot_force = dict()
        tot_mask = dict()
        mask = torch.zeros(self.num_envs, device= self.device)
        for id in self._cs_foot_ids.keys():
            tot_force[id] =     torch.sum(net_contact_forces[:, :, self._cs_foot_ids[id]], dim=2, keepdim=True)
            tot_mask[id] =      torch.any(torch.max(torch.norm(tot_force[id], dim=-1), dim=1)[0] > self.cfg.feet_termination_force, dim=1)
            mask = torch.logical_or(mask,tot_mask[id])

        # if torch.any(died):
        #     print("died-termination-base-contact")
        # if torch.any(mask):
        #     print("mask")
        died =  torch.logical_or(died,mask)
        return died, time_out

    # @track_time
    def _update_pose_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self._robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self._robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self._robot.data.heading_w)

    # @track_time
    def _update_terrain_curriculum(self, env_ids):
        # Implement Terrain curriculum
        if cnt==0:
            # don't change on initial reset
            return

        distance_to_goal_xy = torch.norm(self.pos_command_b[env_ids,:2], dim=1)
        distance_to_goal = torch.norm(self.pos_command_b[env_ids], dim=1)

        move_up = distance_to_goal <= 0.5
        move_down = (distance_to_goal > 1.) * ~move_up

        if hasattr(self._terrain,"terrain_levels"):
            self._terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

            # Robots that solve the last level are sent to a random one
            self._terrain.terrain_levels[env_ids] = torch.where(self._terrain.terrain_levels[env_ids] >=self._terrain.max_terrain_level,
                                                                    torch.randint_like(self._terrain.terrain_levels[env_ids], self._terrain.max_terrain_level),
                                                                    torch.clip(self._terrain.terrain_levels[env_ids], 0)) # (the minumum level is zero)

            self._terrain.env_origins[env_ids]    = self._terrain.terrain_origins[self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids]]
        # else:
        #     print("terreno piatto o senza attributo terrain_levels")

    # @track_time
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._update_terrain_curriculum(env_ids)

        global cnt
        cnt = 1

        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # self._resample_pose_command(env_ids)
        self._resample_command_terrain_based(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        #muovo robot nel nuovo terreno in accordo curriculum
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     if isinstance(self.cfg, PosGraceRoughEnvCfg):
    #         self._pos_command_visualizer._set_debug_vis_impl(debug_vis)
    #
    # def _debug_vis_callback(self, event):
    #     # update the markers
    #     if isinstance(self.cfg, PosGraceRoughEnvCfg):
    #         self._pos_command_visualizer._debug_vis_callback(event)
    #
    #         translations = self.scene.env_origins
    #         self._vacuum_visualizer.visualize()
