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

import sys
sys.path.append("/home/amosca/IsaacLab/vacuum/")

from LSTM_Helper import *
import itertools

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
                "theta_marg_sum",
                "a_marg"

            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        self._foot_ids = {'rl': self._contact_sensor.find_bodies("LR_FOOT_FINGER.*")[0],
                          'fr': self._contact_sensor.find_bodies("RF_FOOT_FINGER.*")[0],
                          'fl': self._contact_sensor.find_bodies("LF_FOOT_FINGER.*")[0],
                          'rr': self._contact_sensor.find_bodies("RR_FOOT_FINGER.*")[0]}
        self._vacuum_ids = [self._foot_ids[idx] for idx in self._foot_ids.keys()]
        self._vacuum_name = [idx for idx in self._foot_ids.keys()]
        self._vacuum_ids = list(itertools.chain.from_iterable(self._vacuum_ids ))
        self._id_acc_foot = { 'rl': self._robot.find_bodies("LR_FOOT")[0],
                              'fr': self._robot.find_bodies("RF_FOOT")[0],
                              'fl': self._robot.find_bodies("LF_FOOT")[0],
                              'rr': self._robot.find_bodies("RR_FOOT")[0]}

        # zero_force_finger = torch.tensor(self.num_envs, 3)
        # self._vacuum_force = {  "rl": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        #                         "fl": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        #                         "rr": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        #                         "fr": {"finger_1": zero_force_finger.clone(), "finger_2": zero_force_finger.clone(), "finger_3": zero_force_finger.clone()},
        # }


        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")

        self._min_finger_contacts = 3

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*HFE", ".*KFE"])
        self._all_joints, _ = self._robot.find_joints(['^(?!.*_FOOT.*$).*'])


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
        self._num_bodies_vacuum = len(self._vacuum_ids)
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
        self.pos_command_w[env_ids, 2] += self._robot.data.default_root_state[env_ids, 2]

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
        # self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:,self._all_joints]
        self._actions_pos = self._actions[:,:-4*3]
        self._processed_actions_pos = self.cfg.action_scale * self._actions_pos + self._robot.data.default_joint_pos[:, self._all_joints]

        self._action_vacuum = self._actions[:,-4*3:]
        self._processed_action_vacuum = self.cfg.action_scale * self._action_vacuum
        self._processed_action_vacuum = torch.abs(self._processed_action_vacuum )
        self._processed_action_vacuum = torch.clamp(self._processed_action_vacuum,min=0.,max=1.)
        self._processed_action_vacuum = torch.where(self._processed_action_vacuum<3/5, 0., self._processed_action_vacuum) # voltage
        contact_time = self._contact_sensor.data.current_contact_time[:, self._vacuum_ids]
        if self._vacuum_time is None:
            self._vacuum_time = contact_time
        if self._vacuum_old is None:
            self._vacuum_old = contact_time

        mask = torch.logical_and(self._processed_action_vacuum>0., contact_time>0.)
        self._vacuum_old = torch.where(mask,self._vacuum_old, contact_time)
        self._vacuum_time = contact_time - self._vacuum_old
        self._forces_vacuum = torch.zeros_like(self._forces_vacuum, device=self.device)
        self._forces_vacuum[:, :, 2][mask] = -self._lstm_vacuum.predict(self._vacuum_time, self._processed_action_vacuum)[mask]


        if self.sim.has_gui():
            scales = torch.ones_like(self._forces_vacuum, device=self.device)
            scales[:, :, 2][mask] = self._forces_vacuum[:, :, 2][mask] / 380 # 380 --> max force from LSTM
            translations = self._robot.data.body_pos_w[:, self._vacuum_ids, :]
            translations[:, :, 2][torch.logical_not(mask)] += self.cfg.vacuum_visualizer.markers["cylinder_no_contact"].height / 2
            translations[:, :, 2][mask] += -scales[:, :, 2][mask] * self.cfg.vacuum_visualizer.markers["cylinder_no_contact"].height / 2
            scales = scales.reshape((-1, 3))
            translations = translations.reshape((-1, 3))

            no_contact_mask = (contact_time==0).flatten()
            contact_mask = torch.logical_and(contact_time>0, mask==False).flatten()
            vacuum_mask = mask.flatten()
            vacuum_indices = torch.ones_like(vacuum_mask, device=self.device).int()
            vacuum_indices[no_contact_mask] = 0
            vacuum_indices[contact_mask] = 1
            vacuum_indices[vacuum_mask] = 2

            self._vacuum_visualizer.visualize(translations=translations, scales=scales, marker_indices=vacuum_indices)

    # @track_time
    def _apply_action(self):
        # self._robot.set_joint_position_target(self._processed_actions, self._all_joints)
        self._robot.set_joint_position_target(self._processed_actions_pos, self._all_joints)
        self._robot.set_external_force_and_torque(self._forces_vacuum, self._torques_vacuum, env_ids=torch.arange(self.num_envs, device=self.device), body_ids=self._vacuum_ids)
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
                    self.pose_command(),  # 3
                    self._remaining_time(),  # 1
                    height_data,#187
                    self._actions
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
        pos_fingers = self._robot.data.body_pos_w[:, self._foot_ids[name], :]
        self.pos_foot_w[name] = pos_fingers.mean(dim=1)  # Media delle posizioni delle dita

        self.foot_in_contact[name] = self._contact_sensor.data.current_contact_time[:, self._foot_ids[name]].sum(dim=1) > 0
        # forces_foot = torch.zeros([self.num_envs, 3, 3], dtype=torch.float32, device=self.device)

        forces_foot = self._contact_sensor.data.net_forces_w[:, self._foot_ids[name], :]

        #ordinate _forces_vacuum in accordo a vacuum_ids e vacuum_names
        vacuum = torch.zeros_like(self._forces_vacuum[:, :3, :], device=self.device)
        if name in "rl":
            vacuum = self._forces_vacuum[:, :3, :]#[17,18,19]
        if name in "fr":
            vacuum = self._forces_vacuum[:,3:6,:] #[26,27,28]
        if name in "fl":
            vacuum = self._forces_vacuum[:,6:9,:] #[23,24,25]
        if name in "rr":
            vacuum = self._forces_vacuum[:,9:12,:] #[20,21,22]

        mask = torch.norm(vacuum, dim=-1) > 0.
        # if torch.any(mask):
        #     pippo=1
        forces_foot[mask] = vacuum[mask]
        forces_world = math_utils.quat_rotate(self._robot.data.body_quat_w[:, self._foot_ids[name]], forces_foot)
        self.force_w[name] = forces_world.sum(dim=1)

    # @track_time
    def _theta_marg_and_a_marg(self):
        # Gravito-inertial acceleration
        acc_mass_w  = self._robot.data.body_lin_acc_w * self._robot.data.default_mass.unsqueeze(-1).to(device=self.device)
        ag_total_w  = acc_mass_w.sum(dim=1) / self.tot_mass
        self.a_gi_w = self._robot.data.GRAVITY_VEC_W - ag_total_w

        # Center of mass
        self.com_w = torch.sum(
            self._robot.data.body_pos_w * self._robot.data.default_mass.unsqueeze(-1).to(device=self.device), dim=1
        ) / self.tot_mass

        # Foot properties and contacts
        for name in self._foot_ids.keys():
            self.compute_foot_properties(name)

        # Cross products for tumbling axes
        self.n_gab_w = {
            "fl-fr": torch.cross(self.com_w - self.pos_foot_w["fl"], self.com_w - self.pos_foot_w["fr"], dim=1),
            "fr-rr": torch.cross(self.com_w - self.pos_foot_w["fr"], self.com_w - self.pos_foot_w["rr"], dim=1),
            "rr-rl": torch.cross(self.com_w - self.pos_foot_w["rr"], self.com_w - self.pos_foot_w["rl"], dim=1),
            "rl-fl": torch.cross(self.com_w - self.pos_foot_w["rl"], self.com_w - self.pos_foot_w["fl"], dim=1),
            "fl-rr": torch.cross(self.com_w - self.pos_foot_w["fl"], self.com_w - self.pos_foot_w["rr"], dim=1),
            "fr-rl": torch.cross(self.com_w - self.pos_foot_w["fr"], self.com_w - self.pos_foot_w["rl"], dim=1),
        }

        self.bitmap_contatc = {
                "fl": self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0,
                "fr": self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0,
                "rr": self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0,
                "rl": self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0,
                "fl-fr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0),
                "fr-rr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0),
                "rr-rl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0),
                "rl-fl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0),
                "fl-rr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0),
                "fr-rl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0),
        }

        is_active           = torch.stack([self.bitmap_contatc["fl-fr"], self.bitmap_contatc["fr-rr"], self.bitmap_contatc["rr-rl"], self.bitmap_contatc["rl-fl"], self.bitmap_contatc["fl-rr"], self.bitmap_contatc["fr-rl"]], dim=0)

        if torch.any(is_active):
            pippo = 1

        # Normalize tumbling axis vectors
        for key, value in self.n_gab_w.items():
            self.n_gab_w[key] = self.safe_normalize(value)

        # Compute mass_times_agilim_dot_n_agab
        for key, value in self.foot_faces.items():
            foot_j1 = self.check_face[key][0]
            foot_j2 = self.check_face[key][1]
            foot_b = self.foot_faces[key][1]
            foot_a = self.foot_faces[key][0]
            temp_j1 = torch.cross(self.pos_foot_w[foot_b] - self.pos_foot_w[foot_j1], self.pos_foot_w[foot_a] - self.pos_foot_w[foot_j1], dim=1)
            temp_j2 = torch.cross(self.pos_foot_w[foot_b] - self.pos_foot_w[foot_j2], self.pos_foot_w[foot_a] - self.pos_foot_w[foot_j2], dim=1)
            self.mass_times_agilim_dot_n_agab_w[key] = torch.sum(self.force_w[foot_j1] * temp_j1, dim=1) + torch.sum(self.force_w[foot_j2] * temp_j2, dim=1)  # DA VERIFICARE SEGNO

        # Solve for agilim
        for key in self.n_gab_w.keys(): #metto a zero le N_gab che non sono attive
            mask = torch.logical_not(self.bitmap_contatc[key])
            self.n_gab_w[key][mask] = torch.zeros((3), device=self.device)
            self.mass_times_agilim_dot_n_agab_w[key][mask] = 0.


        A = torch.stack([self.n_gab_w[key] for key in self.n_gab_w.keys()], dim=1).to(self.device)
        b = torch.stack([self.mass_times_agilim_dot_n_agab_w[key] for key in self.n_gab_w.keys()], dim=1).to(self.device)
        b = b.unsqueeze(2)  # (num_envs, num_faces, 1)
        if A.shape[1] >= 3:  # Assicura che ci siano almeno 3 vincoli
            A_pseudo_inv = torch.linalg.pinv(A)  # Calcolo robusto della pseudo-inversa
            self.a_gilim_w = torch.matmul(A_pseudo_inv, b).squeeze(-1)
        else:
            raise ValueError("Numero insufficiente di vincoli per calcolare a_gi,lim.")


        rew_eth = torch.zeros(self.num_envs, device=self.device)
        rew_amarg = torch.zeros(self.num_envs, device=self.device)

        epsilon = 1e-8  # Piccolo valore per evitare divisioni per zero

        for key in self.foot_faces.keys():
            n_agb = self.n_gab_w[key]

            # Calcolo delle norme con aggiunta di epsilon per evitare divisioni per zero
            norm_n_agb = torch.linalg.norm(n_agb, dim=1) + epsilon
            norm_a_gi_w = torch.linalg.norm(self.a_gi_w, dim=1) + epsilon
            norm_a_gilim_w = torch.linalg.norm(self.a_gilim_w, dim=1) + epsilon

            # Calcolo di cos_theta_agi e cos_theta_gilim con denominatore corretto
            cos_theta_agi = torch.clip(torch.sum(n_agb * self.a_gi_w, dim=1) / (norm_n_agb * norm_a_gi_w), -1.0, 1.0)
            cos_theta_gilim = torch.clip(norm_a_gilim_w / norm_a_gi_w, -1.0, 1.0)

            # Calcolo di theta_marg
            self.theta_marg[key] = (torch.arccos(cos_theta_agi) - torch.arccos(cos_theta_gilim)) - torch.pi / 2

            # Calcolo di a_marg
            self.a_marg[key] = norm_a_gilim_w - torch.sum(n_agb * self.a_gi_w, dim=1) / norm_n_agb

            # Calcolo di is_inside_poly
            self.is_inside_poly[key] = torch.sum(n_agb * self.a_gi_w, dim=1) <= torch.sum(n_agb * self.a_gilim_w, dim=1)

            # Aggiornamento di rew_eth e rew_amarg
            rew_eth[:] += self.theta_marg[key]
            rew_amarg[:] += self.a_marg[key]

        a_marg_stack        = torch.stack([self.a_marg["fl-fr"], self.a_marg["fr-rr"], self.a_marg["rr-rl"], self.a_marg["rl-fl"], self.a_marg["fl-rr"], self.a_marg["fr-rl"]], dim=0)
        theta_marg_stack    = torch.stack([self.theta_marg["fl-fr"], self.theta_marg["fr-rr"], self.theta_marg["rr-rl"], self.theta_marg["rl-fl"], self.theta_marg["fl-rr"], self.theta_marg["fr-rl"]], dim=0)
        is_in_poly          = torch.stack([self.is_inside_poly["fl-fr"], self.is_inside_poly["fr-rr"], self.is_inside_poly["rr-rl"], self.is_inside_poly["rl-fl"], self.is_inside_poly["fl-rr"], self.is_inside_poly["fr-rl"]], dim=0)

        mask_active_in_poly = is_in_poly * is_active
        zeros = torch.zeros(self.num_envs, device=self.device)

        if torch.any(mask_active_in_poly.sum(dim=0) >= 3):
            pippo = 1

        amin        = torch.where(mask_active_in_poly.sum(dim=0) >= 3, a_marg_stack.min(dim=0).values, 0)
        theta_min   = torch.where(mask_active_in_poly.sum(dim=0) >= 3, theta_marg_stack.min(dim=0).values, 0)

        if torch.any(amin):
            pippo = 1
        if torch.any(theta_min):
            pippo = 1

        # #IN ACCORDO CON THESIS CEWEILBEL
        self._amarg         = torch.max(zeros, amin).to(device=self.device)
        # self._amarg = a_marg_stack.min(dim=0).values.sum(dim=0).to(device=self.device)
        # self._thetamarg     = torch.max(zeros, theta_min).to(device=self.device)
        #
        # #IN ACCORDO ARTICOLO VALSECCHI
        self._sumthetamarg  = theta_marg_stack.sum(dim=0).to(device=self.device)





    # def _theta_marg_and_a_marg(self):
    #     self.acc_mass_w = self._robot.data.body_lin_acc_w * self._robot.data.default_mass.unsqueeze(-1).to(device=self.device)
    #     self.ag_total_w = self.acc_mass_w.sum(dim=1) / self.tot_mass
    #     self.a_gi_w = self._robot.data.GRAVITY_VEC_W - self.ag_total_w
    #     self.com_w = torch.sum(self._robot.data.body_pos_w[:, :, :] * self._robot.data.default_mass.unsqueeze(-1).to(device=self.device), dim=1) / self.tot_mass
    #
    #     self.bitmap_contatc = {
    #         "fl": self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0,
    #         "fr": self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0,
    #         "rr": self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0,
    #         "rl": self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0,
    #         "fl-fr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0),
    #         "fr-rr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0),
    #         "rr-rl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0),
    #         "rl-fl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0),
    #         "fl-rr": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fl"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rr"]].sum(dim=1) > 0),
    #         "fr-rl": torch.logical_and(self._contact_sensor.data.current_contact_time[:, self._foot_ids["fr"]].sum(dim=1) > 0,self._contact_sensor.data.current_contact_time[:, self._foot_ids["rl"]].sum(dim=1) > 0),
    #     }
    #
    #     for name in self._foot_ids.keys():
    #         pos_finger_1_w = self._robot.data.body_pos_w[:, self._foot_ids[name][0], :]
    #         pos_finger_2_w = self._robot.data.body_pos_w[:, self._foot_ids[name][1], :]
    #         pos_finger_3_w = self._robot.data.body_pos_w[:, self._foot_ids[name][2], :]
    #
    #         self.pos_foot_w[name] = (pos_finger_1_w + pos_finger_2_w + pos_finger_3_w) / 3
    #
    #         self.foot_in_contact[name] = self._contact_sensor.data.current_contact_time[:, self._foot_ids[name]].sum(dim=1) > 0
    #
    #         force_1_foot = torch.zeros([self.num_envs, 3], dtype=torch.float32, device=self.device)
    #         force_2_foot = torch.zeros([self.num_envs, 3], dtype=torch.float32, device=self.device)
    #         force_3_foot = torch.zeros([self.num_envs, 3], dtype=torch.float32, device=self.device)
    #
    #         #DA ADD QUI VACUUM FORCE
    #         # force_1_foot[:, 2] = - contact_sensor.data.vacuum_action_force[:, self._foot_ids[name][0]] * self._contact_sensor.data.current_contact_time[:, self._foot_ids[name][0]] > 0
    #         # force_2_foot[:, 2] = - contact_sensor.data.vacuum_action_force[:, self._foot_ids[name][1]] * self._contact_sensor.data.current_contact_time[:, self._foot_ids[name][1]] > 0
    #         # force_3_foot[:, 2] = - contact_sensor.data.vacuum_action_force[:, self._foot_ids[name][2]] * self._contact_sensor.data.current_contact_time[:, self._foot_ids[name][2]] > 0
    #
    #         force_1_w = math_utils.quat_rotate(self._robot.data.body_quat_w[:, self._foot_ids[name][0]], force_1_foot)
    #         force_2_w = math_utils.quat_rotate(self._robot.data.body_quat_w[:, self._foot_ids[name][1]], force_2_foot)
    #         force_3_w = math_utils.quat_rotate(self._robot.data.body_quat_w[:, self._foot_ids[name][2]], force_3_foot)
    #
    #         self.force_w[name] = (force_1_w + force_3_w + force_2_w) / 3
    #
    #     self.n_gab_w = {
    #         "fl-fr": torch.cross(self.com_w-self.pos_foot_w["fl"], self.com_w-self.pos_foot_w["fr"], dim=1),
    #         "fr-rr": torch.cross(self.com_w-self.pos_foot_w["fr"], self.com_w-self.pos_foot_w["rr"], dim=1),
    #         "rr-rl": torch.cross(self.com_w-self.pos_foot_w["rr"], self.com_w-self.pos_foot_w["rl"], dim=1),
    #         "rl-fl": torch.cross(self.com_w-self.pos_foot_w["rl"], self.com_w-self.pos_foot_w["fl"], dim=1),
    #         "fl-rr": torch.cross(self.com_w-self.pos_foot_w["fl"], self.com_w-self.pos_foot_w["rr"], dim=1),
    #         "fr-rl": torch.cross(self.com_w-self.pos_foot_w["fr"], self.com_w-self.pos_foot_w["rl"], dim=1),
    #     }
    #
    #     for key, value in self.n_gab_w.items():
    #         self.n_gab_w[key] = value / torch.norm(value, dim=1, keepdim=True)
    #
    #     for key, value in self.foot_faces.items():
    #         foot_j1 = self.check_face[key][0]
    #         foot_j2 = self.check_face[key][1]
    #         foot_b = self.foot_faces[key][1]
    #         foot_a = self.foot_faces[key][0]
    #         temp_j1 = torch.cross(self.pos_foot_w[foot_b] - self.pos_foot_w[foot_j1], self.pos_foot_w[foot_a] - self.pos_foot_w[foot_j1], dim=1)
    #         temp_j2 = torch.cross(self.pos_foot_w[foot_b] - self.pos_foot_w[foot_j2], self.pos_foot_w[foot_a] - self.pos_foot_w[foot_j2], dim=1)
    #         self.mass_times_agilim_dot_n_agab_w[key] = torch.sum(self.force_w[foot_j1] * temp_j1, dim=1) + torch.sum(self.force_w[foot_j2] * temp_j2, dim=1)  # DA VERIFICARE SEGNO
    #
    #     A = torch.stack([self.n_gab_w[key] for key in self.foot_faces.keys()], dim=1).to(device=self.device)
    #     b = torch.stack([self.mass_times_agilim_dot_n_agab_w[key] for key in self.foot_faces.keys()], dim=1).to(device=self.device)
    #     b = b.unsqueeze(2) # (n_envs, n_faces, 1)
    #     A_pseudo_inv = torch.linalg.pinv(A)  # (n_envs, 3, n_faces)
    #     self.a_gilim_w = torch.matmul(A_pseudo_inv, b)  # (n_envs, 3, 1)
    #     self.a_gilim_w = self.a_gilim_w.squeeze(-1)
    #
    #
    #     rew_eth = torch.zeros(self.num_envs, device=self.device)
    #     rew_amarg = torch.zeros(self.num_envs, device=self.device)
    #
    #     for key in self.foot_faces.keys():
    #         n_agb = self.n_gab_w[key]
    #         cos_theta_agi = torch.clip(torch.sum(n_agb * self.a_gi_w, dim=1) / (torch.linalg.norm(n_agb, dim=1) * torch.linalg.norm(self.a_gi_w, dim=1)), -1.0, 1.0)
    #         cos_theta_gilim = torch.clip(torch.linalg.norm(self.a_gilim_w, dim=1) / torch.linalg.norm(self.a_gi_w, dim=1), -1.0, 1.0)
    #         self.theta_marg[key] = (torch.arccos(cos_theta_agi) - torch.arccos(cos_theta_gilim)) - torch.pi / 2
    #         self.a_marg[key] = torch.linalg.norm(self.a_gilim_w, dim=1) - torch.sum(n_agb * self.a_gi_w, dim=1) / torch.linalg.norm(n_agb, dim=1)
    #         self.is_inside_poly[key] = torch.sum(n_agb * self.a_gi_w, dim=1) <= torch.sum(n_agb * self.a_gilim_w, dim=1)
    #         rew_eth[:] += self.theta_marg[key]
    #         rew_amarg[:] += self.a_marg[key]
    #
    #     # mask = torch.zeros(asset.num_instances, device=asset.device)
    #     # for key in foot_in_contact.keys():
    #     #     mask += foot_in_contact[key].to(torch.float)
    #
    #     a_marg_stack = torch.stack([self.a_marg["fl-fr"], self.a_marg["fr-rr"], self.a_marg["rr-rl"], self.a_marg["rl-fl"], self.a_marg["fl-rr"], self.a_marg["fr-rl"]], dim=0)
    #     theta_marg_stack = torch.stack([self.theta_marg["fl-fr"], self.theta_marg["fr-rr"], self.theta_marg["rr-rl"], self.theta_marg["rl-fl"], self.theta_marg["fl-rr"], self.theta_marg["fr-rl"]], dim=0)
    #     is_in_poly = torch.stack([self.is_inside_poly["fl-fr"], self.is_inside_poly["fr-rr"], self.is_inside_poly["rr-rl"], self.is_inside_poly["rl-fl"], self.is_inside_poly["fl-rr"], self.is_inside_poly["fr-rl"]], dim=0)
    #     is_active = torch.stack([self.bitmap_contatc["fl-fr"], self.bitmap_contatc["fr-rr"], self.bitmap_contatc["rr-rl"], self.bitmap_contatc["rl-fl"], self.bitmap_contatc["fl-rr"], self.bitmap_contatc["fr-rl"]], dim=0)
    #
    #     mask_active_in_poly = is_in_poly * is_active
    #     zeros = torch.zeros(self.num_envs, device=self.device)
    #
    #     amin = torch.where(mask_active_in_poly.sum(dim=0) >= 3, a_marg_stack.min(dim=0).values, 0)
    #     theta_min = torch.where(mask_active_in_poly.sum(dim=0) >= 3, theta_marg_stack.min(dim=0).values, 0)
    #
    #     self._amarg = torch.max(zeros, amin).to(device=self.device)
    #     self._thetamarg = torch.max(zeros, theta_min).to(device=self.device)
    #     self._sumthetamarg = theta_marg_stack.sum(dim=0).to(device=self.device)


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
        base_acc = (self.cfg.base_lin_acc_weight * torch.square(torch.norm(self._robot.data.body_lin_acc_w[:, self._base_id, :], dim=-1)) +
                    self.cfg.base_ang_acc_weight * torch.square(torch.norm(self._robot.data.body_ang_acc_w[:, self._base_id, :], dim=-1))).squeeze(dim=1)
        # Feet acc and Feet Force
        feet_acc    = torch.zeros(self.num_envs, device=self.device)
        feet_force  = torch.zeros(self.num_envs, self._contact_sensor.data.net_forces_w_history.shape[1], device=self.device)
        stumble     = torch.zeros(self.num_envs, device=self.device)
        combined_mask = torch.zeros(self.num_envs, device=self.device)
        norm_feet_force_dict = dict()
        for foot in self._id_acc_foot.keys():
            #FEET ACC
            feet_acc    = feet_acc + torch.norm(self._robot.data.body_lin_acc_w[:, self._id_acc_foot[foot], :], dim=-1).squeeze(dim=-1)
            #CONTACT FORCE
            norm_feet_force_dict[foot] = torch.norm(torch.sum(self._contact_sensor.data.net_forces_w_history[:, :, self._foot_ids[foot]], dim=2), dim=-1)
            feet_force  = feet_force + torch.clamp(norm_feet_force_dict[foot] - self.cfg.max_feet_contact_force, min=0)** 2
            #STUMBLE
            fxy = torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._foot_ids[foot], :2], dim=-1)
            fz = torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._foot_ids[foot], 2:], dim=-1)
            stumble = stumble + torch.sum(torch.where(fxy>2*fz,1,0),dim=(1,2))
            #TERMINATION FEET CONTACT
            combined_mask = torch.logical_or(torch.max(norm_feet_force_dict[foot], dim=1)[0] > self.cfg.feet_termination_force,combined_mask)

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
        mask_base_collision = torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0
        termination = torch.where(mask_base_collision.squeeze() | combined_mask, 1, 0)

        theta_marg_sum = self.get_sumthetamarg()

        a_marg = self.get_amarg()

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
            "theta_marg_sum":           theta_marg_sum              * self.cfg.theta_marg_sum_reward_scale      * self.step_dt,
            "a_marg":                   a_marg                      * self.cfg.a_marg_reward_scale              * self.step_dt,
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
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)

        tot_force = dict()
        tot_mask = dict()
        mask = torch.zeros(self.num_envs, device= self.device)
        for id in self._foot_ids.keys():
            tot_force[id] =     torch.sum(net_contact_forces[:, :, self._foot_ids[id]], dim=2, keepdim=True)
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
