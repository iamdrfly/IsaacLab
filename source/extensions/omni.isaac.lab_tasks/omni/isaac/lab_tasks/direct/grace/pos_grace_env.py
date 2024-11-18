# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from hid import device
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .pos_grace_env_cfg import PosGraceFlatEnvCfg, PosGraceRoughEnvCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, wrap_to_pi, quat_rotate_inverse, yaw_quat
from collections.abc import Sequence

class GraceEnv(DirectRLEnv):
    cfg: PosGraceFlatEnvCfg | PosGraceRoughEnvCfg

    def __init__(self, cfg: PosGraceFlatEnvCfg | PosGraceRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self.init_done = False
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands ------------------------------------------------------> da cambiare per POS
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

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
                "termination"

            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        self._foot_ids = {'lr': self._contact_sensor.find_bodies("LR_FOOT_FINGER.*")[0],
                          'rf': self._contact_sensor.find_bodies("RF_FOOT_FINGER.*")[0],
                          'lf': self._contact_sensor.find_bodies("LF_FOOT_FINGER.*")[0],
                          'rr': self._contact_sensor.find_bodies("RR_FOOT_FINGER.*")[0]}

        self._id_acc_foot = { 'lr': self._robot.find_bodies("LR_FOOT")[0],
                              'rf': self._robot.find_bodies("RF_FOOT")[0],
                              'lf': self._robot.find_bodies("LF_FOOT")[0],
                              'rr': self._robot.find_bodies("RR_FOOT")[0]}


        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")

        self._min_finger_contacts = 3

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*HFE", ".*KFE"])
        self._all_joints, _ = self._robot.find_joints(['^(?!.*_FOOT.*$).*'])


        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        self.error_pos = torch.zeros(self.num_envs, device=self.device)
        self.error_heading = torch.zeros(self.num_envs, device=self.device)
        self.remaining_time = torch.zeros(self.num_envs, device=self.device)

        self.joint_vel_limit = torch.zeros((self.num_envs,len(self._all_joints)), device=self.device)
        self.joint_effort_limit = torch.zeros_like(self.joint_vel_limit )

        for act in self._robot.actuators.keys():
            self.joint_vel_limit[:,self._robot.actuators[act]._joint_indices] = self._robot.actuators[act].velocity_limit
            self.joint_effort_limit[:, self._robot.actuators[act]._joint_indices] = self._robot.actuators[act].effort_limit

        # self.init_done = True

    def pose_command(self) -> torch.Tensor:
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    def _update_pose_metrics(self):
        self.error_pos_2d = torch.norm(self.pos_command_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1)
        self.error_heading = torch.abs(wrap_to_pi(self.heading_command_w - self._robot.data.heading_w))

    def _resample_pose_command(self, env_ids: Sequence[int], init_done=False):
        if init_done:
            # don't change on initial reset
            return
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self.scene.env_origins[env_ids]
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)

        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges["pos_x"])
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges["pos_y"])

        # self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2] #da mettere altezza qui

        if (self.cfg.simple_heading):
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
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges["heading"])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, PosGraceRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:,self._all_joints]

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, self._all_joints)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, PosGraceRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)

        self._update_pose_command()

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self.pose_command(),
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

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

    def _remaining_time(self):
        self.remaining_time = self.max_episode_length_s - (self.episode_length_buf * (self.cfg.sim.dt * self.cfg.decimation)).squeeze(dim=-1)

    def _get_rewards(self) -> torch.Tensor:

        #compute remaining time
        self._remaining_time()

        #XY-Position Tracking
        self._update_pose_metrics()
        position_tracking_mapped = torch.where(self.remaining_time < 1, (1 - 0.5 * self.error_pos_2d), 0.0)
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


        feet_force = torch.max(feet_force, dim=-1)[0]

            # Action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # Don't wait
        dont_wait = torch.where(torch.norm(self._robot.data.root_lin_vel_b, dim=-1) < self.cfg.wait_time, 1., 0.)
        # Move in direction
        target_vec = self.pos_command_w  - self._robot.data.root_pos_w
        move_in_direction = torch.sum(self._robot.data.root_lin_vel_b * target_vec, dim=-1) / (torch.norm(self._robot.data.root_lin_vel_b, dim=-1) * torch.norm(target_vec, dim=-1)  + 1e-6)
        # Stand at target
        mask = torch.logical_and(torch.where(self.error_pos_2d <self.cfg.stand_min_dist,1,0),torch.where(self.error_heading < self.cfg.stand_min_ang, 1, 0))
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



        # # linear velocity tracking
        # lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        # lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # # yaw rate tracking
        # yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        # yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        # z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # # angular velocity x/y
        # ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        #
        # # joint acceleration
        # joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        # last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (torch.norm(self._commands[:, :2], dim=1) > 0.1)
        # first_contacts, last_air_times = self._compute_foot_contact(self._contact_sensor, self.step_dt, self._foot_ids, min_contacts=self._min_finger_contacts)
        # air_time = 0
        # for foot in ['lr', 'rf', 'lf', 'rr']:
        #     air_time += (last_air_times[foot] - 0.5) * first_contacts[foot]
        #
        # # Applica condizione aggiuntiva per il movimento
        # air_time *= (torch.norm(self._commands[:, :2], dim=1) > 0.1)


        # flat orientation
        # flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "position_tracking_xy":     position_tracking_mapped * self.cfg.position_tracking_reward_scale* self.step_dt,
            "heading_tracking_xy":      heading_tracking_mapped * self.cfg.heading_tracking_reward_scale * self.step_dt,
            "dof_vel_l2":               joint_vel * self.cfg.joint_vel_reward_scale * self.step_dt,
            "dof_torques_l2":           joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_vel_limit":            joint_vel_limit * self.cfg.joint_vel_limit_reward_scale * self.step_dt,
            "dof_torques_limit":        joint_eff_limit * self.cfg.joint_torque_limit_reward_scale * self.step_dt,
            "base_acc":                 base_acc * self.cfg.base_acc_reward_scale * self.step_dt,
            "feet_acc":                 feet_acc * self.cfg.feet_acc_reward_scale * self.step_dt,
            "action_rate_l2":           action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_contact_force":       feet_force * self.cfg.feet_contact_force_reward_scale * self.step_dt,
            "dont_wait":                dont_wait * self.cfg.dont_wait_reward_scale * self.step_dt,
            "move_in_direction":        move_in_direction * self.cfg.move_in_direction_reward_scale * self.step_dt,
            "stand_at_target":          stand_at_target * self.cfg.stand_at_target_reward_scale * self.step_dt,
            "undesired_contacts":       contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "stumble":                  stumble * self.cfg.stumble_reward_scale * self.step_dt,
            "termination":              termination * self.cfg.termination_reward_scale * self.step_dt,
            # "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            # "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            # "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,

            # "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,

            # "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

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

        if torch.any(mask):
            print("ciao")
        died =  torch.logical_or(died,mask)
        return died, time_out

    def _update_pose_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self._robot.data.root_pos_w
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self._robot.data.root_quat_w), target_vec)[:,:2]
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self._robot.data.heading_w)

    def _update_terrain_curriculum(self, env_ids, init_done = False):
        # Implement Terrain curriculum
        if init_done:
            # don't change on initial reset
            return

        distance_to_goal = torch.norm(self.pos_command_b[env_ids,:2], dim=1)

        move_up = distance_to_goal <= 0.5
        move_down = (distance_to_goal > 1.) * ~move_up

        self._terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        # Robots that solve the last level are sent to a random one
        self._terrain.terrain_levels[env_ids] = torch.where(self._terrain.terrain_levels[env_ids] >=self._terrain.max_terrain_level,
                                                                torch.randint_like(self._terrain.terrain_levels[env_ids], self._terrain.max_terrain_level),
                                                                torch.clip(self._terrain.terrain_levels[env_ids], 0)) # (the minumum level is zero)

        self._terrain.env_origins[env_ids]    = self._terrain.terrain_origins[self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids]]

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

        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._resample_pose_command(env_ids)

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
