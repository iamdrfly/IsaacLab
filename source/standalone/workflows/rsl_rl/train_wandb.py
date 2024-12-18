# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

pwd_path = os.getcwd()

PATH_WANDB_FOLDER = pwd_path + "/train_wandb"

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def load_hyperparameters(file_path):
    params = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                params[key] = float(value)
    return params



@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""

    hyper_file = PATH_WANDB_FOLDER + "/hyperisac.txt"
    hyperparams = load_hyperparameters(hyper_file)

    env_cfg.position_tracking_reward_scale  = hyperparams["position_tracking_reward_scale"]
    env_cfg.heading_tracking_reward_scale   = hyperparams["heading_tracking_reward_scale"]
    env_cfg.joint_vel_reward_scale          = hyperparams["joint_vel_reward_scale"]         
    env_cfg.joint_torque_reward_scale       = hyperparams["joint_torque_reward_scale"]      
    env_cfg.joint_vel_limit_reward_scale    = hyperparams["joint_vel_limit_reward_scale"]    
    env_cfg.joint_torque_limit_reward_scale = hyperparams["joint_torque_limit_reward_scale"] 
    env_cfg.base_acc_reward_scale           = hyperparams["base_acc_reward_scale"]           
    env_cfg.base_lin_acc_weight             = hyperparams["base_lin_acc_weight"]             
    env_cfg.base_ang_acc_weight             = hyperparams["base_ang_acc_weight"]             
    env_cfg.feet_acc_reward_scale           = hyperparams["feet_acc_reward_scale"]           
    # env_cfg.action_rate_reward_scale        = hyperparas.action_rate_reward_scale
    env_cfg.max_feet_contact_force          = hyperparams["max_feet_contact_force"]         
    env_cfg.feet_contact_force_reward_scale = hyperparams["feet_contact_force_reward_scale"] 
    env_cfg.wait_time                       = hyperparams["wait_time"]                       
    env_cfg.dont_wait_reward_scale          = hyperparams["dont_wait_reward_scale"]          
    env_cfg.move_in_direction_reward_scale  = hyperparams["move_in_direction_reward_scale"]  
    env_cfg.stand_min_dist                  = hyperparams["stand_min_dist"]                  
    env_cfg.stand_min_ang                   = hyperparams["stand_min_ang"]                   
    env_cfg.stand_at_target_reward_scale    = hyperparams["stand_at_target_reward_scale"]    
    env_cfg.undesired_contact_reward_scale  = hyperparams["undesired_contact_reward_scale"]  
    env_cfg.stumble_reward_scale            = hyperparams["stumble_reward_scale"]            
    env_cfg.feet_termination_force          = hyperparams["feet_termination_force"]          
    env_cfg.termination_reward_scale        = hyperparams["termination_reward_scale"]
    env_cfg.robot.actuators["HAA"].damping[".*"] = hyperparams["haa_damping"]
    env_cfg.robot.actuators["HFE"].damping[".*"] = hyperparams["hfe_damping"]
    env_cfg.robot.actuators["KFE"].damping[".*"] = hyperparams["kfe_damping"]
    env_cfg.robot.actuators["HAA"].stiffness[".*"] = hyperparams["haa_stiffness"]
    env_cfg.robot.actuators["HFE"].stiffness[".*"] = hyperparams["hfe_stiffness"]
    env_cfg.robot.actuators["KFE"].stiffness[".*"] = hyperparams["kfe_stiffness"]
    env_cfg.theta_marg_sum_reward_scale        = hyperparams["theta_marg_sum_reward_scale"]

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    _, training_results = runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()

    # Scrive i risultati nel file resultisaac.txt
    results_file = PATH_WANDB_FOLDER + "/resultisaac.txt"
    with open(results_file, "w") as f:
        f.write(f"mean_reward={training_results['mean_reward']}\n")
        f.write(f"mean_reward_time={training_results['mean_reward_time']}\n")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
