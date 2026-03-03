# satellite.py

from code.envs.vec_task import VecTask

import isaacgym #BugFix
import torch
from isaacgym import gymutil, gymtorch, gymapi

from pathlib import Path
import numpy as np

class Satellite(VecTask):
    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless):
        self.dt =                    config["sim"].get('dt', 1 / 60.0)                             # seconds

        self.env_spacing =           config["env"].get('envSpacing', 0.0)                          # meters

        self.asset_name =            config["env"]["asset"].get('assetName', 'satellite')
        self.asset_root =            config["env"]["asset"].get('assetRoot', str(Path(__file__).resolve().parent.parent))
        self.asset_file =            config["env"]["asset"].get('assetFileName', 'satellite.urdf')

        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)

        ################# SETUP SIM #################
        self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.actor_root_state).view(self.num_envs, 13)
        self.satellite_pos     = self.root_states[:, 0:3]
        self.satellite_quats   = self.root_states[:, 3:7]
        self.satellite_linvels = self.root_states[:, 7:10]
        self.satellite_angvels = self.root_states[:, 10:13]
        #############################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_root_states = self.root_states.clone()
        print(f"Initial root states: {self.initial_root_states[0]}")
        ########################################
        
    def create_sim(self) -> None:
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params) # Acquires the sim pointer
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []       
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            ###################################################
            asset_init_pos_p = [0, 0, 0]
            asset_init_pos_r = np.random.randn(4)
            asset_init_pos_r /= np.linalg.norm(asset_init_pos_r)
            ###################################################
            self.create_actor(i, env, self.asset, asset_init_pos_p, asset_init_pos_r, 1, self.asset_name)
            ###################################################
            self.envs.append(env)

    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        return asset
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        actor_handle =  self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)
        return actor_handle

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        return
    
    def pre_physics_step(self, actions):
        return

    def post_physics_step(self):
        return

