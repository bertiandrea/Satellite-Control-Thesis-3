# evaluate.py

from code.configs.satellite_config import CONFIG
from code.envs.satellite import Satellite
from code.models.custom_model import Shared
from code.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper

import isaacgym #BugFix
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import argparse
import json
import datetime
import os
import re
from pathlib import Path

def deep_update(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of the run to evaluate"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        help="Name of the config file of run to evaluate"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # ──────────────────────────────────────────────────────────────────────────
    with open(str(Path(__file__).resolve().parent.parent.parent / "Training" / "run_config" / f"{args.config_name}.json"), "r") as f:
        training_config = json.load(f)

    deep_update(training_config, CONFIG)

    CONFIG.clear()
    CONFIG.update(training_config)
    # ──────────────────────────────────────────────────────────────────────────

    if CONFIG["set_seed"]:
        set_seed(CONFIG["seed"])
    else:
        CONFIG["seed"] = torch.seed() % (2**32)
        set_seed(CONFIG["seed"])

    # ──────────────────────────────────────────────────────────────────────────
    # 🔹 Salvataggio config + informazioni di run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = "run_config"
    os.makedirs(run_dir, exist_ok=True)
    safe_run_name = re.sub(r'[\\/]+', '_', args.run_name)
    config_path = os.path.join(run_dir, f"evaluate_config_{safe_run_name}_{timestamp}.json")

    config_to_save = CONFIG.copy()
    config_to_save["model_path"] = str(Path(__file__).resolve().parent.parent.parent / "Training" / "runs" / f"{safe_run_name}.pt")

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=4, default=str)

    print(f"[INFO] Config di valutazione salvata in: {config_path}")
    # ──────────────────────────────────────────────────────────────────────────

    print(CONFIG)
    
    #################################################################################

    env = Satellite(
        config=CONFIG,
        rl_device=CONFIG["rl_device"],
        sim_device=CONFIG["sim_device"],
        graphics_device_id=CONFIG["graphics_device_id"],
        headless=CONFIG["headless"],
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=CONFIG["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Shared(env.state_space, env.action_space, env.device)
    models["value"] = models["policy"]  # Shared model for policy and value
   
    CONFIG["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    CONFIG["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update(CONFIG["rl"]["PPO"])
   
    agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)
    agent.load(str(Path(__file__).resolve().parent.parent.parent / "Training" / "runs" / f"{args.run_name}.pt"))

    print(str(Path(__file__).resolve().parent.parent.parent / "Training" / "runs"  / f"{args.run_name}.pt"))
    
    trainer = SequentialTrainer(cfg=CONFIG["rl"]["trainer"], env=env, agents=agent)

    trainer.eval()

    #################################################################################

if __name__ == "__main__":
    main()