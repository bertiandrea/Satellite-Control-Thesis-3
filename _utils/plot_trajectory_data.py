import torch
import matplotlib.pyplot as plt
import random
import os

N_ENV_PLOT = 32
LOG_PATH = "/home/andreaberti/Satellite-Control-Thesis-3/Evaluating/logs/logs_base_1/agent_86400/trajectories_20260128_174539.pt"

print(f"Caricamento log da: {LOG_PATH}")
data = torch.load(LOG_PATH, map_location="cpu", weights_only=True)
steps = [entry["step"] for entry in data]
quat_all     = torch.stack([entry["quat"] for entry in data])
ang_diff_all = torch.stack([entry["ang_diff"] for entry in data]).unsqueeze(-1)
angvel_all   = torch.stack([entry["angvel"] for entry in data])
angacc_all   = torch.stack([entry["angacc"] for entry in data])
actions_all  = torch.stack([entry["actions"] for entry in data])

num_envs = quat_all.shape[1]
env_indices = random.sample(range(num_envs), min(N_ENV_PLOT, num_envs))

def plot_component(title, data_all, labels, non_negative=False, log_scale=False):
    C = data_all.shape[2]

    cmap = plt.get_cmap("gist_rainbow")

    # ------------- LINEAR -------------
    plt.figure(figsize=(14, 3*C))
    for i, label in enumerate(labels):
        plt.subplot(C,1,i+1)

        for j, env in enumerate(env_indices):
            plt.plot(
                steps,
                data_all[:, env, i],
                alpha=0.8,
                color=cmap(j / (len(env_indices)-1))  
            )
        mean = data_all[:,:,i].mean(dim=1)
        plt.plot(steps, mean, color='black')

        std = data_all[:,:,i].std(dim=1)
        lower, upper = mean - std, mean + std
        if non_negative:
            lower = torch.clamp(lower, min=0.0)
        plt.fill_between(steps, lower, upper, color='grey', alpha=0.5)

        plt.ylabel(label)
        plt.title(f"{title} - {label}")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("Step")
    plt.tight_layout()
    plt.savefig(f"_img/plots_trajectories_env/{title.replace(' ','_').lower()}.png", dpi=600)
    plt.close()

    # ------------- LOG -------------
    if log_scale:
        plt.figure(figsize=(14, 3*C))
        for i, label in enumerate(labels):
            plt.subplot(C,1,i+1)
            
            for j, env in enumerate(env_indices):
                plt.plot(
                    steps,
                    data_all[:, env, i],
                    alpha=0.8,
                    color=cmap(j / (len(env_indices)-1))  
              )
            mean = data_all[:,:,i].mean(dim=1)
            plt.plot(steps, mean, color='black')

            std = data_all[:,:,i].std(dim=1)
            lower, upper = mean - std, mean + std
            if non_negative:
                lower = torch.clamp(lower, min=0.0)
            plt.fill_between(steps, lower, upper, color='grey', alpha=0.5)

            plt.yscale("symlog", linthresh=1e0)

            plt.ylabel(label + " [log]")
            plt.title(f"{title} - {label} [log]")
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xlabel("Step")
        plt.tight_layout()
        plt.savefig(f"_img/plots_trajectories_env/{title.replace(' ','_').lower()}_log.png", dpi=600)
        plt.close()

os.makedirs("_img/plots_trajectories_env", exist_ok=True)
plot_component("Quaternion", quat_all, ["x","y","z","w"])
plot_component("Angular difference (deg)", ang_diff_all, ["angle (deg)"], non_negative=True, log_scale=True)
plot_component("Angular velocity", angvel_all, ["x","y","z"])
plot_component("Angular acceleration", angacc_all, ["x","y","z"])
plot_component("Actions", actions_all, ["x","y","z"], log_scale=True)
