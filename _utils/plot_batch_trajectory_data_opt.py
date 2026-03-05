import argparse
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def plot_component_across_files(out_dir, title, list_of_data, labels, method, non_negative=False, log_scale=False, log_threshold=None):
    C = list_of_data[0][2].shape[2]

    list_of_data.sort(key=lambda x: nat_key(x[0]))
    
    unique_groups = sorted(
        set(((gid.split('/')[0][:-6] if gid.split('/')[0].endswith("_noise") else gid.split('/')[0])
             for gid, *_ in list_of_data)),
        key=nat_key
    )
    cmap = plt.get_cmap("gist_rainbow", len(unique_groups))

    for scale in (["linear", "log"] if log_scale else ["linear"]):
        plt.figure(figsize=(20, 8 * C))
        for i, label in enumerate(labels):
            if i >= C: break
            plt.subplot(C, 1, i + 1)

            for run_name, steps, mean_mat, std_mat in list_of_data:
                if method == "median":
                    mean = np.median(mean_mat[:, :, i], axis=1)
                    std  = np.median(std_mat[:, :, i], axis=1)
                else:
                    mean = np.mean(mean_mat[:, :, i], axis=1)
                    std  = np.mean(std_mat[:, :, i], axis=1)

                step_np = steps.numpy()

                lower, upper = mean - std, mean + std
                if non_negative:
                    lower = np.maximum(lower, 0.0)

                run_name = run_name.split('/')[0]
                group_name = (run_name[:-6] if run_name.endswith("_noise") else run_name)
                print(f"Plotting {run_name} in group {group_name}")

                group_idx = unique_groups.index(group_name)
                base_color = cmap(group_idx)
                alpha_val = 1.0 if not run_name.endswith("_noise") else 0.5
                color = (base_color[0], base_color[1], base_color[2], alpha_val)

                last_val = np.mean(mean[-max(1, int(len(mean) * 0.05)):]) if len(mean) > 0 else 0.0
                plt.plot(step_np, mean, label=f"{run_name} ({last_val:.2f})", color=color, linewidth=1.5)
                plt.fill_between(step_np, lower, upper, color=color, alpha=0.15)

            if scale == "log":
                plt.yscale("symlog", linthresh=log_threshold)

            plt.title(f"{title} – {label} ({scale})", fontsize='xx-large')
            plt.ylabel(f"{label} ({scale})")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(loc='upper right', fontsize='xx-large', ncol=2)

        plt.xlabel("Step")
        plt.tight_layout()
        plt.savefig(out_dir / f"{title.replace(' ', '_').lower()}_{scale}.png", dpi=300)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--method", type=str, choices=["mean", "median"], required=True)
    ap.add_argument("--outdir", type=str, default="_img/plots_trajectories")
    args = ap.parse_args()

    base_path = Path(args.input)
    paths = list(base_path.rglob("*.pt*"))

    grouped = defaultdict(list)
    for p in paths:
        gid = "/".join(p.relative_to(base_path).parts[:2])
        grouped[gid].append(p)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- LOAD DATA ----------------
    all_results = {} 
    for gid, files in grouped.items():
        print(f"--> Loading Group: {gid} ({len(files)} files)")
        try:
            list_quat_m, list_quat_s = [], []
            list_angdiff_m, list_angdiff_s = [], []
            list_angvel_m, list_angvel_s = [], []
            list_angacc_m, list_angacc_s = [], []
            list_actions_m, list_actions_s = [], []
            list_energy_m, list_energy_s = [], []
            list_denergy_m, list_denergy_s = [], []
            list_maxaction_m, list_maxaction_s = [], []

            list_steps = []
            for f in files:
                data = torch.load(f, map_location="cpu", weights_only=True)

                d_steps   = torch.tensor([d["step"] for d in data])
                d_quat    = torch.stack([d["quat"] for d in data])
                d_angdiff = torch.stack([d["ang_diff"] for d in data]).unsqueeze(-1)
                d_angvel  = torch.stack([d["angvel"] for d in data])
                d_angacc  = torch.stack([d["angacc"] for d in data])
                d_actions = torch.stack([d["actions"] for d in data])

                # Energy: ||a_t||^2  -> (T, E, 1)
                d_energy = (d_actions ** 2).sum(dim=-1, keepdim=True)

                # Delta-energy: ||a_t - a_{t-1}||^2 -> (T, E, 1)
                d_denergy = torch.zeros_like(d_energy)
                d_denergy[1:] = ((d_actions[1:] - d_actions[:-1]) ** 2).sum(dim=-1, keepdim=True)

                # Max action
                d_maxaction = d_actions.abs().max(dim=-1, keepdim=True)[0]
                
                def ms(x):
                    return x.mean(dim=1), x.std(dim=1)

                qm, qs = ms(d_quat);        list_quat_m.append(qm);        list_quat_s.append(qs)
                am, as_ = ms(d_angdiff);    list_angdiff_m.append(am);     list_angdiff_s.append(as_)
                vm, vs = ms(d_angvel);      list_angvel_m.append(vm);      list_angvel_s.append(vs)
                acm, acs = ms(d_angacc);    list_angacc_m.append(acm);     list_angacc_s.append(acs)
                actm, acts = ms(d_actions); list_actions_m.append(actm);   list_actions_s.append(acts)
                em, es = ms(d_energy);      list_energy_m.append(em);      list_energy_s.append(es)
                dem, des = ms(d_denergy);   list_denergy_m.append(dem);    list_denergy_s.append(des)
                mm, ms_ = ms(d_maxaction);  list_maxaction_m.append(mm);   list_maxaction_s.append(ms_)

                list_steps.append(d_steps)

            def stack_runs(lst):
                return torch.stack(lst, dim=1).numpy()

            all_results[gid] = {
                "steps": list_steps[0], # assume all runs have the same steps
                "quat":      (stack_runs(list_quat_m),      stack_runs(list_quat_s)),
                "angdiff":   (stack_runs(list_angdiff_m),   stack_runs(list_angdiff_s)),
                "angvel":    (stack_runs(list_angvel_m),    stack_runs(list_angvel_s)),
                "angacc":    (stack_runs(list_angacc_m),    stack_runs(list_angacc_s)),
                "actions":   (stack_runs(list_actions_m),   stack_runs(list_actions_s)),
                "energy":    (stack_runs(list_energy_m),    stack_runs(list_energy_s)),
                "denergy":   (stack_runs(list_denergy_m),   stack_runs(list_denergy_s)),
                "maxaction": (stack_runs(list_maxaction_m), stack_runs(list_maxaction_s)),
            }
        except Exception as e:
            print(f"   [!] Error loading {gid}: {e}")

    def extract(key):
        return [(gid, d["steps"], d[key][0], d[key][1]) for gid, d in all_results.items()]
    
    if not all_results:
        print("No valid data found.")
        return

    plot_component_across_files(out, "Quaternion", extract("quat"), ["x", "y", "z", "w"], args.method)
    plot_component_across_files(out, "Angular Difference", extract("angdiff"), ["angle"], args.method, non_negative=True, log_scale=True, log_threshold=1e0)
    plot_component_across_files(out, "Angular Velocity", extract("angvel"), ["x", "y", "z"], args.method)
    plot_component_across_files(out, "Angular Acceleration", extract("angacc"), ["x", "y", "z"], args.method)
    plot_component_across_files(out, "Actions", extract("actions"), ["x", "y", "z"], args.method, log_scale=True, log_threshold=1e0)
    plot_component_across_files(out, "Action Energy ||a_t||^2", extract("energy"), ["energy"], args.method, non_negative=True, log_scale=True, log_threshold=1e2)
    plot_component_across_files(out, "Action Delta Energy ||a_t-a_{t-1}||^2", extract("denergy"), ["delta_energy"], args.method, non_negative=True, log_scale=True, log_threshold=1e2)
    plot_component_across_files(out, "Max Action", extract("maxaction"), ["max_action"], args.method, non_negative=True, log_scale=True, log_threshold=1e1)

if __name__ == "__main__":
    main()