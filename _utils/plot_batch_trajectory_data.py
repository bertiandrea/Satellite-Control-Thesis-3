import argparse
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import gc

def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def plot_component_across_files(out_dir, title, list_of_data, labels, method, non_negative=False, log_scale=False, log_threshold=None):
    C = list_of_data[0][2].shape[2]

    list_of_data.sort(key=lambda x: nat_key(x[0]))
    
    unique_groups = sorted(
        set(((gid.split('/')[0][:-6] if gid.split('/')[0].endswith("_noise") else gid.split('/')[0])
             for gid, _, _ in list_of_data)),
        key=nat_key
    )
    cmap = plt.get_cmap("gist_rainbow", len(unique_groups))

    for scale in (["linear", "log"] if log_scale else ["linear"]):
        plt.figure(figsize=(20, 8 * C))
        for i, label in enumerate(labels):
            if i >= C: break
            plt.subplot(C, 1, i + 1)

            for run_name, steps, data in list_of_data:
                if method == "median":
                    mean = data[:, :, i].median(dim=1).values.numpy()
                    std = data[:, :, i].std(dim=1).numpy()
                else:
                    mean = data[:, :, i].mean(dim=1).numpy()
                    std = data[:, :, i].std(dim=1).numpy()
                
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
        print(f"Found file: {p} (group: {gid})")
        grouped[gid].append(p)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    results = defaultdict(dict)  # results[gid][col] = float

    def load_metric_for_all_groups(metric: str):
        list_of_data = []
        for gid, files in grouped.items():
            print(f"--> Loading Group: {gid} ({len(files)} files) [metric={metric}]")
            try:
                list_steps = []
                list_metric = []

                for f in files:
                    data = torch.load(f, map_location="cpu", weights_only=True)

                    d_steps = torch.tensor([d["step"] for d in data])

                    if metric == "quat":
                        d_val = torch.stack([d["quat"] for d in data])  # (T, E, 4)
                    elif metric == "angdiff":
                        d_val = torch.stack([d["ang_diff"] for d in data]).unsqueeze(-1)  # (T, E, 1)
                    elif metric == "angvel":
                        d_val = torch.stack([d["angvel"] for d in data])  # (T, E, 3)
                    elif metric == "angacc":
                        d_val = torch.stack([d["angacc"] for d in data])  # (T, E, 3)
                    else:
                        d_actions = torch.stack([d["actions"] for d in data])  # (T, E, A)

                        if metric == "actions":
                            d_val = d_actions
                        elif metric == "energy":
                            d_val = (d_actions ** 2).sum(dim=-1, keepdim=True)  # (T, E, 1)
                        elif metric == "denergy":
                            d_energy = (d_actions ** 2).sum(dim=-1, keepdim=True)
                            d_val = torch.zeros_like(d_energy)
                            d_val[1:] = ((d_actions[1:] - d_actions[:-1]) ** 2).sum(dim=-1, keepdim=True)
                        elif metric == "maxaction":
                            d_val = d_actions.abs().max(dim=-1, keepdim=True)[0]
                        else:
                            raise ValueError(f"Unknown metric: {metric}")

                    list_steps.append(d_steps)
                    list_metric.append(d_val)

                list_of_data.append((gid, list_steps[0], torch.cat(list_metric, dim=1)))
                del list_steps, list_metric
                gc.collect()
            except Exception as e:
                print(f"   [!] Error loading {gid} metric={metric}: {e}")

        return list_of_data

    def update_summary(metric_key: str, data_list):
        for gid, steps, tensor in data_list:  # tensor: (T, E, C)
            print(f"Updating summary for {gid} metric={metric_key}")
            if args.method == "median":
                curve = tensor[:, :, 0].median(dim=1).values.numpy()
            else:
                curve = tensor[:, :, 0].mean(dim=1).numpy()
            results[gid][metric_key] = np.mean(curve[-max(1, int(len(curve) * 0.05)):]) if len(curve) > 0 else None

    data = load_metric_for_all_groups("quat")
    plot_component_across_files(out, "Quaternion", data, ["x", "y", "z", "w"], method=args.method)
    del data
    gc.collect()

    data = load_metric_for_all_groups("angdiff")
    update_summary("phi_mean", data)
    plot_component_across_files(out, "Angular Difference", data, ["angle"], method=args.method, non_negative=True, log_scale=True, log_threshold=1e0)
    del data
    gc.collect()

    data = load_metric_for_all_groups("angvel")
    plot_component_across_files(out, "Angular Velocity", data, ["x", "y", "z"], method=args.method)
    del data
    gc.collect()

    data = load_metric_for_all_groups("angacc")
    plot_component_across_files(out, "Angular Acceleration", data, ["x", "y", "z"], method=args.method)
    del data
    gc.collect()

    data = load_metric_for_all_groups("actions")
    plot_component_across_files(out, "Actions", data, ["x", "y", "z"], method=args.method, log_scale=True, log_threshold=1e0)
    del data
    gc.collect()

    data = load_metric_for_all_groups("energy")
    update_summary("energy_mean", data)
    plot_component_across_files(out, "Action Energy ||a_t||^2", data, ["energy"], method=args.method, non_negative=True, log_scale=True, log_threshold=1e2)
    del data
    gc.collect()

    data = load_metric_for_all_groups("denergy")
    update_summary("du_energy_mean", data)
    plot_component_across_files(out, "Action Delta Energy ||a_t-a_{t-1}||^2", data, ["delta_energy"], method=args.method, non_negative=True, log_scale=True, log_threshold=1e2)
    del data
    gc.collect()

    data = load_metric_for_all_groups("maxaction")
    update_summary("max_torque_mean", data)
    plot_component_across_files(out, "Max Action", data, ["max_action"], method=args.method, non_negative=True, log_scale=True, log_threshold=1e1)
    del data
    gc.collect()

    # ================== PRINT TABLE ==================
    cols = ["phi_mean", "energy_mean", "du_energy_mean", "max_torque_mean"]

    base_runs = sorted([gid for gid in results.keys() if "_noise" not in gid], key=nat_key)
    print(f"Base runs: {base_runs}")
    
    sub_h = " |  NOMINAL  |   NOISE   |   DIFF%   "
    name_w = 80

    h_top = " " * name_w
    h_mid = f"{'RUN ID':<{name_w}}"

    for c in cols:
        h_top += f" | {c[:12]:^34}"
        h_mid += sub_h

    print(f"\n{h_top}\n{h_mid}\n{'-' * len(h_mid)}")

    for base in base_runs:
        row = f"{base:<{name_w}}"
        p = base.split("/")
        noise_id = f"{p[0]}_noise/{'/'.join(p[1:])}"

        for c in cols:
            v_nom = results.get(base, {}).get(c, None)
            v_noi = results.get(noise_id, {}).get(c, None)

            f = lambda v: f"{v:9.2e}" if v is not None and abs(v) > 9999 else (f"{v:9.2f}" if v is not None else "      N/A")

            s_nom = f(v_nom)
            s_noi = f(v_noi)

            s_diff = "        -"
            if (v_nom is not None) and (v_noi is not None):
                if v_nom == 0.0:
                    s_diff = "   +inf%" if v_noi > 0 else ("   -inf%" if v_noi < 0 else "    0.0%")
                else:
                    diff = (v_noi - v_nom) / abs(v_nom) * 100.0
                    s_diff = f"{diff:+8.1e}%" if abs(diff) > 999 else f"{diff:+8.1f}%"

            row += f" | {s_nom} | {s_noi} | {s_diff} "

        print(row)

    print("=" * len(h_mid))
    # ================================================

if __name__ == "__main__":
    main()