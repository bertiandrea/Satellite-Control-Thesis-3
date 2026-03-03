import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

TAGS = ["Reward_policy/phi_mean", "Reward_policy/energy_mean", 
        "Reward_policy/du_energy_mean", "Reward_policy/max_torque_mean"]

LOG_DISPLAY = ["Reward_policy/phi_mean", "Reward_policy/energy_mean", 
        "Reward_policy/du_energy_mean", "Reward_policy/max_torque_mean"]

LOG_VAL = [1e1, 1e3, 1e2, 1e2]

def load_data(files, method):
    step_data = defaultdict(lambda: defaultdict(list))
    for f in files:
        try:
            ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
            ea.Reload()
            for t in set(ea.Tags().get("scalars", [])) & set(TAGS):
                for e in ea.Scalars(t):
                    step_data[t][e.step].append(e.value)
        except: continue
    
    result = {}
    for t in step_data:
        steps = sorted(step_data[t].keys())
        if method == "median":
            m = [np.median(step_data[t][s]) for s in steps]
        else:
            m = [np.mean(step_data[t][s]) for s in steps]
        mi = [np.min(step_data[t][s]) for s in steps]
        ma = [np.max(step_data[t][s]) for s in steps]
        result[t] = {"x": np.array(steps), "m": np.array(m), "min": np.array(mi), "max": np.array(ma)}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--method", choices=["median", "mean"], required=True)
    ap.add_argument("--outdir", default="_img/plots_reward_policy")
    args = ap.parse_args()

    base_path = Path(args.input)
    paths = list(base_path.rglob("*tfevents*"))
    
    grouped = defaultdict(list)
    for p in paths:
        gid = "/".join(p.relative_to(base_path).parts[:2])
        grouped[gid].append(p)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    
    all_results = {gid: load_data(files, args.method) for gid, files in grouped.items()}

    gids_sorted = sorted(all_results.keys(), key=nat_key)
    unique_groups = sorted(
        set((n[:-6] if n.endswith("_noise") else n) for n in (gid.split('/')[0] for gid in all_results.keys())),
        key=nat_key
    )

    if args.training:
        cmap = plt.get_cmap("gist_rainbow", len(gids_sorted))
    else:
        cmap = plt.get_cmap("gist_rainbow", len(unique_groups))

    print (f"Found groups: {unique_groups}")
    for t in TAGS:
        plt.figure(figsize=(20, 8))
        
        # --- PATCH LOG SCALE ---
        if t in LOG_DISPLAY:
            plt.yscale("symlog", linthresh=LOG_VAL[TAGS.index(t)])
        # -----------------------

        for i, gid in enumerate(sorted(all_results.keys(), key=nat_key)):
            if t not in all_results[gid]: continue

            run_name = gid.split('/')[0]

            if args.training:
                # TRAINING: niente gestione _noise, colore unico per gid
                color = cmap(i)
            else:
                # EVAL: raggruppa nominal/noise sullo stesso colore base

                group_name = (run_name[:-6] if run_name.endswith("_noise") else run_name)
                print(f"Plotting {run_name} in group {group_name} for tag {t}...")

                group_idx = unique_groups.index(group_name)
                base_color = cmap(group_idx)
                alpha_val = 1.0 if not run_name.endswith("_noise") else 0.5
                color = (base_color[0], base_color[1], base_color[2], alpha_val)

            d = all_results[gid][t]
            last_val = np.mean(d["m"][-max(1, int(len(d["m"]) * 0.05)):]) if len(d["m"]) > 0 else 0.0
            line, = plt.plot(d["x"], d["m"], label=f"{run_name} ({last_val:.2f})", color=color, linewidth=1.5)
            #plt.fill_between(d["x"], d["min"], d["max"], color=line.get_color(), alpha=0.2)

        plt.title(t.replace('_', ' ').title() + (" (Log Scale)" if t in LOG_DISPLAY else ""), fontsize='xx-large')
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize='large', ncol=2, loc='upper right')
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(out / f"{t.replace('/', '_')}.png", dpi=150)
        plt.close()

    base_runs = sorted((all_results.keys() if args.training else [gid for gid in all_results.keys() if "_noise" not in gid]), key=nat_key)
    
    sub_h = " |  NOMINAL  |   NOISE   |   DIFF%   "
    name_w = 80
    h_top = " " * name_w
    h_mid = f"{'RUN ID':<{name_w}}"
    
    for t in TAGS:
        h_top += f" | {t.split('/')[-1][:12]:^34}"
        h_mid += sub_h

    print(f"\n{h_top}\n{h_mid}\n{'-' * len(h_mid)}")

    for base in base_runs:
        row = f"{base:<{name_w}}"
        p = base.split('/')
        noise_id = base if args.training else (f"{p[0]}_noise/{'/'.join(p[1:])}" if len(p) >= 2 else f"{base}_noise")

        for t in TAGS:
            d_nom = all_results.get(base, {}).get(t, {})
            d_noi = all_results.get(noise_id, {}).get(t, {})
            
            v_nom = np.mean(d_nom["m"][-max(1, int(len(d_nom["m"])*0.05)):]) if "m" in d_nom else None
            v_noi = np.mean(d_noi["m"][-max(1, int(len(d_noi["m"])*0.05)):]) if "m" in d_noi else None

            f = lambda v: f"{v:9.2e}" if v is not None and abs(v) > 9999 else (f"{v:9.2f}" if v is not None else "      N/A")
            
            s_nom = f(v_nom)
            s_noi = f(v_noi)
            s_diff = "        -"
            if v_nom and v_noi:
                diff = (v_noi - v_nom) / abs(v_nom) * 100
                s_diff = f"{diff:+8.1e}%" if abs(diff) > 999 else f"{diff:+8.1f}%"

            row += f" | {s_nom} | {s_noi} | {s_diff} "
        
        print(row)
    print("=" * len(h_mid))

if __name__ == "__main__":
    main()