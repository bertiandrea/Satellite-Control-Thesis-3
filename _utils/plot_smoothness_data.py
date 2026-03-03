import argparse
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

# ------------------- FFT + CoM -------------------
def compute_fft(actions, fps):
    T, _, _ = actions.shape
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps)
    fft_envs = torch.fft.rfft(actions, dim=0)
    amp_envs = fft_envs.abs() / T
    mean_fft = amp_envs.mean(dim=1)
    std_fft = amp_envs.std(dim=1)
    com_envs = (amp_envs * freqs[:, None, None]).sum(dim=0) / amp_envs.sum(dim=0)
    return mean_fft, std_fft, freqs, com_envs.mean(dim=0), com_envs.std(dim=0)

def plot_component_across_files(out_dir, title, list_of_data, labels):
    C = list_of_data[0][2]["mean"].shape[1] 
    list_of_data.sort(key=lambda x: nat_key(x[0]))

    unique_groups = sorted(
        set(((gid.split('/')[0][:-6] if gid.split('/')[0].endswith("_noise") else gid.split('/')[0])
             for gid, _, _ in list_of_data)),
        key=nat_key
    )
    cmap = plt.get_cmap("gist_rainbow", len(unique_groups))

    for i, label in enumerate(labels):
        if i >= C: break
        plt.figure(figsize=(20, 8))

        for run_name, freqs, data in list_of_data:
            fr = freqs.numpy() 
            mean = data["mean"][:, i].numpy()
            std  = data["mean_std"][:, i].numpy()
            c_mean = data["com_mean"][i].item()
            # c_std  = data["com_std"][i].item()

            run_name  = run_name.split('/')[0]
            group_name = (run_name[:-6] if run_name.endswith("_noise") else run_name)
            print(f"Plotting {run_name} in group {group_name}")

            group_idx = unique_groups.index(group_name)
            base_color = cmap(group_idx)
            alpha_val = 1.0 if not run_name.endswith("_noise") else 0.5
            color = (base_color[0], base_color[1], base_color[2], alpha_val)
            
            plt.plot(fr, mean, color=color, label=f"{run_name} (CoM: {c_mean:.1f}Hz)", lw=1.5)
            plt.fill_between(fr, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.1)
            plt.axvline(c_mean, color=color, linestyle="--", alpha=1.0, lw=1.0)
            # plt.axvspan(c_mean-c_std, c_mean+c_std, color=color, alpha=0.05)  # Optional

        plt.yscale("symlog", linthresh=1e0)
        plt.ylim(bottom=0)
        plt.title(f"{title} - {label}", fontsize='xx-large')
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xlabel("Frequency (Hz)")
        plt.legend(loc='upper right', fontsize='x-large', ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{title.lower().replace(' ', '_')}_{label.lower()}.png", dpi=300)
        plt.close()

# ------------------- MAIN -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--outdir", type=str, default="_img/plots_smoothness_fft")
    args = ap.parse_args()

    base_path = Path(args.input)
    paths = list(base_path.rglob("*.pt*"))

    grouped = defaultdict(list)
    for p in paths:
        gid = "/".join(p.relative_to(base_path).parts[:2])
        grouped[gid].append(p)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for gid, files in grouped.items():
        print(f"--> Loading Group: {gid} ({len(files)} files)")
        try:
            all_actions = []
            for f in files:
                data = torch.load(f, map_location="cpu", weights_only=True)

                actions = torch.stack([d["actions"] for d in data])

                all_actions.append(actions)

            mean_fft, std_fft, freqs, com_m, com_s = compute_fft(
                torch.cat(all_actions, dim=1),
                args.fps
            )
            mean_norm, std_norm, freqs_n, com_m_n, com_s_n = compute_fft(
                torch.linalg.norm(torch.cat(all_actions, dim=1), dim=2, keepdim=True),
                args.fps
            )

            all_results[gid] = {
                "freqs": freqs,
                "axis": {"mean": mean_fft, "mean_std": std_fft, "com_mean": com_m, "com_std": com_s},
                "norm": {"mean": mean_norm, "mean_std": std_norm, "com_mean": com_m_n, "com_std": com_s_n},
            }
        except Exception as e:
            print(f"   [!] Error loading {gid}: {e}")

    def extract(key):
        return [(gid, d["freqs"], d[key]) for gid, d in all_results.items()]

    if not all_results:
        print("No valid data found.")
        return

    plot_component_across_files(out, "FFT Per Axis", extract("axis"), ["X", "Y", "Z"])
    plot_component_across_files(out, "FFT Magnitude", extract("norm"), ["Norm"])

if __name__ == "__main__":
    main()