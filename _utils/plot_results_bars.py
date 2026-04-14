import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SRC = Path("_img/results.txt")
OUT = Path("_img/results_bars")


def to_float(x: str):
    x = x.strip().replace("%", "")
    if x in {"N/A", "-", ""}:
        return np.nan
    return float(x)


def parse_table(filepath: Path) -> pd.DataFrame:
    lines = filepath.read_text(encoding="utf-8").splitlines()
    rows = []

    for line in lines:
        s = line.strip()
        
        if "|" not in line:
            continue
        if "RUN ID" in line or "NOMINAL" in line:
            continue
        if not s or set(s) <= {"-", "=", "|", " "}:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 13:
            continue

        run_id = parts[0]
        vals = parts[1:13]

        rows.append({
            "run_id": run_id,
            "phi_mean_nominal": to_float(vals[0]),
            "phi_mean_noise": to_float(vals[1]),
            "phi_mean_diff": to_float(vals[2]),
            "energy_mean_nominal": to_float(vals[3]),
            "energy_mean_noise": to_float(vals[4]),
            "energy_mean_diff": to_float(vals[5]),
            "du_energy_me_nominal": to_float(vals[6]),
            "du_energy_me_noise": to_float(vals[7]),
            "du_energy_me_diff": to_float(vals[8]),
            "max_torque_m_nominal": to_float(vals[9]),
            "max_torque_m_noise": to_float(vals[10]),
            "max_torque_m_diff": to_float(vals[11]),
        })

    df = pd.DataFrame(rows)

    extracted = df["run_id"].str.extract(
        r"logs_(base|randomization)_(.+?)/agent_86400"
    )
    df["kind"] = extracted[0]
    df["config"] = extracted[1]

    return df

def save_barplots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["phi_mean", "energy_mean", "du_energy_me", "max_torque_m"]
    value_types = ["nominal", "noise", "diff"]

    configs = sorted(df["config"].dropna().unique())

    for config in configs:
        df_cfg = df[df["config"] == config]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        width = 0.35
        x = np.arange(len(metrics))

        for ax, value_type in zip(axes, value_types):
            base_vals = []
            rand_vals = []

            for metric in metrics:
                base_series = df_cfg.loc[
                    df_cfg["kind"] == "base",
                    f"{metric}_{value_type}"
                ]
                rand_series = df_cfg.loc[
                    df_cfg["kind"] == "randomization",
                    f"{metric}_{value_type}"
                ]

                base_vals.append(base_series.iloc[0] if len(base_series) else np.nan)
                rand_vals.append(rand_series.iloc[0] if len(rand_series) else np.nan)

            ax.bar(x - width / 2, base_vals, width, label="base")
            ax.bar(x + width / 2, rand_vals, width, label="randomization")

            ax.set_title(value_type)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=25, ha="right")
            ax.set_ylabel(value_type)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_yscale("symlog", linthresh=1e0)

            if value_type in {"nominal", "noise"}:
                ax.set_yscale("symlog", linthresh=1e-3)

        axes[0].legend()
        fig.suptitle(f"Confronto base vs randomization - {config}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"{config}_bars.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"File non trovato: {SRC}")

    df = parse_table(SRC)
    save_barplots(df, OUT)
    df.to_csv(OUT / "parsed_results.csv", index=False)

    print(f"Letto: {SRC}")
    print(f"Salvato in: {OUT}")


if __name__ == "__main__":
    main()