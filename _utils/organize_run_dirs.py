import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

# --- CONFIGURAZIONE ---
EXPECTED_SEEDS = [420, 4200, 42000, 420000, 4200000, 420, 4200, 42000, 420000, 4200000]  # lista: può contenere duplicati
EXPECTED_COUNTS = Counter(EXPECTED_SEEDS)
EXPECTED_N = len(EXPECTED_SEEDS)

# Config file example:
# evaluate_config_runs_base_..._agent_28800_20260109_100400.json
CFG_RE = re.compile(r"^(?P<prefix>.*)agent_(?P<a>\d+?)_(?P<y>\d{8})_(?P<tm>\d{6})\.json$")

# Run dir example:
# Jan09_10-04-00...
RUN_RE = re.compile(r"(?P<mo>[A-Z][a-z]{2})(?P<d>\d{2})_(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})")

# Log file example:
# trajectories_20260109_100400.pt
LOGS_RE = re.compile(r"^trajectories_(?P<y>\d{8})_(?P<tm>\d{6})\.pt$")

def base_dir(p: Path) -> Path:
    parent = p.parent
    return parent.parent if parent.name.startswith("agent_") else parent

def get_ts_cfg(name: str) -> datetime:
    m = CFG_RE.search(name)
    return datetime.strptime(
        m.group("y") + m.group("tm"),
        "%Y%m%d%H%M%S"
    ).replace(year=9999)

def get_ts_run(name: str) -> datetime:
    m = RUN_RE.search(name)
    return datetime.strptime(
        f"9999 {m.group('mo')} {m.group('d')} {m.group('h')}:{m.group('m')}:{m.group('s')}",
        "%Y %b %d %H:%M:%S",
    )

def get_ts_logs(name: str) -> datetime:
    m = LOGS_RE.search(name)
    return datetime.strptime(
        m.group("y") + m.group("tm"),
        "%Y%m%d%H%M%S"
    ).replace(year=9999)

def find_seed(data):
    if isinstance(data, dict):
        if isinstance(data.get("seed"), int):
            return data["seed"]
        for v in data.values():
            res = find_seed(v)
            if res is not None:
                return res
    return None

def mv(src: Path, dst: Path, dry: bool) -> None:
    """Move file/dir to dst, creating parents. No-op if already at destination."""
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        return
    if dry:
        print(f"[DRY] mv {src} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"[OK ] mv {src} -> {dst}")

def main():
    ap = argparse.ArgumentParser(description="AUDIT/MOVE configs and runs with exact timestamp matching")
    ap.add_argument("--mode", choices=["audit", "move"], required=True,
                    help="audit: only checks, move: organize into agent_*")
    ap.add_argument("--root", default="~/Satellite-Control-Thesis-3/Evaluating",
                    help="Evaluating root (default: %(default)s)")
    ap.add_argument("--config-dir", default="run_config",
                    help="relative to root (default: %(default)s)")
    ap.add_argument("--runs-dir", default="runs",
                    help="relative to root (default: %(default)s)")
    ap.add_argument("--logs-dir", default="logs",
                    help="relative to root (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only for move mode: print operations without moving")
    ap.add_argument("--logs-delta-s", type=int, default=15,
                    help="delta seconds applied to logs timestamp matching (default: %(default)s)")
    ap.add_argument("--no-log", action="store_true",
                    help="exclude log files from checks and moves (only for move mode)")
    ap.add_argument("--no-seed", action="store_true",
                    help="exclude seed checks")

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    cfg_root = (root / args.config_dir).resolve()
    runs_root = (root / args.runs_dir).resolve()
    if not args.no_log:
        logs_root = (root / args.logs_dir).resolve()

    if not cfg_root.is_dir():
        raise SystemExit(f"config dir not found: {cfg_root}")
    if not runs_root.is_dir():
        raise SystemExit(f"runs dir not found: {runs_root}")
    if not args.no_log:
        if not logs_root.is_dir():
            raise SystemExit(f"logs dir not found: {logs_root}")

    configs = [p for p in cfg_root.rglob("*.json") if p.is_file() and CFG_RE.search(p.name)]
    run_dirs = [d for d in runs_root.rglob("*") if d.is_dir() and RUN_RE.search(d.name)]
    if not args.no_log:
        logs = [p for p in logs_root.rglob("*.pt") if p.is_file() and LOGS_RE.search(p.name)]

    errors = []
    seeds_by_key = {}
    planned_moves = []

    cfg_by_ts = {get_ts_cfg(p.name): p for p in configs}
    runs_by_ts = {get_ts_run(d.name): d for d in run_dirs}
    if not args.no_log:
        logs_by_ts = {get_ts_logs(p.name): p for p in logs}

    for ts, config in sorted(cfg_by_ts.items()):
        m = CFG_RE.search(config.name)
        prefix = m.group("prefix").rstrip("_")
        agent_id = m.group("a")

        try:
            with open(config, "r") as f:
                data = json.load(f)
        except Exception as e:
            errors.append(f"Agente {agent_id} ({prefix}): Impossibile leggere {config.name} ({e})")
            continue

        seed = find_seed(data)
        if seed is None:
            errors.append(f"Agente {agent_id} ({prefix}): Seed non trovato in {config.name}")
            continue

        seeds_by_key.setdefault((prefix, agent_id), []).append(seed)

        run_dir = runs_by_ts.get(ts)
        if run_dir is None:
            errors.append(f"Agente {agent_id} ({prefix}): Nessuna cartella RUN trovata per {config.name} (atteso {ts})")
            continue

        if not args.no_log:
            log_file = None
            for off in range(0, args.logs_delta_s + 1):
                log_file = logs_by_ts.get(ts + timedelta(seconds=off))
                if log_file is not None:
                    break
            
            if log_file is None:
                errors.append(f"Agente {agent_id} ({prefix}): Nessun file LOG trovato per {config.name}")
                continue

        if args.mode == "move":
            agent_dir = f"agent_{agent_id}"

            cfg_base = config.parent.parent if config.parent.name.startswith("agent_") else config.parent
            run_base = run_dir.parent.parent if run_dir.parent.name.startswith("agent_") else run_dir.parent
            if not args.no_log:
                log_base = log_file.parent.parent if log_file.parent.name.startswith("agent_") else log_file.parent

            planned_moves.append((config, cfg_base / agent_dir / config.name))
            planned_moves.append((run_dir, run_base / agent_dir / run_dir.name))
            if not args.no_log:
                planned_moves.append((log_file, log_base / agent_dir / log_file.name))

    # --- SEEDS check ---
    if not args.no_seed:
        for (prefix, agent_id), seeds in seeds_by_key.items():
            found_counts = Counter(seeds)

            if len(seeds) != EXPECTED_N:
                errors.append(
                    f"Agente {agent_id} ({prefix}): numero seed trovato {len(seeds)} (atteso {EXPECTED_N})"
                )

            if found_counts != EXPECTED_COUNTS:
                errors.append(
                    f"Agente {agent_id} ({prefix}): seed trovati {dict(found_counts)} (attesi {dict(EXPECTED_COUNTS)})"
                )

    if errors:
        print(f"\n[FAIL] {args.mode.upper()} fallito con {len(errors)} errori:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    if args.mode == "move":
        moved_src = set()
        for src, dst in planned_moves:
            src_res = src.resolve()
            if src_res in moved_src:
                continue
            mv(src, dst, args.dry_run)
            moved_src.add(src_res)

    print(f"\n[OK] {args.mode.upper()} completato")

if __name__ == "__main__":
    main()
