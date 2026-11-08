"""Run the FGVC-Aircraft Table 4 ablation over several seeds and summarize it.

Usage (from the repository root, on a GPU machine):
    python scripts/run_aircraft_ablation.py --output-dir results/aircraft_ablation
Add --wandb to log online; otherwise the runs are logged offline.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

CONDITIONS = ("a0_full", "a1_without_ff", "a2_without_sp", "a3_without_both")
SEEDS = (42, 123, 2026)


def run_condition(name: str, seed: int, output_dir: Path, cache_dir: str, online: bool) -> dict:
    """Train one condition with one seed.

    Args:
        name: condition name, matching a file in configs/aircraft/.
        seed: value used for the base, class-order, projection and stream seeds.
        output_dir: parent folder of the run folder.
        cache_dir: folder for cached backbone features.
        online: log to W&B online when True, offline otherwise.

    Returns:
        The numeric metrics of the run, with its name and seed.
    """
    run_name = f"aircraft_{name}_seed{seed}"
    run_dir = output_dir / run_name
    command = [
        sys.executable, "scripts/train.py", "--config", f"configs/aircraft/aircraft_{name}.yaml",
        "--seed", str(seed), "--class-order-seed", str(seed), "--projection-seed", str(seed),
        "--stream-seed", str(seed), "--feature-cache-dir", cache_dir, "--run-name", run_name,
        "--output-dir", str(run_dir), "--wandb", "--wandb-group", "aircraft_table4_ablation",
        "--wandb-tags", f"aircraft,table4,seed{seed}",
    ]
    if not online:
        command.append("--wandb-offline")
    subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    return {"name": name, "seed": seed, **numeric}


def main() -> None:
    """Parse the arguments, run every condition and write summary.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/aircraft_ablation"))
    parser.add_argument("--cache-dir", default=".cache/aircraft_features")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    rows = [run_condition(name, seed, args.output_dir, args.cache_dir, args.wandb)
            for seed in SEEDS for name in CONDITIONS]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
