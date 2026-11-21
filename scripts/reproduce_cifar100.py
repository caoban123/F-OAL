"""Unified CIFAR-100 F-OAL reproduction launcher (smoke & full modes).

This thin launcher delegates directly to `scripts.train.run` without duplicating
training, dataset loading, RLS, or evaluation logic.

Usage:
    python -m scripts.reproduce_cifar100 --mode smoke
    python -m scripts.reproduce_cifar100 --mode full --config configs/c3_d_bilinear_raw.yaml --seed 42
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_PACKAGES = ("torch", "torchvision", "timm", "datasets", "yaml", "numpy")
PROTECTED_RESULT_PREFIXES = (
    "results/c1_",
    "results/c2_",
    "results/c3_",
    "results/table4_",
    "results/headline_",
    "results/cifar100_multiseed",
    "results/official_foal_cifar100_seed0",
    "results/foal_full_cifar100_seed42",
)


def check_environment() -> dict[str, str]:
    """Verify Python >= 3.9 and required scientific packages are importable.

    Returns:
        Mapping of package name to installed version string.
    """
    if sys.version_info < (3, 9):
        raise RuntimeError(
            f"Python 3.9+ is required, but found {sys.version_info.major}.{sys.version_info.minor}."
        )
    versions: dict[str, str] = {}
    missing: list[str] = []
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "installed"))
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            f"Missing required dependencies: {', '.join(missing)}. "
            "Install them with: pip install -r requirements.txt"
        )
    return versions


def resolve_config_path(config_arg: str) -> Path:
    """Resolve and validate a YAML configuration file path relative to repo root or CWD.

    Args:
        config_arg: Path to YAML config file.

    Returns:
        Resolved Path to the existing YAML file.
    """
    raw = Path(config_arg)
    candidate = raw if raw.is_absolute() else (Path.cwd() / raw)
    if not candidate.exists():
        candidate = REPO_ROOT / raw
    if not candidate.exists():
        raise FileNotFoundError(f"Config file not found: {config_arg}")
    cfg = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "dataset" not in cfg or "model" not in cfg:
        raise ValueError(f"Invalid F-OAL config structure in {candidate}")
    return candidate


def resolve_output_dir(mode: str, seed: int, output_dir_arg: str | None) -> Path:
    """Resolve an output directory while protecting official verified result folders in smoke mode.

    Args:
        mode: Execution mode ('smoke' or 'full').
        seed: Random seed.
        output_dir_arg: Optional user-provided output directory.

    Returns:
        Path to the target output directory.
    """
    if output_dir_arg:
        out = Path(output_dir_arg)
        resolved = out if out.is_absolute() else (REPO_ROOT / out)
    elif mode == "smoke":
        resolved = REPO_ROOT / "results" / f"smoke_cifar100_seed{seed}"
    else:
        resolved = REPO_ROOT / "results" / f"reproduce_cifar100_seed{seed}"

    if mode == "smoke":
        try:
            rel = resolved.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            rel = resolved.as_posix()
        for prefix in PROTECTED_RESULT_PREFIXES:
            if rel.startswith(prefix):
                raise ValueError(
                    f"Smoke mode cannot write into protected official directory '{rel}'. "
                    "Use an isolated directory such as results/smoke_cifar100."
                )
    return resolved


def build_train_namespace(args: argparse.Namespace, config_path: Path, output_dir: Path) -> argparse.Namespace:
    """Build an argparse.Namespace compatible with `scripts.train.run`.

    Args:
        args: Parsed launcher arguments.
        config_path: Validated config file path.
        output_dir: Validated output directory.

    Returns:
        Namespace ready for `scripts.train.run`.
    """
    is_smoke = args.mode == "smoke"
    return argparse.Namespace(
        config=str(config_path),
        max_tasks=2 if is_smoke else None,
        limit_train_per_task=20 if is_smoke else None,
        limit_test_per_task=20 if is_smoke else None,
        seed=args.seed,
        class_order_seed=args.seed,
        projection_seed=args.seed,
        projection_init=None,
        stream_seed=args.seed,
        device=args.device or ("auto" if is_smoke else None),
        experiment_type="pilot" if is_smoke else "reproduction",
        no_feature_fusion=False,
        no_projection=False,
        feature_cache_dir=args.feature_cache_dir,
        run_name=args.run_name or (f"smoke_cifar100_seed{args.seed}" if is_smoke else f"reproduce_cifar100_seed{args.seed}"),
        output_dir=str(output_dir),
        wandb=False if is_smoke else bool(args.wandb),
        wandb_offline=False if is_smoke else bool(args.wandb_offline),
        wandb_project=None,
        wandb_entity=None,
        wandb_base_url=None,
        wandb_group=None,
        wandb_tags=None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the unified CIFAR-100 reproduction launcher.

    Args:
        argv: Optional list of CLI argument strings.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Unified CIFAR-100 F-OAL reproduction launcher (smoke & full modes)."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Execution mode: 'smoke' (2 tasks, 20 samples/task, no W&B) or 'full' (10 tasks, full CIFAR-100).",
    )
    parser.add_argument(
        "--config",
        default="configs/c3_d_bilinear_raw.yaml",
        help="YAML configuration file (default: configs/c3_d_bilinear_raw.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for class order, projection, and stream (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Target output directory (default: results/smoke_cifar100_seed<seed> or results/reproduce_cifar100_seed<seed>).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help="Optional directory to cache deterministic frozen ViT features.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run name.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging in 'full' mode (ignored in 'smoke' mode).",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Use offline W&B mode in 'full' mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run CIFAR-100 F-OAL reproduction in smoke or full mode.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Metrics dictionary returned by `scripts.train.run`.
    """
    args = parse_args(argv)
    versions = check_environment()
    config_path = resolve_config_path(args.config)
    output_dir = resolve_output_dir(args.mode, args.seed, args.output_dir)

    print("=" * 72)
    print(f"CIFAR-100 F-OAL Launcher | mode={args.mode} | seed={args.seed}")
    print(f"Config     : {config_path}")
    print(f"Output Dir : {output_dir}")
    print(f"Stack      : torch={versions['torch']}, timm={versions['timm']}, datasets={versions['datasets']}")
    print("=" * 72)

    from scripts import train as train_module

    train_args = build_train_namespace(args, config_path, output_dir)
    prev_wandb_mode = os.environ.get("WANDB_MODE")
    if args.mode == "smoke":
        os.environ["WANDB_MODE"] = "disabled"
    try:
        metrics = train_module.run(train_args)
    finally:
        if args.mode == "smoke":
            if prev_wandb_mode is None:
                os.environ.pop("WANDB_MODE", None)
            else:
                os.environ["WANDB_MODE"] = prev_wandb_mode

    expected_files = ("metrics.json", "metadata.json", "task_accuracy.csv")
    missing_outputs = [name for name in expected_files if not (output_dir / name).exists()]
    if missing_outputs:
        raise RuntimeError(f"Expected output files missing in {output_dir}: {missing_outputs}")

    print(
        f"[OK] Completed mode={args.mode} | "
        f"avg_acc={metrics.get('average_incremental_accuracy_pct', metrics.get('average_incremental_accuracy'))} | "
        f"outputs={output_dir}"
    )
    return metrics


if __name__ == "__main__":
    main()
