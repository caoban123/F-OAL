"""Sync verified Lab 2 Empirical Demonstration results to public W&B with `lab2-empirical-demo` tags.

Creates clean, curve-structured W&B runs in `hcmus-foal-reproduction` without touching any
existing Lab 1 reproduction runs, recording exact run IDs and URLs to
`results/lab2_empirical/wandb_lab2_manifest.json` so the Portrait A2 Poster and `README.md`
cite live, verifiable W&B runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.foal.environment import get_environment_info

DEFAULT_ENTITY = "caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm"
DEFAULT_PROJECT = "hcmus-foal-reproduction"


def sha256_file(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    """Sync verified Lab 2 E1 and E3 JSON summaries to W&B and write run manifest."""
    parser = argparse.ArgumentParser(description="Sync Lab 2 empirical runs to W&B.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--lab2-dir", default="results/lab2_empirical")
    parser.add_argument("--wandb-dir", default="D:/.cache/wandb")
    args = parser.parse_args()

    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(Path(args.wandb_dir).resolve())

    import wandb

    lab2_root = Path(args.lab2_dir)
    e1_path = lab2_root / "e1_longtail" / "cifar100" / "e1_longtail_summary.json"
    e3_path = lab2_root / "e3_scaling_gamma" / "cifar100" / "e3_scaling_gamma_summary.json"
    if not e1_path.exists() or not e3_path.exists():
        raise FileNotFoundError("Run E1 and E3 experiments before syncing to W&B.")

    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    e3 = json.loads(e3_path.read_text(encoding="utf-8"))
    e1_sha = sha256_file(e1_path)
    e3_sha = sha256_file(e3_path)
    env_info = get_environment_info()

    manifest_runs: list[dict[str, Any]] = []

    # 1. Sync 3 E1 Long-Tail Method Curves (3-seed Mean +/- Std across rho = 1..100)
    for m_id in ("vanilla_foal", "norm_calibrated", "balanced_woodbury"):
        rows = sorted([r for r in e1["aggregated"] if r["method"] == m_id], key=lambda x: x["rho"])
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=f"lab2_e1_cifar100_{m_id}_3seed",
            group="lab2_e1_longtail_cifar100",
            job_type="lab2_empirical_demo",
            tags=["lab2-empirical-demo", "critical-analysis", "e1-longtail", "cifar100", m_id],
            config={
                "lab": "Lab2_Empirical_Demonstration",
                "experiment": "E1_LongTail_OCIL_3Seed",
                "dataset": "cifar100",
                "method": m_id,
                "method_label": rows[0]["method_label"],
                "seeds": e1["seeds"],
                "rhos": e1["rhos"],
                "longtail_mode": e1["longtail_mode"],
                "source_json_sha256": e1_sha,
                "provenance": e1["provenance"],
                "environment": env_info,
            },
            reinit=True,
        )
        for row in rows:
            rho_int = int(round(row["rho"]))
            run.log(
                {
                    "imbalance_ratio_rho": row["rho"],
                    "average_accuracy_mean": row["average_accuracy_mean"],
                    "average_accuracy_std": row["average_accuracy_std"],
                    "last_accuracy_mean": row["last_accuracy_mean"],
                    "tail_acc_mean": row["tail_acc_mean"],
                    "tail_acc_std": row["tail_acc_std"],
                    "head_acc_mean": row["head_acc_mean"],
                    "harmonic_mean_acc_mean": row["harmonic_mean_acc_mean"],
                    "norm_quota_pearson_raw_mean": row["norm_quota_pearson_raw_mean"],
                },
                step=rho_int,
            )
        r100 = rows[-1]
        run.summary.update({
            "rho100_tail_acc_mean": r100["tail_acc_mean"],
            "rho100_tail_acc_std": r100["tail_acc_std"],
            "rho100_average_accuracy_mean": r100["average_accuracy_mean"],
            "rho100_last_accuracy_mean": r100["last_accuracy_mean"],
            "rho100_norm_quota_pearson": r100["norm_quota_pearson_raw_mean"],
        })
        url = run.url
        rid = run.id
        run.finish()
        manifest_runs.append({
            "experiment": "E1_LongTail",
            "key": m_id,
            "name": f"lab2_e1_cifar100_{m_id}_3seed",
            "run_id": rid,
            "url": url,
        })
        print(f"[W&B Synced] {m_id:20s} -> {rid} ({url})")

    # 2. Sync E3A Quadratic O(D^2) Complexity Scaling Run
    dims = sorted({r["projection_dim"] for r in e3["scaling_runs"]})
    run_scale = wandb.init(
        entity=args.entity,
        project=args.project,
        name="lab2_e3a_cifar100_quadratic_scaling_3seed",
        group="lab2_e3_scaling_cifar100",
        job_type="lab2_empirical_demo",
        tags=["lab2-empirical-demo", "critical-analysis", "e3-quadratic-scaling", "cifar100"],
        config={
            "lab": "Lab2_Empirical_Demonstration",
            "experiment": "E3A_Quadratic_Complexity_Scaling",
            "dataset": "cifar100",
            "scaling_dims": dims,
            "empirical_scaling_exponent_alpha": e3["empirical_scaling_exponent_alpha"],
            "source_json_sha256": e3_sha,
            "provenance": e3["provenance"],
            "environment": env_info,
        },
        reinit=True,
    )
    for d_val in dims:
        sub = [r for r in e3["scaling_runs"] if r["projection_dim"] == d_val]
        t_mean = sum(x["rls_update_seconds"] for x in sub) / len(sub)
        a_mean = sum(x["average_accuracy"] for x in sub) / len(sub)
        run_scale.log(
            {
                "projection_dim": d_val,
                "rls_update_seconds_mean": round(t_mean, 4),
                "r_matrix_mb": sub[0]["r_matrix_mb"],
                "average_accuracy_mean": round(a_mean, 4),
            },
            step=int(d_val),
        )
    run_scale.summary["empirical_scaling_exponent_alpha"] = e3["empirical_scaling_exponent_alpha"]
    scale_url, scale_id = run_scale.url, run_scale.id
    run_scale.finish()
    manifest_runs.append({
        "experiment": "E3A_Quadratic_Scaling",
        "key": "quadratic_scaling",
        "name": "lab2_e3a_cifar100_quadratic_scaling_3seed",
        "run_id": scale_id,
        "url": scale_url,
    })
    print(f"[W&B Synced] E3A Scaling          -> {scale_id} ({scale_url})")

    # 3. Sync E3B (D x gamma) Double-Descent & FD2 Adaptive Ridge Runs (v50000 & v1600)
    for regime in ("native", "small_sample_v1600"):
        sub_g = [r for r in e3["grid_runs"] if r["regime"] == regime]
        run_g = wandb.init(
            entity=args.entity,
            project=args.project,
            name=f"lab2_e3b_cifar100_{regime}_double_descent_fd2",
            group="lab2_e3_scaling_cifar100",
            job_type="lab2_empirical_demo",
            tags=["lab2-empirical-demo", "critical-analysis", "e3-double-descent", "cifar100", regime],
            config={
                "lab": "Lab2_Empirical_Demonstration",
                "experiment": f"E3B_DoubleDescent_Grid_{regime}",
                "dataset": "cifar100",
                "regime": regime,
                "total_train_samples": sub_g[0]["total_train_samples"],
                "source_json_sha256": e3_sha,
                "provenance": e3["provenance"],
                "environment": env_info,
            },
            reinit=True,
        )
        for idx, row in enumerate(sub_g, start=1):
            run_g.log(
                {
                    "projection_dim": row["projection_dim"],
                    "resolved_gamma": row["resolved_gamma"],
                    "average_accuracy": row["average_accuracy"],
                    "last_accuracy": row["last_accuracy"],
                    "w_frobenius_norm": row["w_frobenius_norm"],
                },
                step=idx,
            )
        fd2_d4000 = next(
            (x for x in sub_g if x["projection_dim"] == 4000 and x["gamma_policy"] == "trace_auto"),
            None,
        )
        dd_d2000 = next(
            (x for x in sub_g if x["projection_dim"] == 2000 and x["gamma_policy"] == "fixed_0.001"),
            None,
        )
        if fd2_d4000:
            run_g.summary["fd2_d4000_average_accuracy"] = fd2_d4000["average_accuracy"]
            run_g.summary["fd2_d4000_gamma_auto"] = fd2_d4000["resolved_gamma"]
        if dd_d2000:
            run_g.summary["d2000_gamma1e3_last_accuracy"] = dd_d2000["last_accuracy"]
            run_g.summary["d2000_gamma1e3_w_frobenius_norm"] = dd_d2000["w_frobenius_norm"]
        g_url, g_id = run_g.url, run_g.id
        run_g.finish()
        manifest_runs.append({
            "experiment": "E3B_DoubleDescent_FD2",
            "key": regime,
            "name": f"lab2_e3b_cifar100_{regime}_double_descent_fd2",
            "run_id": g_id,
            "url": g_url,
        })
        print(f"[W&B Synced] E3B {regime:18s} -> {g_id} ({g_url})")

    manifest_path = lab2_root / "wandb_lab2_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entity": args.entity,
                "project": args.project,
                "required_tags": ["lab2-empirical-demo", "critical-analysis"],
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[W&B Sync Complete] Wrote {len(manifest_runs)} live runs to {manifest_path}")


if __name__ == "__main__":
    main()
