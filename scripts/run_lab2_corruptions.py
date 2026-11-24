"""Lab 2 Experiment E2: Sensor & Environmental Covariate-Shift Stress-Test & FD2 Stabilization.

Tests Hypothesis H2a (`docs/CRITICAL_ANALYSIS_REPORT.md` Section 4.2 & 5.2):
Evaluates F-OAL on corrupted image streams (`gaussian_noise`, `gaussian_blur`, `jpeg_compression`,
`contrast`) across severities `s in {0, 1, 2, 3, 4, 5}`, measuring:
1. Cumulative average accuracy `A_bar(s)` and final accuracy `A_10(s)`.
2. Frozen ViT feature cosine similarity drift `cos(h_clean, h_corrupt)` and norm shift.
3. Vanilla F-OAL (`gamma = 1.0`) vs FD2 (`pre_l2_normalize` + `trace_scaled_gamma`).

Supports full dataset execution on GPU (Colab/Kaggle) as well as bounded `--limit-train-per-class`
/ `--limit-test-per-class` execution for fast CPU verification.

Usage:
    python -m scripts.run_lab2_corruptions --dataset dtd --device cuda --wandb
    python -m scripts.run_lab2_corruptions --dataset cifar100 --severities 0 2 4 --limit-train-per-class 20 --limit-test-per-class 10
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lab2_common import DEFAULT_WANDB_PROJECT, init_lab2_wandb_run, make_dataset_class_order
from src.foal.backbone import FrozenViT
from src.foal.critical_analysis import (
    apply_image_corruption,
    compute_trace_scaled_gamma,
    pre_l2_normalize,
)
from src.foal.dataset_adapters.fgvc_aircraft import load_fgvc_aircraft
from src.foal.datasets import load_cifar100
from src.foal.dtd_datasets import load_dtd
from src.foal.metrics import summary_from_task_matrix
from src.foal.projection import SmoothProjection
from src.foal.rls import OnlineRidgeRLS


class _RawTensorSubset(Dataset):
    """Lightweight dataset resizing PIL images to `[0, 1]` tensors prior to corruption."""

    def __init__(self, source: Any, indices: list[int], contig_labels: list[int], interp: transforms.InterpolationMode):
        self.source = source
        self.indices = indices
        self.labels = contig_labels
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224), interpolation=interp),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        raw_idx = self.indices[item]
        img = self.source[raw_idx]
        if not hasattr(img, "convert"):
            img = img["image"] if isinstance(img, dict) and "image" in img else img[0]
        return self.tfm(img.convert("RGB")), int(self.labels[item]), int(raw_idx)


def load_raw_sources(dataset: str) -> tuple[Any, Any, int, int, transforms.InterpolationMode]:
    """Load programmatic Hugging Face dataset split sources and official resize mode."""
    ds = dataset.lower().strip()
    if ds == "cifar100":
        tr, te = load_cifar100()
        return tr, te, 100, 10, transforms.InterpolationMode.BILINEAR
    if ds == "dtd":
        tr, te = load_dtd()
        return tr, te, 40, 4, transforms.InterpolationMode.BILINEAR
    if ds == "aircraft":
        tr, te = load_fgvc_aircraft()
        return tr, te, 100, 10, transforms.InterpolationMode.LANCZOS
    raise ValueError(f"Unsupported dataset: {dataset}")


def extract_corrupted_features(
    encoder: FrozenViT,
    subset: _RawTensorSubset,
    *,
    corruption: str,
    severity: int,
    seed: int,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen ViT features after applying deterministic corruption per batch."""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, lbls = [], []
    with torch.no_grad():
        for b_idx, (xb, yb, _) in enumerate(loader):
            xb_corr = apply_image_corruption(xb, corruption, severity, seed=int(seed) * 10007 + b_idx)
            feats.append(encoder(xb_corr.to(device)).cpu().to(torch.float32))
            lbls.append(yb.to(torch.long))
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)


def evaluate_ocil_from_features(
    tr_f: torch.Tensor,
    tr_y: torch.Tensor,
    te_f: torch.Tensor,
    te_y: torch.Tensor,
    *,
    num_classes: int,
    cpt: int,
    seed: int,
    use_fd2: bool,
    projection_dim: int = 1000,
    batch_size: int = 25,
) -> dict[str, float]:
    """Run 10-task OCIL on extracted features comparing Vanilla F-OAL vs FD2 stabilization."""
    num_tasks = num_classes // cpt
    if use_fd2:
        tr_in = pre_l2_normalize(tr_f, scale_sqrt_dim=True)
        te_in = pre_l2_normalize(te_f, scale_sqrt_dim=True)
    else:
        tr_in, te_in = tr_f, te_f

    proj = SmoothProjection(tr_in.shape[1], projection_dim, seed=seed, initialization="official_uniform")
    with torch.no_grad():
        tr_z = proj(tr_in)
        te_z = proj(te_in)

    t1_mask = tr_y < cpt
    if use_fd2:
        gamma = compute_trace_scaled_gamma(
            tr_z[t1_mask], gamma_0=2.5, total_expected_samples=len(tr_y), ref_dim=1000
        )
    else:
        gamma = 1.0

    clf = OnlineRidgeRLS(projection_dim, gamma=gamma, dtype=torch.float64)
    matrix: list[list[float]] = []

    for stage in range(num_tasks):
        lo, hi = stage * cpt, (stage + 1) * cpt
        clf.expand_classes(hi)
        mask_tr = (tr_y >= lo) & (tr_y < hi)
        zs, ys = tr_z[mask_tr], tr_y[mask_tr]
        for start in range(0, len(ys), batch_size):
            clf.update(zs[start:start + batch_size], ys[start:start + batch_size])

        stage_row: list[float] = []
        for eval_t in range(stage + 1):
            elo, ehi = eval_t * cpt, (eval_t + 1) * cpt
            mask_te = (te_y >= elo) & (te_y < ehi)
            preds = clf.predict(te_z[mask_te])
            stage_row.append(float((preds == te_y[mask_te]).to(torch.float64).mean().item()))
        matrix.append(stage_row)

    inc = summary_from_task_matrix(matrix)
    return {
        "average_accuracy": round(float(inc["average_incremental_accuracy"]) * 100.0, 4),
        "last_accuracy": round(float(inc["last_accuracy"]) * 100.0, 4),
        "forgetting_past": round(float(inc["forgetting"]) * 100.0, 4),
        "resolved_gamma": round(float(gamma), 4),
    }


def main() -> None:
    """CLI entry point for Lab 2 Experiment E2 (Covariate-Shift / Corruptions)."""
    parser = argparse.ArgumentParser(description="Run Lab 2 E2 Sensor Corruption Stress-Test.")
    parser.add_argument("--dataset", default="dtd", choices=["cifar100", "dtd", "aircraft"])
    parser.add_argument("--corruptions", nargs="+", default=["gaussian_noise", "gaussian_blur", "jpeg_compression"])
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    parser.add_argument("--limit-train-per-class", type=int, default=None)
    parser.add_argument("--limit-test-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/lab2_empirical/e2_corruptions")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_src, te_src, num_classes, cpt, interp = load_raw_sources(args.dataset)
    order = make_dataset_class_order(args.dataset, seed=args.seed)

    tr_np = np.asarray(tr_src.fine_label, dtype=np.int64)
    te_np = np.asarray(te_src.fine_label, dtype=np.int64)
    default_cap = 40 if args.dataset == "dtd" else None
    tr_cap = args.limit_train_per_class if args.limit_train_per_class is not None else default_cap
    te_cap = args.limit_test_per_class

    tr_indices, tr_contig = [], []
    te_indices, te_contig = [], []
    rng = np.random.default_rng(args.seed)

    for c_id, orig_cls in enumerate(order):
        ti = np.flatnonzero(tr_np == int(orig_cls)).tolist()
        if tr_cap is not None and len(ti) > tr_cap:
            ti = sorted(rng.choice(ti, size=tr_cap, replace=False).tolist())
        tr_indices.extend(ti)
        tr_contig.extend([c_id] * len(ti))

        ei = np.flatnonzero(te_np == int(orig_cls)).tolist()
        if te_cap is not None and len(ei) > te_cap:
            ei = sorted(rng.choice(ei, size=te_cap, replace=False).tolist())
        te_indices.extend(ei)
        te_contig.extend([c_id] * len(ei))

    tr_sub = _RawTensorSubset(tr_src, tr_indices, tr_contig, interp)
    te_sub = _RawTensorSubset(te_src, te_indices, te_contig, interp)

    print(f"[Lab2-E2] Initializing FrozenViT '{args.backbone}' on {device} (N_tr={len(tr_sub)}, N_te={len(te_sub)})...")
    encoder = FrozenViT(args.backbone, pretrained=True, feature_fusion=True).to(device)
    encoder.eval()

    print("[Lab2-E2] Extracting clean reference features (severity=0)...")
    clean_tr_f, clean_tr_y = extract_corrupted_features(
        encoder, tr_sub, corruption="clean", severity=0, seed=args.seed, device=device, batch_size=args.batch_size
    )
    clean_te_f, clean_te_y = extract_corrupted_features(
        encoder, te_sub, corruption="clean", severity=0, seed=args.seed, device=device, batch_size=args.batch_size
    )

    records: list[dict[str, Any]] = []
    for corr in args.corruptions:
        for sev in args.severities:
            if sev == 0:
                cur_tr_f, cur_te_f = clean_tr_f, clean_te_f
                cos_sim = 1.0
            else:
                cur_tr_f, _ = extract_corrupted_features(
                    encoder, tr_sub, corruption=corr, severity=sev, seed=args.seed, device=device, batch_size=args.batch_size
                )
                cur_te_f, _ = extract_corrupted_features(
                    encoder, te_sub, corruption=corr, severity=sev, seed=args.seed + 1, device=device, batch_size=args.batch_size
                )
                cos_sim = float(F.cosine_similarity(clean_te_f, cur_te_f, dim=-1).mean().item())

            for use_fd2, m_id in ((False, "vanilla_foal"), (True, "fd2_l2norm_adaptive")):
                # Scenario 1: Clean-trained classifier evaluated on corrupted test stream (pure covariate shift)
                shift_res = evaluate_ocil_from_features(
                    clean_tr_f, clean_tr_y, cur_te_f, clean_te_y,
                    num_classes=num_classes, cpt=cpt, seed=args.seed, use_fd2=use_fd2
                )
                # Scenario 2: Corrupted online training + corrupted test stream
                stream_res = evaluate_ocil_from_features(
                    cur_tr_f, clean_tr_y, cur_te_f, clean_te_y,
                    num_classes=num_classes, cpt=cpt, seed=args.seed, use_fd2=use_fd2
                )
                rec = {
                    "dataset": args.dataset,
                    "corruption": corr,
                    "severity": int(sev),
                    "method": m_id,
                    "seed": int(args.seed),
                    "feature_cosine_similarity": round(cos_sim, 5),
                    "test_shift_avg_acc": shift_res["average_accuracy"],
                    "test_shift_last_acc": shift_res["last_accuracy"],
                    "corrupted_stream_avg_acc": stream_res["average_accuracy"],
                    "corrupted_stream_last_acc": stream_res["last_accuracy"],
                    "resolved_gamma": stream_res["resolved_gamma"],
                }
                records.append(rec)
                print(
                    f"  {corr:16s} | s={sev} | {m_id:20s} | cos={cos_sim:.4f} -> "
                    f"Shift_A_bar={rec['test_shift_avg_acc']:6.2f}% | Stream_A_bar={rec['corrupted_stream_avg_acc']:6.2f}%"
                )

                wb_run = init_lab2_wandb_run(
                    enabled=args.wandb,
                    offline=args.wandb_offline,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=f"lab2_e2_{args.dataset}_{corr}_s{sev}_{m_id}",
                    group=f"lab2_e2_corruption_{args.dataset}",
                    extra_tags=["e2-corruption", args.dataset, corr, m_id],
                    config=rec,
                )
                if wb_run is not None:
                    wb_run.log(rec)
                    wb_run.finish()

    payload = {
        "experiment": "Lab2_E2_Covariate_Shift_Corruptions",
        "dataset": args.dataset,
        "seed": args.seed,
        "num_train_samples": len(tr_sub),
        "num_test_samples": len(te_sub),
        "runs": records,
    }
    (out_dir / "e2_corruptions_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if records:
        with (out_dir / "e2_corruptions_runs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    print(f"[Lab2-E2] Saved summary to {out_dir / 'e2_corruptions_summary.json'}")


if __name__ == "__main__":
    main()
