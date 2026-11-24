"""Shared dataset, cached-feature, and W&B tracking helpers for Lab 2 Empirical Demonstration.

Ensures every Lab 2 experiment:
1. Works seamlessly across CIFAR-100, DTD, and FGVC-Aircraft via programmatic Hugging Face
   loaders or validated pre-extracted ViT-B/16 feature caches.
2. Enforces strict RNG isolation across class order, SmoothProjection, and mini-batch stream.
3. Logs to the shared public W&B project (`hcmus-foal-reproduction`) with mandatory Lab 2 tags
   (`['lab2-empirical-demo', 'critical-analysis', ...]`) when `--wandb` is enabled.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.foal.backbone import FrozenViT
from src.foal.dataset_adapters.fgvc_aircraft import load_fgvc_aircraft
from src.foal.datasets import IndexedImageDataset, cifar100_transforms, load_cifar100, make_class_order
from src.foal.dtd_datasets import load_dtd, make_dtd_class_order
from src.foal.environment import get_environment_info

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CIFAR_C2_CACHE = REPO_ROOT / "results" / "c2_full_gpu_verified" / "results" / "c2_feature_cache"
DEFAULT_WANDB_PROJECT = "hcmus-foal-reproduction"
LAB2_REQUIRED_TAGS = ("lab2-empirical-demo", "critical-analysis")


def load_cifar100_from_c2_cache(cache_dir: Path = DEFAULT_CIFAR_C2_CACHE) -> dict[str, Any]:
    """Reconstruct full 50,000 train and 10,000 test CIFAR-100 ViT-B/16 embeddings with original labels.

    The C2 verified cache stores all 10 tasks of CIFAR-100 extracted under ``seed=42`` class order
    with ``vit_base_patch16_224.augreg2_in21k_ft_in1k`` and ``feature_fusion=True``. Because
    ``FrozenViT`` has no cross-sample state (LayerNorm + eval mode), mapping contiguous labels
    back through ``order_42`` recovers exact per-image embeddings usable under ANY class-order seed.

    Args:
        cache_dir: Directory containing ``train_vit_base_patch16_224_augreg2_in21k_ft_in1k_c42_s42_t01..t10_trfull.pt``
            and ``test_..._tefull.pt``.

    Returns:
        Dictionary containing ``train_features`` ``(50000, 768)``, ``train_orig_labels`` ``(50000,)``,
        ``test_features`` ``(10000, 768)``, ``test_orig_labels`` ``(10000,)``, and provenance metadata.
    """
    order_42 = make_class_order(100, seed=42)
    contig_to_orig = torch.tensor(order_42, dtype=torch.long)

    tr_feats, tr_orig = [], []
    te_feats, te_orig = [], []
    meta_sample: dict[str, Any] = {}

    for t_idx in range(1, 11):
        tr_file = cache_dir / f"train_vit_base_patch16_224_augreg2_in21k_ft_in1k_c42_s42_t{t_idx:02d}_trfull.pt"
        te_file = cache_dir / f"test_vit_base_patch16_224_augreg2_in21k_ft_in1k_c42_s42_t{t_idx:02d}_tefull.pt"
        if not tr_file.exists() or not te_file.exists():
            raise FileNotFoundError(f"Missing cached CIFAR-100 feature shard at task {t_idx}: {tr_file}")

        tr_blob = torch.load(tr_file, map_location="cpu", weights_only=False)
        te_blob = torch.load(te_file, map_location="cpu", weights_only=False)
        if t_idx == 1:
            meta_sample = {
                "backbone_name": tr_blob.get("backbone_name"),
                "pretrained_tag": tr_blob.get("pretrained_tag"),
                "feature_fusion": tr_blob.get("feature_fusion"),
                "dataset_name": tr_blob.get("dataset_name"),
                "preprocessing": tr_blob.get("preprocessing"),
                "source": "verified_c2_feature_cache",
            }
        tr_feats.append(tr_blob["features"].to(torch.float32))
        tr_orig.append(contig_to_orig[tr_blob["labels"].to(torch.long)])
        te_feats.append(te_blob["features"].to(torch.float32))
        te_orig.append(contig_to_orig[te_blob["labels"].to(torch.long)])

    train_features = torch.cat(tr_feats, dim=0)
    train_orig_labels = torch.cat(tr_orig, dim=0)
    test_features = torch.cat(te_feats, dim=0)
    test_orig_labels = torch.cat(te_orig, dim=0)

    return {
        "dataset": "cifar100",
        "num_classes": 100,
        "classes_per_task": 10,
        "max_samples_per_class": 500,
        "train_features": train_features,
        "train_orig_labels": train_orig_labels,
        "test_features": test_features,
        "test_orig_labels": test_orig_labels,
        "provenance": meta_sample,
    }


def extract_or_load_backbone_features(
    dataset: str,
    *,
    backbone_name: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    feature_fusion: bool = True,
    device: str = "cpu",
    batch_size: int = 64,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Load pre-extracted ViT features from cache or extract them programmatically from Hugging Face.

    Args:
        dataset: One of ``'cifar100'``, ``'dtd'``, or ``'aircraft'``.
        backbone_name: ``timm`` model identifier.
        feature_fusion: Whether 12-block CLS averaging is enabled.
        device: Torch compute device string (``'cpu'`` or ``'cuda'``).
        batch_size: Extraction batch size.
        cache_dir: Optional directory for storing/reading extracted ``.pt`` feature bundles.

    Returns:
        Feature bundle dictionary matching ``load_cifar100_from_c2_cache``.
    """
    ds = dataset.lower().strip()
    if ds == "cifar100" and cache_dir is None and DEFAULT_CIFAR_C2_CACHE.exists():
        try:
            return load_cifar100_from_c2_cache(DEFAULT_CIFAR_C2_CACHE)
        except FileNotFoundError:
            pass

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_bb = backbone_name.replace("/", "_").replace(".", "_")
        bundle_path = cache_dir / f"lab2_bundle_{ds}_{safe_bb}_ff{int(feature_fusion)}.pt"
        if bundle_path.exists():
            return torch.load(bundle_path, map_location="cpu", weights_only=False)
    else:
        bundle_path = None

    if ds == "cifar100":
        train_src, test_src = load_cifar100()
        tfm = cifar100_transforms(image_size=224, interpolation="bilinear", normalize=False)
        tr_ds = IndexedImageDataset(train_src, list(range(len(train_src))), list(train_src.fine_label), tfm)
        te_ds = IndexedImageDataset(test_src, list(range(len(test_src))), list(test_src.fine_label), tfm)
        num_classes, classes_per_task, n_max = 100, 10, 500
    elif ds == "dtd":
        train_src, test_src = load_dtd()
        tfm = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        tr_ds = IndexedImageDataset(train_src, list(range(len(train_src))), list(train_src.fine_label), tfm)
        te_ds = IndexedImageDataset(test_src, list(range(len(test_src))), list(test_src.fine_label), tfm)
        num_classes, classes_per_task, n_max = 40, 4, 40
    elif ds == "aircraft":
        train_src, test_src = load_fgvc_aircraft()
        tfm = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
        tr_ds = IndexedImageDataset(train_src, list(range(len(train_src))), list(train_src.fine_label), tfm)
        te_ds = IndexedImageDataset(test_src, list(range(len(test_src))), list(test_src.fine_label), tfm)
        num_classes, classes_per_task, n_max = 100, 10, 34
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'cifar100', 'dtd', or 'aircraft'.")

    encoder = FrozenViT(backbone_name, pretrained=True, feature_fusion=feature_fusion).to(device)
    encoder.eval()

    def _run_extract(img_ds: IndexedImageDataset) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(img_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        f_list, y_list = [], []
        with torch.no_grad():
            for xb, yb in loader:
                f_list.append(encoder(xb.to(device)).cpu().to(torch.float32))
                y_list.append(yb.cpu().to(torch.long))
        return torch.cat(f_list, dim=0), torch.cat(y_list, dim=0)

    tr_f, tr_y = _run_extract(tr_ds)
    te_f, te_y = _run_extract(te_ds)

    bundle = {
        "dataset": ds,
        "num_classes": num_classes,
        "classes_per_task": classes_per_task,
        "max_samples_per_class": n_max,
        "train_features": tr_f,
        "train_orig_labels": tr_y,
        "test_features": te_f,
        "test_orig_labels": te_y,
        "provenance": {
            "backbone_name": backbone_name,
            "feature_fusion": feature_fusion,
            "source": "live_hf_extraction",
            "train_sha256": hashlib.sha256(tr_f.contiguous().numpy().tobytes()).hexdigest(),
        },
    }
    if bundle_path is not None:
        torch.save(bundle, bundle_path)
    return bundle


def make_dataset_class_order(dataset: str, seed: int) -> list[int]:
    """Generate the canonical incremental class order for a dataset and seed.

    Args:
        dataset: ``'cifar100'``, ``'dtd'``, or ``'aircraft'``.
        seed: Integer class-order seed.

    Returns:
        Ordered list of selected original class IDs.
    """
    ds = dataset.lower().strip()
    if ds == "dtd":
        return make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=seed)
    return make_class_order(100, seed=seed)


def build_remapped_pools(
    train_orig_labels: torch.Tensor,
    test_orig_labels: torch.Tensor,
    class_order: list[int],
    *,
    max_train_per_class: int | None = None,
    seed: int = 42,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Map original class labels onto contiguous incremental IDs ``0..C-1`` and group indices.

    For DTD (where Hugging Face ``train`` split has 80 images/class while the paper protocol
    uses 40 images/class), ``max_train_per_class=40`` deterministically caps each class pool.

    Args:
        train_orig_labels: 1D tensor of original training labels.
        test_orig_labels: 1D tensor of original test labels.
        class_order: Sequence of original class IDs defining contiguous IDs ``0..C-1``.
        max_train_per_class: Optional cap on training pool size per class.
        seed: Deterministic seed when capping class pools.

    Returns:
        Tuple ``(train_pool_by_contig, test_pool_by_contig)``.
    """
    tr_np = train_orig_labels.numpy()
    te_np = test_orig_labels.numpy()
    tr_pool: dict[int, list[int]] = {}
    te_pool: dict[int, list[int]] = {}

    for contig_id, orig_cls in enumerate(class_order):
        tr_idx = np.flatnonzero(tr_np == int(orig_cls)).tolist()
        if max_train_per_class is not None and len(tr_idx) > max_train_per_class:
            rng = np.random.default_rng(int(seed) * 9973 + int(contig_id))
            tr_idx = sorted(rng.choice(tr_idx, size=int(max_train_per_class), replace=False).tolist())
        te_idx = np.flatnonzero(te_np == int(orig_cls)).tolist()
        tr_pool[contig_id] = tr_idx
        te_pool[contig_id] = te_idx
    return tr_pool, te_pool


def init_lab2_wandb_run(
    *,
    enabled: bool,
    offline: bool,
    project: str,
    entity: str | None,
    name: str,
    group: str,
    extra_tags: list[str],
    config: dict[str, Any],
) -> Any:
    """Initialize a W&B run strictly tagged for Lab 2 (`lab2-empirical-demo`) if enabled.

    Args:
        enabled: Whether to initialize W&B logging.
        offline: Whether to use W&B offline mode.
        project: Target W&B project (default ``hcmus-foal-reproduction``).
        entity: Optional W&B entity.
        name: Run display name.
        group: W&B run group (prefixed with ``lab2_``).
        extra_tags: Experiment- and dataset-specific tags appended to ``LAB2_REQUIRED_TAGS``.
        config: Full hyperparameter and environment metadata dictionary.

    Returns:
        Active ``wandb.Run`` object, or ``None`` when ``enabled`` is False.
    """
    if not enabled:
        return None
    import wandb

    tags = list(dict.fromkeys([*LAB2_REQUIRED_TAGS, *extra_tags]))
    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type="lab2_empirical_demo",
        tags=tags,
        mode="offline" if offline else "online",
        config={**config, "environment": get_environment_info()},
        reinit=True,
    )
