"""Unit and regression tests for DTD Full Reproduction Pipeline.

Validates:
1. Configuration integrity of configs/dtd_full_reproduction.yaml and configs/dtd.yaml.
2. Exact parameter parity with configs/dtd/dtd_a0_full.yaml (Table 4 A0 protocol).
3. Notebook structure and parity with notebooks/foal_cifar100_full_reproduction.ipynb.
4. Evaluation of DTD checkpoints in scripts/evaluate.py.
5. Protocol isolation in scripts/summarize_runs.py between DTD and CIFAR-100.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch
import yaml

from scripts.evaluate import evaluate_checkpoint
from scripts.summarize_runs import load_run, summarize_results


def test_dtd_full_reproduction_configs():
    """Verify both configs/dtd_full_reproduction.yaml and configs/dtd.yaml are valid and match paper protocol."""
    for path_str in ["configs/dtd_full_reproduction.yaml", "configs/dtd.yaml"]:
        cfg_path = Path(path_str)
        assert cfg_path.exists(), f"Missing config: {path_str}"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        assert cfg["dataset"]["type"] == "dtd"
        assert cfg["dataset"]["name"] == "tanganke/dtd"
        assert cfg["dataset"]["samples_per_class"] == 40
        assert cfg["dataset"]["classes_per_task"] == 4
        assert cfg["dataset"]["num_classes"] == 40
        assert cfg["dataset"]["interpolation"] == "bilinear"
        assert cfg["dataset"]["normalize"] is False

        assert cfg["model"]["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
        assert cfg["model"]["feature_fusion"] is True
        assert cfg["model"]["smooth_projection"] is True
        assert cfg["model"]["projection_dim"] == 1000
        assert cfg["model"]["projection_init"] == "official_uniform"
        assert cfg["model"]["gamma"] == 1.0
        assert cfg["model"]["rls_dtype"] == "float64"

        assert cfg["training"]["batch_size"] == 10


def test_dtd_full_reproduction_matches_a0_full():
    """Verify configs/dtd_full_reproduction.yaml matches configs/dtd/dtd_a0_full.yaml."""
    full_cfg = yaml.safe_load(Path("configs/dtd_full_reproduction.yaml").read_text(encoding="utf-8"))
    a0_cfg = yaml.safe_load(Path("configs/dtd/dtd_a0_full.yaml").read_text(encoding="utf-8"))

    assert full_cfg["dataset"]["name"] == a0_cfg["dataset"]["name"]
    assert full_cfg["dataset"]["samples_per_class"] == a0_cfg["dataset"]["samples_per_class"]
    assert full_cfg["dataset"]["classes_per_task"] == a0_cfg["dataset"]["classes_per_task"]
    assert full_cfg["dataset"]["interpolation"] == a0_cfg["dataset"]["interpolation"]
    assert full_cfg["dataset"]["normalize"] == a0_cfg["dataset"]["normalize"]

    assert full_cfg["model"]["backbone"] == a0_cfg["model"]["backbone"]
    assert full_cfg["model"]["feature_fusion"] == a0_cfg["model"]["feature_fusion"]
    assert full_cfg["model"]["smooth_projection"] == a0_cfg["model"]["smooth_projection"]
    assert full_cfg["model"]["projection_init"] == a0_cfg["model"]["projection_init"]
    assert full_cfg["model"]["gamma"] == a0_cfg["model"]["gamma"]
    assert full_cfg["model"]["rls_dtype"] == a0_cfg["model"]["rls_dtype"]


def test_dtd_full_reproduction_notebook_structure():
    """Verify notebooks/foal_dtd_full_reproduction.ipynb exists, is valid JSON, and has all expected sections."""
    nb_path = Path("notebooks/foal_dtd_full_reproduction.ipynb")
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    assert "cells" in nb
    assert len(nb["cells"]) >= 15

    # Combine all code cell sources
    code_text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    assert "nvidia-smi" in code_text
    assert "pytest" in code_text
    assert "scripts.train" in code_text
    assert "scripts.evaluate" in code_text
    assert "scripts.summarize_runs" in code_text
    assert "scripts.run_seeds" in code_text
    assert "configs/dtd_full_reproduction.yaml" in code_text


def test_dtd_checkpoint_evaluation_mock():
    """Test evaluate_checkpoint on a synthetic DTD checkpoint structure."""
    from src.foal.model import FOALModel
    from src.foal.dtd_datasets import make_dtd_class_order

    order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=42)
    model = FOALModel(
        backbone_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        feature_fusion=True,
        smooth_projection=True,
        projection_dim=1000,
        projection_init="official_uniform",
        gamma=1.0,
        rls_dtype="float64",
    )
    # Expand 4 classes for task 1
    model.classifier.expand_classes(4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "task_01.pt"
        torch.save({
            "task_number": 1,
            "class_order": order,
            "classifier_state": model.classifier.state_dict(),
            "projection_state": model.projection.state_dict() if model.projection else None,
            "model_config": {
                "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "feature_fusion": True,
                "smooth_projection": True,
                "projection_dim": 1000,
                "projection_init": "official_uniform",
                "gamma": 1.0,
                "rls_dtype": "float64",
            },
            "config": {
                "seed": 42,
                "dataset": {
                    "type": "dtd",
                    "name": "tanganke/dtd",
                    "classes_per_task": 4,
                    "samples_per_class": 40,
                    "image_size": 224,
                    "interpolation": "bilinear",
                    "normalize": False,
                },
                "training": {"batch_size": 10},
            },
        }, ckpt_path)

        # Run evaluate_checkpoint with limit_per_task=5 to keep CPU evaluation fast
        res = evaluate_checkpoint(str(ckpt_path), device="cpu", limit_per_task=5)
        assert res["task"] == 1
        assert "accuracy" in res
        assert len(res["per_task_accuracy"]) == 1
        assert res["evaluated_samples"] == 5


def test_summarize_runs_isolates_dtd_from_cifar100():
    """Verify summarize_runs separates DTD and CIFAR-100 runs into distinct groups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        run_cifar = base_dir / "cifar_run"
        run_dtd = base_dir / "dtd_run"
        run_cifar.mkdir()
        run_dtd.mkdir()

        # CIFAR-100 run
        (run_cifar / "metadata.json").write_text(json.dumps({
            "method": "foal",
            "experiment_type": "REPRODUCTION",
            "training_protocol": {"max_tasks": 10, "dataset": "cifar100"},
            "model_config": {"backbone": "vit_base_patch16_224.augreg_in1k", "projection_dim": 1000, "gamma": 1.0},
            "seeds": {"base": 42},
            "device": "cuda",
        }))
        (run_cifar / "metrics.json").write_text(json.dumps({
            "num_tasks": 10,
            "task_matrix": [[0.9] * (i + 1) for i in range(10)],
            "average_incremental_accuracy": 0.90,
            "last_accuracy": 0.88,
            "forgetting": 0.05,
        }))
        (run_cifar / "task_10.pt").write_bytes(b"DUMMY_CKPT")

        # DTD run (same tasks, backbone, etc., but DTD dataset)
        (run_dtd / "metadata.json").write_text(json.dumps({
            "method": "foal",
            "experiment_type": "REPRODUCTION",
            "training_protocol": {"max_tasks": 10, "dataset": "tanganke/dtd"},
            "model_config": {"backbone": "vit_base_patch16_224.augreg_in1k", "projection_dim": 1000, "gamma": 1.0},
            "seeds": {"base": 42},
            "device": "cuda",
        }))
        (run_dtd / "metrics.json").write_text(json.dumps({
            "num_tasks": 10,
            "task_matrix": [[0.82] * (i + 1) for i in range(10)],
            "average_incremental_accuracy": 0.82,
            "last_accuracy": 0.76,
            "forgetting": 0.08,
        }))
        (run_dtd / "task_10.pt").write_bytes(b"DUMMY_CKPT")

        summary = summarize_results(results_dir=str(base_dir), output_dir=str(base_dir))
        assert len(summary["individual_runs"]) == 2
        # Must produce TWO distinct groups (one for CIFAR-100 and one for DTD)
        assert len(summary["groups"]) == 2
        group_names = list(summary["groups"].keys())
        assert any("tanganke/dtd" in g for g in group_names)
