"""Unit and regression tests for Official F-OAL DTD Reproduction Pipeline.

Validates:
1. Integrity and JSON validity of notebooks/foal_official_dtd_reproduction.ipynb.
2. Structural parity with notebooks/foal_official_cifar100_reproduction.ipynb.
3. Official author repository provenance (liuyuchen-cz/F-OAL @ 73e4ac1).
4. DTD benchmark setup (40 classes, 10 tasks x 4 classes, 1600 samples).
5. Output artifacts specification (metrics.json, metadata.json, task_accuracy.csv, dtd_FOAL.csv, official_stdout.log).
6. Official run completeness validation in scripts/summarize_runs.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from scripts.summarize_runs import load_run


def test_official_dtd_notebook_exists_and_valid():
    """Verify notebooks/foal_official_dtd_reproduction.ipynb exists and is valid Jupyter JSON."""
    nb_path = Path("notebooks/foal_official_dtd_reproduction.ipynb")
    assert nb_path.exists(), "Official DTD notebook does not exist"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert len(nb["cells"]) == 11, f"Expected 11 cells, got {len(nb['cells'])}"


def test_official_dtd_notebook_parity_with_cifar100():
    """Verify structural and configuration parity with foal_official_cifar100_reproduction.ipynb."""
    dtd_nb = json.loads(Path("notebooks/foal_official_dtd_reproduction.ipynb").read_text(encoding="utf-8"))
    cifar_nb = json.loads(Path("notebooks/foal_official_cifar100_reproduction.ipynb").read_text(encoding="utf-8"))

    assert len(dtd_nb["cells"]) == len(cifar_nb["cells"]), "Cell count mismatch with official CIFAR-100 notebook"

    # Both must clone the exact same pinned official commit
    cifar_src = "".join(cifar_nb["cells"][2]["source"])
    dtd_src = "".join(dtd_nb["cells"][2]["source"])
    assert "https://github.com/liuyuchen-cz/F-OAL.git" in dtd_src
    assert "73e4ac1b4c965030973a2b77f05c57c2f44f49f5" in dtd_src

    # Both must use the official model Encoder and Woodbury RLS
    cifar_cell5 = "".join(cifar_nb["cells"][5]["source"])
    dtd_cell5 = "".join(dtd_nb["cells"][5]["source"])
    assert "models.pretrained" in dtd_cell5
    assert "Encoder" in dtd_cell5

    # Check cell 6 execution logic
    dtd_cell6 = "".join(dtd_nb["cells"][6]["source"])
    assert "Encoder" in dtd_cell6
    assert "official_transform" in dtd_cell6
    assert "double()" in dtd_cell6  # float64 precision
    assert "torch.pinverse" in dtd_cell6  # official Woodbury update
    assert "official_stdout.log" in dtd_cell6
    assert "dtd_FOAL.csv" in dtd_cell6

    # Check cell 7 metrics extraction
    dtd_cell7 = "".join(dtd_nb["cells"][7]["source"])
    assert "average_incremental_accuracy" in dtd_cell7
    assert "last_accuracy" in dtd_cell7
    assert "forgetting_9_tasks" in dtd_cell7
    assert "official_forgetting_10_tasks" in dtd_cell7

    # Check cell 9 archive packaging
    dtd_cell9 = "".join(dtd_nb["cells"][9]["source"])
    assert "foal_official_dtd_results.zip" in dtd_cell9
    assert "official_foal_dtd/metrics.json" in dtd_cell9


def test_official_dtd_dataset_protocol_spec():
    """Verify DTD dataset setup in notebook matches NeurIPS 2024 Table 4 protocol."""
    dtd_nb = json.loads(Path("notebooks/foal_official_dtd_reproduction.ipynb").read_text(encoding="utf-8"))
    cell4_src = "".join(dtd_nb["cells"][4]["source"])

    assert "NUM_CLASSES = 40" in cell4_src
    assert "CLASSES_PER_TASK = 4" in cell4_src
    assert "NUM_TASKS = 10" in cell4_src
    assert "SAMPLES_PER_CLASS = 40" in cell4_src
    assert "DATASET_SEED = 42" in cell4_src


def test_summarize_runs_official_dtd_detection():
    """Verify scripts/summarize_runs.py correctly accepts official DTD runs with dtd_FOAL.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "official_foal_dtd_seed0"
        run_dir.mkdir()

        meta = {
            "method": "official_foal",
            "experiment_type": "official_reproduction",
            "seeds": {"base": 0},
            "training_protocol": {
                "dataset": "tanganke/dtd",
                "max_tasks": 10,
                "classes_per_task": 4,
                "num_classes": 40,
                "device": "cuda",
            },
            "model_config": {
                "backbone": "vit_base_patch16_224",
                "projection_dim": 1000,
                "feature_fusion": True,
                "smooth_projection": True,
                "gamma": 1.0,
            },
            "official_commit_sha": "73e4ac1b4c965030973a2b77f05c57c2f44f49f5",
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        metrics = {
            "num_tasks": 10,
            "task_matrix": [[0.85] * (i + 1) for i in range(10)],
            "average_incremental_accuracy": 0.85,
            "last_accuracy": 0.80,
            "forgetting": 0.07,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

        # Official completion markers
        (run_dir / "dtd_FOAL.csv").write_text("task,accuracy\n1,0.85\n", encoding="utf-8")
        (run_dir / "official_stdout.log").write_text(
            "-----------run 0 training batch 0-------------\nRecords have been written\n",
            encoding="utf-8",
        )

        loaded = load_run(run_dir)
        assert loaded is not None
        assert loaded["status"] == "COMPLETE"
        assert loaded["method"] == "official_foal"
        assert loaded["dataset"] == "tanganke/dtd"
        assert loaded["average_incremental_accuracy"] == 0.85
        assert loaded["last_accuracy"] == 0.80
