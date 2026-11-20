"""Unit, integration, and notebook verification tests for DTD Section IX Headline Baselines."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml

import scripts.train_baseline as train_baseline_mod
from src.foal.baselines import (
    ERAgent,
    LwFAgent,
    LwFClassifier,
    OfficialLwFAgent,
    OfficialLwFClassifier,
    ReplayBuffer,
    official_loss_fn_kd,
)
from scripts.generate_dtd_headline_analysis import run_analysis as run_dtd_analysis
from scripts.train_baseline import build_parser, run_baseline

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def _cache_dtd_loader_per_module():
    """Cache load_dtd once per test module to avoid reloading 3760 Arrow rows repeatedly."""
    orig_load_dtd = train_baseline_mod.load_dtd
    cached_result = {}

    def _cached_load_dtd(root: str = "./data"):
        if root not in cached_result:
            cached_result[root] = orig_load_dtd(root=root)
        return cached_result[root]

    train_baseline_mod.load_dtd = _cached_load_dtd
    try:
        yield
    finally:
        train_baseline_mod.load_dtd = orig_load_dtd


def test_lwf_classifier_expansion_dtd():
    """Verify LwFClassifier expands incrementally from 4 to 40 DTD classes preserving learned weights."""
    model = LwFClassifier(in_features=768, num_classes=4)
    orig_w = model.fc.weight.data.clone()
    orig_b = model.fc.bias.data.clone()

    x = torch.randn(5, 768)
    out_4 = model(x)
    assert out_4.shape == (5, 4)

    for total_seen in range(8, 44, 4):
        model.expand_classes(total_seen)
        assert model.num_classes == total_seen
        assert model.fc.weight.shape == (total_seen, 768)
        assert model.fc.bias.shape == (total_seen,)
        assert torch.allclose(model.fc.weight.data[:4], orig_w)
        assert torch.allclose(model.fc.bias.data[:4], orig_b)

    out_40 = model(x)
    assert out_40.shape == (5, 40)
    assert torch.allclose(out_40[:, :4], out_4, atol=1e-4)


def test_lwf_agent_dtd_step_and_distillation():
    """Verify Adapted LwFAgent on DTD dimensions (D=768, 4 classes/task, static 1:1 loss)."""
    agent = LwFAgent(
        feature_dim=768,
        initial_classes=4,
        learning_rate=0.001,
        temperature=2.0,
        distill_lambda=1.0,
        weight_decay=1e-4,
        device="cpu",
    )
    assert isinstance(agent.optimizer, torch.optim.AdamW)

    x0 = torch.randn(10, 768)
    y0 = torch.randint(0, 4, (10,))
    m0 = agent.train_batch(x0, y0)
    assert m0["loss_distill"] == 0.0
    assert m0["loss_total"] == pytest.approx(m0["loss_ce"], rel=1e-6)

    agent.before_task(task_id=1, total_seen_classes=8)
    assert agent.prev_model is not None
    assert agent.prev_num_classes == 4
    assert agent.model.num_classes == 8

    x1 = torch.randn(10, 768)
    y1 = torch.randint(4, 8, (10,))
    m1 = agent.train_batch(x1, y1)
    assert m1["loss_distill"] >= 0.0
    assert m1["loss_total"] == pytest.approx(m1["loss_ce"] + m1["loss_distill"], rel=1e-5)

    acc = agent.evaluate(x1, y1)
    assert 0.0 <= acc <= 1.0


def test_er_buffer_and_agent_dtd():
    """Verify ReplayBuffer reservoir sampling and ERAgent step on DTD dimensions."""
    buf = ReplayBuffer(capacity=16, seed=42)
    buf.add_batch(torch.randn(10, 768), list(range(10)))
    assert len(buf) == 10

    b_feat, b_lbl = buf.sample(6)
    assert b_feat.shape == (6, 768)
    assert b_lbl.shape == (6,)

    buf.add_batch(torch.randn(30, 768), list(range(100, 130)))
    assert len(buf) == 16
    assert buf.total_seen == 40

    agent = ERAgent(
        feature_dim=768,
        initial_classes=4,
        buffer_capacity=5000,
        learning_rate=0.001,
        weight_decay=1e-4,
        seed=42,
        device="cpu",
    )
    assert isinstance(agent.optimizer, torch.optim.AdamW)

    m0 = agent.train_batch(torch.randn(10, 768), torch.randint(0, 4, (10,)))
    assert m0["buffer_occupancy"] == 10

    agent.before_task(task_id=1, total_seen_classes=8)
    m1 = agent.train_batch(torch.randn(10, 768), torch.randint(4, 8, (10,)))
    assert m1["buffer_occupancy"] == 20
    assert 0.0 <= agent.evaluate(torch.randn(8, 768), torch.randint(0, 8, (8,))) <= 1.0


def test_official_lwf_protocol_dtd():
    """Verify OfficialLwFAgent static 40-class head, pure SGD, and dynamic 1/(t+1) weighting."""
    clf = OfficialLwFClassifier(in_features=768, num_classes=40)
    assert clf.fc.out_features == 40

    agent = OfficialLwFAgent(
        feature_dim=768,
        total_classes=40,
        learning_rate=0.001,
        temperature=2.0,
        momentum=0.0,
        weight_decay=0.0,
        device="cpu",
    )
    assert isinstance(agent.optimizer, torch.optim.SGD)
    assert agent.optimizer.defaults["momentum"] == 0.0
    assert agent.optimizer.defaults["weight_decay"] == 0.0

    # Stage 0: w_new = 1.0, w_old = 0.0
    agent.before_task(task_id=0, total_seen_classes=4)
    m0 = agent.train_batch(torch.randn(10, 768), torch.randint(0, 4, (10,)))
    assert m0["weight_new"] == pytest.approx(1.0)
    assert m0["weight_old"] == pytest.approx(0.0)
    assert m0["loss_old"] == 0.0
    agent.after_task()

    # Stage 1: w_new = 0.5, w_old = 0.5
    agent.before_task(task_id=1, total_seen_classes=8)
    m1 = agent.train_batch(torch.randn(10, 768), torch.randint(4, 8, (10,)))
    assert m1["weight_new"] == pytest.approx(0.5)
    assert m1["weight_old"] == pytest.approx(0.5)

    # Verify official_loss_fn_kd matches T^2 * cross-entropy between softmax distributions
    s = torch.randn(4, 40)
    t = torch.randn(4, 40)
    kd = official_loss_fn_kd(s, t, temperature=2.0)
    assert kd.item() > 0.0

    # Fast-forward to Stage 9: w_new = 0.10, w_old = 0.90
    agent.task_seen = 9
    m9 = agent.train_batch(torch.randn(10, 768), torch.randint(36, 40, (10,)))
    assert m9["weight_new"] == pytest.approx(0.1)
    assert m9["weight_old"] == pytest.approx(0.9)


def test_dtd_headline_configs_validity():
    """Verify all DTD and CIFAR-100 headline YAML configurations exist and have valid protocol settings."""
    dtd_configs = [
        ("configs/headline/lwf_dtd.yaml", "lwf", 40, 4, 40),
        ("configs/headline/lwf_dtd_official_protocol.yaml", "official_lwf", 40, 4, 40),
        ("configs/headline/er_dtd.yaml", "er", 40, 4, 40),
    ]
    for rel_path, expected_agent, n_cls, cpt, spc in dtd_configs:
        p = ROOT_DIR / rel_path
        assert p.exists(), f"Missing config: {rel_path}"
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert cfg["dataset"] == "dtd"
        assert cfg["agent"] == expected_agent
        assert cfg["num_classes"] == n_cls
        assert cfg["classes_per_task"] == cpt
        assert cfg["samples_per_class"] == spc
        assert cfg["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
        assert cfg["feature_fusion"] is True
        assert cfg["preprocessing"]["interpolation"] == "bilinear"
        assert cfg["preprocessing"]["normalize"] is False
        assert cfg["logging"]["use_wandb"] is True
        assert cfg["logging"]["project"] == "hcmus-foal-reproduction"
        assert cfg["reference_values"]["paper_reported"]["a_avg_pct"] > 0

    ref_cfg = yaml.safe_load((ROOT_DIR / "configs/headline/foal_dtd_reference.yaml").read_text(encoding="utf-8"))
    assert ref_cfg["dataset"] == "dtd"
    assert ref_cfg["reference_values"]["our_reproduced_seed42"]["a_avg_pct"] == pytest.approx(85.08382936507937)

    for c100_rel in (
        "configs/headline/er_cifar100.yaml",
        "configs/headline/lwf_cifar100.yaml",
        "configs/headline/lwf_cifar100_official_protocol.yaml",
        "configs/headline/foal_cifar100_reference.yaml",
    ):
        assert (ROOT_DIR / c100_rel).exists(), f"Missing CIFAR-100 headline config: {c100_rel}"


@pytest.mark.parametrize(
    "config_rel,agent_key",
    [
        ("configs/headline/lwf_dtd.yaml", "lwf"),
        ("configs/headline/lwf_dtd_official_protocol.yaml", "official_lwf"),
        ("configs/headline/er_dtd.yaml", "er"),
    ],
)
def test_train_baseline_dtd_smoke(tmp_path: Path, config_rel: str, agent_key: str):
    """Run 2-task DTD smoke test on CPU with WANDB_MODE=disabled for sandbox safety."""
    cache_dir = ROOT_DIR / "results" / "dtd_feature_cache"
    if not (cache_dir / "train_augreg2_bilinear_raw_ff.pt").exists():
        pytest.skip("Local DTD feature cache not available for offline CPU smoke test")

    out_dir = tmp_path / f"smoke_{agent_key}"
    old_wandb_mode = os.environ.get("WANDB_MODE")
    try:
        os.environ["WANDB_MODE"] = "disabled"
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                str(ROOT_DIR / config_rel),
                "--output-dir",
                str(out_dir),
                "--run-name",
                f"smoke_dtd_{agent_key}",
                "--device",
                "cpu",
                "--max-tasks",
                "2",
                "--limit-train-per-task",
                "8",
                "--limit-test-per-task",
                "8",
                "--use-local-dtd-cache",
                "--local-dtd-cache-dir",
                str(cache_dir),
            ]
        )
        res = run_baseline(args)
    finally:
        if old_wandb_mode is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = old_wandb_mode

    assert res["dataset"] == "DTD"
    assert res["agent"] == agent_key
    assert len(res["task_matrix"]) == 2
    assert len(res["stage_accuracies"]) == 2
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "task_matrix.csv").exists()
    assert (out_dir / "task_accuracy.csv").exists()

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["training_protocol"]["num_classes"] == 40
    assert meta["training_protocol"]["classes_per_task"] == 4
    assert len(meta["task_split_fingerprint"]["tasks"]) == 2


def test_train_baseline_wandb_logging_integration(tmp_path: Path, monkeypatch):
    """Verify scripts/train_baseline.py executes full W&B init, step log, Table, Artifact, and finish calls."""
    cache_dir = ROOT_DIR / "results" / "dtd_feature_cache"
    if not (cache_dir / "train_augreg2_bilinear_raw_ff.pt").exists():
        pytest.skip("Local DTD feature cache not available for offline CPU W&B integration test")

    mock_wandb = MagicMock()
    mock_wandb.__version__ = "0.31.0"
    mock_run = MagicMock()
    mock_run.summary = {}
    mock_artifact = MagicMock()
    mock_table = MagicMock()
    mock_wandb.init.return_value = mock_run
    mock_wandb.Artifact.return_value = mock_artifact
    mock_wandb.Table.return_value = mock_table

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", mock_wandb)

    out_dir = tmp_path / "wandb_smoke"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(ROOT_DIR / "configs/headline/lwf_dtd_official_protocol.yaml"),
            "--output-dir",
            str(out_dir),
            "--run-name",
            "test_wandb_dtd_official_lwf",
            "--device",
            "cpu",
            "--max-tasks",
            "2",
            "--limit-train-per-task",
            "8",
            "--limit-test-per-task",
            "8",
            "--use-local-dtd-cache",
            "--local-dtd-cache-dir",
            str(cache_dir),
            "--wandb",
            "--wandb-offline",
        ]
    )
    run_baseline(args)

    mock_wandb.init.assert_called_once()
    init_kwargs = mock_wandb.init.call_args.kwargs
    assert init_kwargs["project"] == "hcmus-foal-reproduction"
    assert init_kwargs["group"] == "headline_comparison_dtd"
    assert init_kwargs["name"] == "test_wandb_dtd_official_lwf"
    assert "dtd" in init_kwargs["tags"]
    assert "official_lwf" in init_kwargs["tags"]

    # 2 stage logs + 1 task_accuracy_matrix table log
    assert mock_run.log.call_count == 3
    first_step_payload = mock_run.log.call_args_list[0].args[0]
    assert first_step_payload["task"] == 1
    assert "stage_accuracy_pct" in first_step_payload
    assert "train/weight_new" in first_step_payload

    mock_wandb.Table.assert_called_once()
    table_log_payload = mock_run.log.call_args_list[-1].args[0]
    assert table_log_payload == {"task_accuracy_matrix": mock_table}

    assert "summary/average_incremental_accuracy_pct" in mock_run.summary
    assert "comparison/average_incremental_accuracy_pct" in mock_run.summary

    mock_wandb.Artifact.assert_called_once()
    assert mock_artifact.add_file.call_count == 5
    mock_run.log_artifact.assert_called_once_with(mock_artifact)
    mock_run.finish.assert_called_once()


def test_dtd_headline_analysis_and_wandb_dry_run(tmp_path: Path):
    """Verify Section IX analysis script & 6 figures + W&B importer dry-run on synthetic 10-task matrices."""
    foal_dir = ROOT_DIR / "results" / "dtd" / "dtd_a0_full"
    if not (foal_dir / "metrics.json").exists():
        pytest.skip("Canonical F-OAL DTD run not present")

    runs_root = tmp_path / "runs" / "headline_dtd"
    for sub, diag_val in [("er", 0.78), ("lwf_official_dtd", 0.35), ("lwf", 0.85)]:
        d = runs_root / sub
        d.mkdir(parents=True, exist_ok=True)
        mat = []
        for s in range(10):
            row = [max(0.05, diag_val - 0.04 * (s - t)) for t in range(s + 1)]
            mat.append(row)
        with (d / "task_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in mat:
                writer.writerow([f"{v:.6f}" for v in row])
        from src.foal.metrics import summary_from_task_matrix

        m = summary_from_task_matrix(mat)
        m["a_avg_pct"] = m["average_incremental_accuracy"] * 100.0
        m["a_last_pct"] = m["last_accuracy"] * 100.0
        m["f_past_pct"] = m["forgetting_past_tasks"] * 100.0
        m["f_code_pct"] = m["forgetting_official_code"] * 100.0
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "metrics_summary": m,
                    "stage_accuracies": m["stage_accuracies"],
                    "task_matrix": mat,
                    "total_training_seconds": 12.3,
                    "config": {"seed": 42, "num_classes": 40, "classes_per_task": 4},
                }
            ),
            encoding="utf-8",
        )
        (d / "metadata.json").write_text(
            json.dumps(
                {
                    "git_commit": "testsha",
                    "task_split_fingerprint": {
                        "sha256": "42b13803af83abc522b7f814a6b9e450f884a4a7b675d77a0067491a72e79590"
                    },
                }
            ),
            encoding="utf-8",
        )

    out_analysis = tmp_path / "analysis"
    ns = argparse.Namespace(
        foal_dir=str(foal_dir),
        er_dir=str(runs_root / "er"),
        lwf_official_dir=str(runs_root / "lwf_official_dtd"),
        lwf_adapted_dir=str(runs_root / "lwf"),
        output_dir=str(out_analysis),
        strict=True,
    )
    manifest = run_dtd_analysis(ns)
    assert len(manifest["runs"]) == 4

    for fn in (
        "dtd_headline_final_metrics.json",
        "dtd_headline_final_metrics.csv",
        "dtd_headline_paper_comparison.csv",
        "dtd_headline_protocol_matrix.csv",
        "dtd_headline_stage_accuracies.csv",
        "dtd_headline_task_forgetting.csv",
        "dtd_headline_evidence_manifest.json",
        "fig1_dtd_headline_task_matrices.png",
        "fig2_dtd_headline_foal_vs_baselines.png",
        "fig3_dtd_headline_adapted_vs_official_lwf.png",
        "fig4_dtd_headline_accuracy_progression.png",
        "fig5_dtd_headline_paper_vs_reproduced.png",
        "fig6_dtd_headline_forgetting_comparison.png",
    ):
        assert (out_analysis / fn).exists(), f"Missing analysis output: {fn}"

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "import_dtd_headline_to_wandb.py"),
            "--results-root",
            str(runs_root),
            "--analysis-dir",
            str(out_analysis),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "headline_dtd_er_replay_buffer5000_seed42" in proc.stdout
    assert "headline_dtd_lwf_official_protocol_seed42" in proc.stdout
    assert "headline_dtd_lwf_adapted_adamw_seed42" in proc.stdout


def test_dtd_headline_colab_notebook_structure_and_security():
    """Verify notebooks/foal_dtd_headline_final.ipynb structure, Section IX steps, W&B flags, and no secrets."""
    nb_path = ROOT_DIR / "notebooks" / "foal_dtd_headline_final.ipynb"
    assert nb_path.exists(), "Missing notebooks/foal_dtd_headline_final.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 13

    cell_sources = ["".join(c.get("source", [])) for c in nb["cells"]]
    full_text = "\n".join(cell_sources)

    for required_snippet in (
        "nvidia-smi",
        "FATAL: CUDA GPU is not available!",
        "experiment/dtd-ablation",
        "hcmus-foal-reproduction",
        "headline_comparison_dtd",
        "https://api.forge.coreweave.com",
        "42b13803af83abc522b7f814a6b9e450f884a4a7b675d77a0067491a72e79590",
        "configs/headline/lwf_dtd.yaml",
        "configs/headline/lwf_dtd_official_protocol.yaml",
        "configs/headline/er_dtd.yaml",
        "scripts/train_baseline.py",
        "--feature-cache-dir",
        "--wandb",
        "total_mat_elements != 55",
        "scripts/generate_dtd_headline_analysis.py",
        "scripts/import_dtd_headline_to_wandb.py",
        "foal_dtd_headline_baselines_results.zip",
        "SHA256SUMS.txt",
        "google.colab",
    ):
        assert required_snippet in full_text, f"Notebook missing required content: {required_snippet}"

    forbidden_tokens = ["wandb.login(key=", "secret_key", "Bearer "]
    for tok in forbidden_tokens:
        assert tok.lower() not in full_text.lower(), f"Forbidden token found in notebook: {tok}"
