"""Unit tests for the unified CIFAR-100 reproduction launcher."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import reproduce_cifar100 as rc


class TestReproduceCifar100Launcher(unittest.TestCase):
    def test_smoke_mode_blocks_protected_official_output_dir(self) -> None:
        with self.assertRaises(ValueError):
            rc.resolve_output_dir("smoke", 42, "results/c3_full_gpu_verified/smoke")

    def test_resolve_config_path_validates_c3d_default(self) -> None:
        cfg_path = rc.resolve_config_path("configs/c3_d_bilinear_raw.yaml")
        self.assertTrue(cfg_path.exists())

        with self.assertRaises(FileNotFoundError):
            rc.resolve_config_path("configs/does_not_exist.yaml")

    def test_build_train_namespace_smoke_vs_full(self) -> None:
        cfg_path = rc.resolve_config_path("configs/c3_d_bilinear_raw.yaml")
        out_dir = Path("results/smoke_test_tmp")

        smoke_args = rc.parse_args(["--mode", "smoke", "--seed", "123", "--wandb"])
        ns_smoke = rc.build_train_namespace(smoke_args, cfg_path, out_dir)
        self.assertEqual(ns_smoke.max_tasks, 2)
        self.assertEqual(ns_smoke.limit_train_per_task, 20)
        self.assertEqual(ns_smoke.limit_test_per_task, 20)
        self.assertEqual(ns_smoke.seed, 123)
        self.assertFalse(ns_smoke.wandb)
        self.assertEqual(ns_smoke.experiment_type, "pilot")

        full_args = rc.parse_args(["--mode", "full", "--seed", "42", "--wandb"])
        ns_full = rc.build_train_namespace(full_args, cfg_path, out_dir)
        self.assertIsNone(ns_full.max_tasks)
        self.assertIsNone(ns_full.limit_train_per_task)
        self.assertIsNone(ns_full.limit_test_per_task)
        self.assertTrue(ns_full.wandb)
        self.assertEqual(ns_full.experiment_type, "reproduction")

    def test_main_delegates_to_train_run_and_verifies_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "smoke_out"

            def _fake_run(ns):
                p = Path(ns.output_dir)
                p.mkdir(parents=True, exist_ok=True)
                for fname in ("metrics.json", "metadata.json", "task_accuracy.csv"):
                    (p / fname).write_text("{}", encoding="utf-8")
                return {"average_incremental_accuracy_pct": 90.0}

            with patch("scripts.train.run", side_effect=_fake_run) as mock_run:
                res = rc.main(["--mode", "smoke", "--output-dir", str(out_dir)])

            self.assertEqual(res["average_incremental_accuracy_pct"], 90.0)
            mock_run.assert_called_once()
            self.assertTrue((out_dir / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
