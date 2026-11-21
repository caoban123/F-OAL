"""Local unit tests for immutable, unique W&B provenance backup snapshots."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from scripts import backfill_wandb_provenance as bwp


@contextmanager
def _patched_temp_run_specs():
    """Provide hermetic temporary source files/archives matching bwp.RUN_SPECS."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        temp_specs: list[dict[str, Any]] = []
        for orig_spec in bwp.RUN_SPECS:
            spec = copy.deepcopy(orig_spec)
            archive_path = tmp_root / spec["source_archive"]["filename"]
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            if not archive_path.exists():
                archive_path.write_bytes(f"synthetic-zip-{archive_path.name}".encode("utf-8"))
            spec["source_archive"]["sha256"] = bwp.sha256_file(archive_path)

            rel_src = orig_spec["source_json_path"].relative_to(bwp.ROOT)
            src_path = tmp_root / rel_src
            src_path.parent.mkdir(parents=True, exist_ok=True)
            if spec["run_id"] == "3v53tmq5":
                src_payload = {
                    "git_commit": spec["expected_base_commit"],
                    "environment": {"git": {"commit": spec["expected_base_commit"], "dirty": False}},
                    "model_config": {
                        "backbone": "vit_base_patch16_224_in21k",
                        "gamma": 100.0,
                        "rls_dtype": "float64",
                        "feature_fusion": True,
                        "smooth_projection": True,
                        "projection_dim": 10000,
                        "projection_init": "normal",
                    },
                    "config": {"dataset": {"interpolation": "bilinear", "normalize": "imagenet", "image_size": 224}},
                    "seeds": {"class_order": 42, "model": 42},
                    "class_order": list(range(100)),
                    "task_split_fingerprint": {"class_order_sha256": "synthetic"},
                    "training_protocol": {"batch_size": 64},
                }
            else:
                src_payload = {
                    "environment": {"git": {"commit": spec["expected_base_commit"], "dirty": True}},
                    "config": {
                        "backbone": "vit_base_patch16_224_in21k",
                        "preprocessing": "bilinear_raw",
                        "seed": 42,
                        "class_order_seed": 42,
                        "agent": spec["expected_name"],
                        "batch_size": 64,
                    },
                }
            src_path.write_text(json.dumps(src_payload), encoding="utf-8")
            spec["source_json_path"] = src_path

            new_files: list[Path] = []
            for fp in orig_spec["artifact_files"]:
                target_fp = tmp_root / fp.relative_to(bwp.ROOT)
                target_fp.parent.mkdir(parents=True, exist_ok=True)
                if not target_fp.exists():
                    target_fp.write_text("{}" if target_fp.suffix == ".json" else "synthetic\n", encoding="utf-8")
                new_files.append(target_fp)
            spec["artifact_files"] = new_files
            temp_specs.append(spec)

        with patch.object(bwp, "ROOT", tmp_root), patch.object(bwp, "RUN_SPECS", temp_specs):
            yield temp_specs


class _FakeRun:
    def __init__(self, run_id: str, name: str, config: dict | None = None) -> None:
        self.id = run_id
        self.name = name
        self.state = "finished"
        self.group = None
        self.job_type = None
        self.config = dict(config or {})
        self.summary = {"acc": 90.0}
        self._history = [{"task": 1, "acc": 90.0}]
        self._logged_artifacts: list[Any] = []
        self.update_calls = 0

    def scan_history(self):
        return iter(self._history)

    def logged_artifacts(self):
        return list(self._logged_artifacts)

    def used_artifacts(self):
        return []

    def update(self) -> None:
        self.update_calls += 1


class _FakeApi:
    def __init__(self, runs_by_id: dict[str, _FakeRun]) -> None:
        self._runs = runs_by_id

    def run(self, run_path: str) -> _FakeRun:
        rid = run_path.rsplit("/", 1)[-1]
        return self._runs[rid]


class TestBackfillBackupMechanism(unittest.TestCase):
    def test_unique_snapshot_never_overwrites_even_with_identical_timestamp(self) -> None:
        fixed_ts = datetime(2026, 11, 16, 14, 45, 0, 123456, tzinfo=timezone.utc)
        snaps = {"3v53tmq5": {"id": "3v53tmq5", "summary": {"acc": 90.4}}}

        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp)
            p1, sha1 = bwp.save_unique_backup_snapshot(
                snaps, "ent", "proj", backup_dir=bdir, now_utc=fixed_ts
            )
            p2, sha2 = bwp.save_unique_backup_snapshot(
                snaps, "ent", "proj", backup_dir=bdir, now_utc=fixed_ts
            )

            self.assertNotEqual(p1, p2)
            self.assertTrue(p1.exists())
            self.assertTrue(p2.exists())
            self.assertEqual(sha1, sha2)
            self.assertEqual(
                hashlib.sha256(p1.read_bytes()).hexdigest(),
                sha1,
            )
            sidecar1 = p1.with_suffix(".json.sha256")
            self.assertTrue(sidecar1.exists())
            self.assertIn(sha1, sidecar1.read_text(encoding="utf-8"))

            data1 = json.loads(p1.read_text(encoding="utf-8"))
            self.assertEqual(data1["created_at_utc"], fixed_ts.isoformat())
            self.assertIn("runs_sha256", data1)

    def test_apply_skips_backup_when_all_up_to_date_and_preserves_prior_backups(self) -> None:
        with _patched_temp_run_specs():
            fake_runs = {
                spec["run_id"]: _FakeRun(spec["run_id"], spec["expected_name"])
                for spec in bwp.RUN_SPECS
            }
            fake_api = _FakeApi(fake_runs)
            fake_wandb = SimpleNamespace(Api=lambda: fake_api)

            with tempfile.TemporaryDirectory() as tmp:
                bdir = Path(tmp)
                with (
                    patch.object(bwp, "BACKUP_DIR", bdir),
                    patch.dict(sys.modules, {"wandb": fake_wandb}),
                    patch.object(sys, "argv", ["backfill_wandb_provenance.py", "--apply"]),
                ):
                    # 1st --apply: updates all 4 runs -> creates 1 backup
                    buf1 = io.StringIO()
                    with patch("sys.stdout", buf1):
                        bwp.main()
                    backups_after_1 = sorted(bdir.glob("*.json"))
                    self.assertEqual(len(backups_after_1), 1)
                    first_backup_bytes = backups_after_1[0].read_bytes()
                    self.assertIn("[BACKUP SAVED]", buf1.getvalue())

                    # 2nd --apply: all ALREADY_UP_TO_DATE -> skips backup, 0 writes
                    buf2 = io.StringIO()
                    with patch("sys.stdout", buf2):
                        bwp.main()
                    backups_after_2 = sorted(bdir.glob("*.json"))
                    self.assertEqual(len(backups_after_2), 1)
                    self.assertEqual(backups_after_2[0].read_bytes(), first_backup_bytes)
                    self.assertIn("[BACKUP SKIPPED]", buf2.getvalue())

                    # 3rd --apply after simulated config drift: creates a NEW backup, preserving the 1st
                    fake_runs["3v53tmq5"].config["provenance_remediation"] = {"stale": True}
                    buf3 = io.StringIO()
                    with patch("sys.stdout", buf3):
                        bwp.main()
                    backups_after_3 = sorted(bdir.glob("*.json"))
                    self.assertEqual(len(backups_after_3), 2)
                    self.assertEqual(backups_after_1[0].read_bytes(), first_backup_bytes)
                    self.assertIn("[BACKUP SAVED]", buf3.getvalue())


class _FakeCollection:
    def __init__(self, name: str, versions: list[Any] | None = None) -> None:
        self.name = name
        self._versions = versions or []

    def artifacts(self):
        return list(self._versions)


class _FakeArtifactType:
    def __init__(self, name: str, collections: list[_FakeCollection]) -> None:
        self.name = name
        self._collections = collections

    def collections(self):
        return list(self._collections)


class _FakeApiWithTypes(_FakeApi):
    def __init__(self, runs_by_id: dict[str, _FakeRun], artifact_types: list[_FakeArtifactType]) -> None:
        super().__init__(runs_by_id)
        self._artifact_types = artifact_types

    def artifact_types(self, project: str = ""):
        return list(self._artifact_types)

    def artifact(self, path: str):
        raise RuntimeError("not found")


class TestArtifactPreflightAndUpload(unittest.TestCase):
    def test_collection_type_match_ready_and_dry_run_no_upload(self) -> None:
        run = _FakeRun("3v53tmq5", "c3_d_bilinear_raw")
        api = _FakeApiWithTypes(
            {"3v53tmq5": run},
            [_FakeArtifactType("evaluation", [_FakeCollection("c3_d_bilinear_raw_results", [])])],
        )
        init_calls = []
        fake_wandb = SimpleNamespace(
            Api=lambda: api,
            init=lambda **kw: init_calls.append(kw),
        )

        with (
            _patched_temp_run_specs(),
            patch.dict(sys.modules, {"wandb": fake_wandb}),
            patch.object(sys, "argv", ["backfill_wandb_provenance.py", "--dry-run", "--include-artifacts", "--run-id", "3v53tmq5"]),
        ):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                bwp.main()

        out = buf.getvalue()
        self.assertIn("DRY-RUN ARTIF STATUS  : READY (ARTIFACT_UPLOAD_NEEDED)", out)
        self.assertIn("OVERALL STATUS: READY", out)
        self.assertEqual(len(init_calls), 0)

    def test_collection_type_mismatch_blocks_dry_run(self) -> None:
        run = _FakeRun("3v53tmq5", "c3_d_bilinear_raw")
        api = _FakeApiWithTypes(
            {"3v53tmq5": run},
            [_FakeArtifactType("wrong_type", [_FakeCollection("c3_d_bilinear_raw_results", [])])],
        )
        fake_wandb = SimpleNamespace(Api=lambda: api)

        with (
            _patched_temp_run_specs(),
            patch.dict(sys.modules, {"wandb": fake_wandb}),
            patch.object(sys, "argv", ["backfill_wandb_provenance.py", "--dry-run", "--include-artifacts", "--run-id", "3v53tmq5"]),
        ):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                bwp.main()

        out = buf.getvalue()
        self.assertIn("BLOCKED_ARTIFACT_TYPE_MISMATCH (expected=wrong_type, actual=evaluation)", out)
        self.assertIn("OVERALL STATUS: BLOCKED", out)

    def test_log_artifact_wait_success_and_finish_called(self) -> None:
        with _patched_temp_run_specs():
            spec = bwp.RUN_SPECS[0]
            ok, issues, payload = bwp.verify_and_build_payload(spec)
            self.assertTrue(ok, issues)
            art_spec = bwp.build_artifact_spec(spec, payload)

            events: list[str] = []

            class _FakeLoggedArtifact:
                def wait(self) -> None:
                    events.append("wait")

            class _FakeLiveRun:
                def log_artifact(self, art: Any) -> _FakeLoggedArtifact:
                    events.append(f"log_artifact:{art.name}:{art.type}")
                    return _FakeLoggedArtifact()

                def finish(self) -> None:
                    events.append("finish")

            class _FakeArtifact:
                def __init__(self, name: str, type: str, metadata: dict) -> None:
                    self.name = name
                    self.type = type
                    self.metadata = metadata
                    self.files: list[tuple[str, str]] = []

                def add_file(self, path: str, name: str) -> None:
                    self.files.append((path, name))

            fake_wandb = SimpleNamespace(
                Artifact=_FakeArtifact,
                Settings=lambda **kw: kw,
                init=lambda **kw: _FakeLiveRun(),
            )
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                bwp.upload_artifact_via_resumed_run("ent", "proj", "3v53tmq5", art_spec)

            self.assertEqual(events, ["log_artifact:c3_d_bilinear_raw_results:evaluation", "wait", "finish"])

    def test_log_artifact_wait_exception_still_finishes_and_propagates(self) -> None:
        with _patched_temp_run_specs():
            spec = bwp.RUN_SPECS[0]
            ok, issues, payload = bwp.verify_and_build_payload(spec)
            self.assertTrue(ok, issues)
            art_spec = bwp.build_artifact_spec(spec, payload)

            events: list[str] = []

            class _FailingLoggedArtifact:
                def wait(self) -> None:
                    events.append("wait_fail")
                    raise RuntimeError("HTTP 400 simulated")

            class _FakeLiveRun:
                def log_artifact(self, art: Any) -> _FailingLoggedArtifact:
                    events.append("log_artifact")
                    return _FailingLoggedArtifact()

                def finish(self) -> None:
                    events.append("finish")

            class _FakeArtifact:
                def __init__(self, name: str, type: str, metadata: dict) -> None:
                    self.name = name
                    self.type = type
                    self.metadata = metadata

                def add_file(self, path: str, name: str) -> None:
                    pass

            fake_wandb = SimpleNamespace(
                Artifact=_FakeArtifact,
                Settings=lambda **kw: kw,
                init=lambda **kw: _FakeLiveRun(),
            )
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                with self.assertRaises(RuntimeError) as ctx:
                    bwp.upload_artifact_via_resumed_run("ent", "proj", "3v53tmq5", art_spec)

            self.assertIn("HTTP 400 simulated", str(ctx.exception))
            self.assertEqual(events, ["log_artifact", "wait_fail", "finish"])

    def test_post_hoc_baseline_artifact_metadata_limitations_and_foal_unchanged(self) -> None:
        # Also verify the real historical ER bundle SHA256 when local git-ignored artifacts exist
        real_er_spec = next(s for s in bwp.RUN_SPECS if s["run_id"] == "xlcme1c3")
        if real_er_spec["source_json_path"].exists():
            ok_real, issues_real, er_real_payload = bwp.verify_and_build_payload(real_er_spec)
            self.assertTrue(ok_real, issues_real)
            er_real_art = bwp.build_artifact_spec(real_er_spec, er_real_payload)
            self.assertEqual(
                er_real_art["bundle_sha256"],
                "190a5870a34471ed417e9a7800d1c978b79b0f756973870cc0497872db998491",
            )

        with _patched_temp_run_specs():
            by_id = {s["run_id"]: s for s in bwp.RUN_SPECS}

            # Verify all 3 POST_HOC_IMPORT baselines have the 5 limitation fields with exact types
            for rid in ("xlcme1c3", "c5pjewk4", "ifl3rz2u"):
                ok, issues, payload = bwp.verify_and_build_payload(by_id[rid])
                self.assertTrue(ok, issues)
                art_spec = bwp.build_artifact_spec(by_id[rid], payload)
                meta = art_spec["artifact_metadata"]

                self.assertIs(meta["config_execution_verified"], False)
                self.assertIn("unverified, not proven incorrect", meta["config_execution_note"])
                self.assertIs(meta["source_worktree_dirty"], True)
                self.assertEqual(meta["uncommitted_diff_status"], "MISSING")
                self.assertEqual(meta["original_stdout_log"], "MISSING")
                self.assertEqual(meta["original_cli_command"], "MISSING")
                self.assertEqual(meta["provenance_classification"], "POST_HOC_IMPORT")
                self.assertIs(meta["uploaded_post_hoc"], True)

            # Verify xlcme1c3 bundle_sha256 is deterministic and matches payload bundle_sha256
            _, _, er_payload = bwp.verify_and_build_payload(by_id["xlcme1c3"])
            er_art_spec = bwp.build_artifact_spec(by_id["xlcme1c3"], er_payload)
            self.assertEqual(er_art_spec["bundle_sha256"], er_payload["bundle_sha256"])
            self.assertEqual(len(er_art_spec["bundle_sha256"]), 64)
            self.assertEqual(len(er_art_spec["artifact_metadata"]["source_files"]), 3)

            # Verify F-OAL (3v53tmq5) artifact_metadata is NOT altered with baseline MISSING fields
            ok_foal, issues_foal, foal_payload = bwp.verify_and_build_payload(by_id["3v53tmq5"])
            self.assertTrue(ok_foal, issues_foal)
            foal_meta = bwp.build_artifact_spec(by_id["3v53tmq5"], foal_payload)["artifact_metadata"]
            for key in (
                "config_execution_verified",
                "config_execution_note",
                "source_worktree_dirty",
                "uncommitted_diff_status",
                "original_stdout_log",
                "original_cli_command",
            ):
                self.assertNotIn(key, foal_meta)

            # Verify idempotency check still detects matching artifact on run
            class _FakeLoggedArt:
                def __init__(self) -> None:
                    self.name = "headline_er_provenance_bundle:v0"
                    self.version = "v0"
                    self.digest = "abc"
                    self.metadata = er_art_spec["artifact_metadata"]

                def files(self):
                    return [
                        SimpleNamespace(name=f["arcname"])
                        for f in er_art_spec["files"]
                    ]

            run_with_art = _FakeRun("xlcme1c3", "headline_er_replay_buffer5000_seed42")
            run_with_art._logged_artifacts = [_FakeLoggedArt()]
            api = _FakeApiWithTypes({"xlcme1c3": run_with_art}, [])
            check = bwp.check_existing_wandb_artifact(api, run_with_art, "ent", "proj", er_art_spec)
            self.assertTrue(check["already_on_run"])
            self.assertFalse(check["needs_upload"])


if __name__ == "__main__":
    unittest.main()
