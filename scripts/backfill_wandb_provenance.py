"""Provenance Remediation & Backfill Script for CIFAR-100 W&B Runs.

Safely backfills verified provenance metadata into `run.config['provenance_remediation']` for:
  - 3v53tmq5 (F-OAL, ONLINE_TRAINING)
  - xlcme1c3 (Experience Replay, POST_HOC_IMPORT)
  - c5pjewk4 (Official-protocol LwF, POST_HOC_IMPORT)
  - ifl3rz2u (Adapted LwF, POST_HOC_IMPORT)

DEFAULT MODE: --dry-run (strictly read-only; zero writes to W&B).
Only writes to W&B when --apply is explicitly passed.

Never prints or stores W&B API keys.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("wandb").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
IMPORT_SCRIPT_REFERENCE_COMMIT = "8e977037dc3fb3e4ae231d111cf0e634deebcdee"
BACKUP_DIR = ROOT / "results" / "headline_analysis" / "wandb_provenance_backups"

RUN_SPECS: list[dict[str, Any]] = [
    {
        "run_id": "3v53tmq5",
        "expected_name": "c3_d_bilinear_raw",
        "method": "F-OAL (C3-D)",
        "provenance_classification": "ONLINE_TRAINING",
        "provenance_verification_status": "VERIFIED_WITH_MINOR_ARTIFACT_GAP",
        "training_script": "scripts/train.py",
        "expected_base_commit": "15b8afc71f6109efdbf3ff98a801ffcc83aa58d5",
        "import_script_reference_commit": None,
        "source_json_path": ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw" / "metadata.json",
        "source_archive": {
            "filename": "foal_c3_preprocessing_results.zip",
            "sha256": "a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6",
        },
        "artifact_name": "c3_d_bilinear_raw_results",
        "artifact_type": "evaluation",
        "artifact_files": [
            ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw" / "metrics.json",
            ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw" / "metadata.json",
            ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw" / "task_accuracy.csv",
        ],
        "missing_evidence": {
            "original_stdout_log": "VERIFIED (output.log present on W&B & local)",
            "original_cli_command": "VERIFIED (wandb-metadata.json args present)",
        },
    },
    {
        "run_id": "xlcme1c3",
        "expected_name": "headline_er_replay_buffer5000_seed42",
        "method": "Experience Replay (Buffer=5000)",
        "provenance_classification": "POST_HOC_IMPORT",
        "provenance_verification_status": "VERIFIED_LOCAL_PARTIAL_ON_WANDB",
        "training_script": "scripts/train_baseline.py",
        "expected_base_commit": "b0658c8837563d845253880da8efb1f0f11387fc",
        "import_script_reference_commit": IMPORT_SCRIPT_REFERENCE_COMMIT,
        "source_json_path": ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "er" / "summary.json",
        "source_archive": {
            "filename": "foal_cifar100_headline_results.zip",
            "sha256": "6ec6a4a4cadecb4e77b7704a217878356ac6f6cd4e2d72c8a78a3813cceea551",
        },
        "artifact_name": "headline_er_provenance_bundle",
        "artifact_files": [
            ROOT / "configs" / "headline" / "er_cifar100.yaml",
            ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "er" / "summary.json",
            ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "er" / "task_matrix.csv",
        ],
        "missing_evidence": {
            "original_stdout_log": "MISSING",
            "original_cli_command": "MISSING",
        },
    },
    {
        "run_id": "c5pjewk4",
        "expected_name": "headline_lwf_official_protocol_seed42",
        "method": "Official-Protocol LwF (SGD + Dynamic)",
        "provenance_classification": "POST_HOC_IMPORT",
        "provenance_verification_status": "VERIFIED_LOCAL_PARTIAL_ON_WANDB",
        "training_script": "scripts/train_baseline.py",
        "expected_base_commit": "c3dd3f23fbf2d8d94d481ec75daceb860190118e",
        "import_script_reference_commit": IMPORT_SCRIPT_REFERENCE_COMMIT,
        "source_json_path": ROOT / "results" / "headline_official_lwf_verified" / "runs" / "headline" / "lwf_official_cifar100" / "summary.json",
        "source_archive": {
            "filename": "foal_cifar100_official_lwf_results.zip",
            "sha256": "2f5e67ea86b22d36e8223d8a68494dc0de0f285e015b97b9fcc29df9ccd8f0ce",
        },
        "artifact_name": "headline_official_lwf_provenance_bundle",
        "artifact_files": [
            ROOT / "configs" / "headline" / "lwf_cifar100_official_protocol.yaml",
            ROOT / "results" / "headline_official_lwf_verified" / "runs" / "headline" / "lwf_official_cifar100" / "summary.json",
            ROOT / "results" / "headline_official_lwf_verified" / "runs" / "headline" / "lwf_official_cifar100" / "task_matrix.csv",
        ],
        "missing_evidence": {
            "original_stdout_log": "MISSING",
            "original_cli_command": "MISSING",
        },
    },
    {
        "run_id": "ifl3rz2u",
        "expected_name": "headline_lwf_adapted_adamw_seed42",
        "method": "Adapted LwF (AdamW + Static)",
        "provenance_classification": "POST_HOC_IMPORT",
        "provenance_verification_status": "VERIFIED_LOCAL_PARTIAL_ON_WANDB",
        "training_script": "scripts/train_baseline.py",
        "expected_base_commit": "b0658c8837563d845253880da8efb1f0f11387fc",
        "import_script_reference_commit": IMPORT_SCRIPT_REFERENCE_COMMIT,
        "source_json_path": ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "lwf" / "summary.json",
        "source_archive": {
            "filename": "foal_cifar100_headline_results.zip",
            "sha256": "6ec6a4a4cadecb4e77b7704a217878356ac6f6cd4e2d72c8a78a3813cceea551",
        },
        "artifact_name": "headline_adapted_lwf_provenance_bundle",
        "artifact_files": [
            ROOT / "configs" / "headline" / "lwf_cifar100.yaml",
            ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "lwf" / "summary.json",
            ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "lwf" / "task_matrix.csv",
        ],
        "missing_evidence": {
            "original_stdout_log": "MISSING",
            "original_cli_command": "MISSING",
        },
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_build_payload(spec: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Verify local source files, SHA256s, and commits, and build provenance payload."""
    issues: list[str] = []
    src_json_path = spec["source_json_path"]
    if not src_json_path.exists():
        return False, [f"Missing source JSON: {src_json_path}"], {}

    src_data = json.loads(src_json_path.read_text(encoding="utf-8"))

    if spec["run_id"] == "3v53tmq5":
        git_block = src_data.get("environment", {}).get("git", {})
        observed_commit = src_data.get("git_commit") or git_block.get("commit")
        worktree_dirty = bool(git_block.get("dirty", False))
        source_commit_evidence = "recorded_in_original_metadata_and_wandb_metadata"
        backbone = src_data.get("model_config", {}).get("backbone")
        prep = {
            "interpolation": src_data.get("config", {}).get("dataset", {}).get("interpolation"),
            "normalize": src_data.get("config", {}).get("dataset", {}).get("normalize"),
            "size": src_data.get("config", {}).get("dataset", {}).get("image_size"),
        }
        seed_info = src_data.get("seeds")
        class_order_info = {
            "class_order_seed": src_data.get("seeds", {}).get("class_order"),
            "class_order_len": len(src_data.get("class_order", [])),
            "class_order_sha256": src_data.get("task_split_fingerprint", {}).get("class_order_sha256"),
        }
        hyperparams = {
            "batch_size": src_data.get("training_protocol", {}).get("batch_size"),
            "gamma": src_data.get("model_config", {}).get("gamma"),
            "rls_dtype": src_data.get("model_config", {}).get("rls_dtype"),
            "feature_fusion": src_data.get("model_config", {}).get("feature_fusion"),
            "smooth_projection": src_data.get("model_config", {}).get("smooth_projection"),
            "projection_dim": src_data.get("model_config", {}).get("projection_dim"),
            "projection_init": src_data.get("model_config", {}).get("projection_init"),
        }
    else:
        git_block = src_data.get("environment", {}).get("git", {})
        observed_commit = git_block.get("commit")
        worktree_dirty = bool(git_block.get("dirty", False))
        source_commit_evidence = "recorded_in_original_summary"
        cfg_block = src_data.get("config", {})
        backbone = cfg_block.get("backbone")
        prep = cfg_block.get("preprocessing")
        seed_info = {
            "seed": cfg_block.get("seed"),
            "class_order_seed": cfg_block.get("class_order_seed"),
        }
        class_order_info = {
            "class_order_seed": cfg_block.get("class_order_seed"),
            "explicit_class_order_list": "NOT_STORED_IN_SUMMARY_JSON (deterministic via make_class_order(100, 42))",
        }
        hyperparams = {
            k: cfg_block[k]
            for k in (
                "agent", "optimizer", "learning_rate", "momentum", "weight_decay",
                "temperature", "distill_lambda", "dynamic_loss_weighting",
                "full_output_head", "buffer_capacity", "retrieve_policy",
                "update_policy", "batch_size", "feature_fusion",
            )
            if k in cfg_block
        }

    if observed_commit != spec["expected_base_commit"]:
        issues.append(f"Commit mismatch: observed={observed_commit} != expected={spec['expected_base_commit']}")

    archive_path = ROOT / spec["source_archive"]["filename"]
    if archive_path.exists():
        actual_zip_sha = sha256_file(archive_path)
        if actual_zip_sha != spec["source_archive"]["sha256"]:
            issues.append(
                f"Archive SHA256 mismatch for {archive_path.name}: {actual_zip_sha} != {spec['source_archive']['sha256']}"
            )
    else:
        issues.append(f"Source archive not found locally: {archive_path.name}")

    verified_artifacts: list[dict[str, Any]] = []
    for fp in spec["artifact_files"]:
        if not fp.exists():
            issues.append(f"Missing artifact file: {fp}")
            continue
        verified_artifacts.append({
            "path": str(fp.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
        })

    if issues:
        return False, issues, {}

    bundle_sha = hashlib.sha256(
        json.dumps(verified_artifacts, sort_keys=True).encode("utf-8")
    ).hexdigest()

    proposed_metadata: dict[str, Any] = {
        "provenance_classification": spec["provenance_classification"],
        "provenance_verification_status": spec["provenance_verification_status"],
        "source_code_base_commit": observed_commit,
        "source_worktree_dirty": worktree_dirty,
        "uncommitted_diff_status": "MISSING" if worktree_dirty else "CLEAN",
        "source_commit_evidence": source_commit_evidence,
        "import_script_reference_commit": spec["import_script_reference_commit"],
        "import_script_commit_evidence": (
            "latest_git_history_commit_for_import_script_not_runtime_verified"
            if spec["import_script_reference_commit"]
            else None
        ),
        "training_script": spec["training_script"],
        "verified_backbone": backbone,
        "verified_preprocessing": prep,
        "verified_seeds": seed_info,
        "verified_class_order": class_order_info,
        "verified_hyperparameters": hyperparams,
        "source_artifact_references": {
            "archive": spec["source_archive"],
            "files": verified_artifacts,
            "bundle_sha256": bundle_sha,
        },
        "missing_evidence": spec["missing_evidence"],
    }

    return True, [], {
        "proposed_config_key": "provenance_remediation",
        "proposed_metadata": proposed_metadata,
        "artifact_name": spec["artifact_name"],
        "verified_artifacts": verified_artifacts,
        "bundle_sha256": bundle_sha,
    }


def snapshot_run_state(run: Any) -> dict[str, Any]:
    """Capture a complete snapshot of run metadata, summary, history, and existing artifacts."""
    history_rows = list(run.scan_history())
    history_digest = hashlib.sha256(
        json.dumps(history_rows, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    summary_clean = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    logged_artifacts = [
        {
            "name": a.name,
            "type": a.type,
            "version": getattr(a, "version", None),
            "digest": getattr(a, "digest", None),
        }
        for a in run.logged_artifacts()
    ]
    used_artifacts = [
        {
            "name": a.name,
            "type": a.type,
            "version": getattr(a, "version", None),
            "digest": getattr(a, "digest", None),
        }
        for a in run.used_artifacts()
    ]
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "group": run.group,
        "job_type": run.job_type,
        "config": dict(run.config),
        "summary": summary_clean,
        "history_len": len(history_rows),
        "history_sha256": history_digest,
        "logged_artifacts": logged_artifacts,
        "used_artifacts": used_artifacts,
    }


def verify_post_apply(before: dict[str, Any], after_run: Any, expected_remediation: dict[str, Any]) -> None:
    """Strictly verify zero unintended mutations after writing provenance_remediation."""
    after = snapshot_run_state(after_run)

    if after["id"] != before["id"]:
        raise RuntimeError(f"Run ID changed! {before['id']} -> {after['id']}")
    if after["name"] != before["name"]:
        raise RuntimeError(f"Run name changed! {before['name']} -> {after['name']}")
    if after["job_type"] != before["job_type"]:
        raise RuntimeError(f"Run job_type changed! {before['job_type']} -> {after['job_type']}")
    if after["summary"] != before["summary"]:
        raise RuntimeError(f"Summary metrics mutated on {before['id']}!")
    if after["history_len"] != before["history_len"] or after["history_sha256"] != before["history_sha256"]:
        raise RuntimeError(f"History mutated on {before['id']}!")
    if before["config"].get("provenance") != after["config"].get("provenance"):
        raise RuntimeError(f"Legacy 'provenance' field mutated on {before['id']}!")
    if after["config"].get("provenance_remediation") != expected_remediation:
        raise RuntimeError(f"Written 'provenance_remediation' does not match expected payload on {before['id']}!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill verified W&B provenance (defaults to --dry-run).")
    parser.add_argument("--entity", default="caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm")
    parser.add_argument("--project", default="hcmus-foal-reproduction")
    parser.add_argument("--run-id", default=None, help="Target a single W&B run ID (e.g. 3v53tmq5)")
    parser.add_argument("--dry-run", action="store_true", help="Explicitly force dry-run (default behavior)")
    parser.add_argument("--apply", action="store_true", help="Actually write provenance_remediation to W&B config")
    parser.add_argument("--include-artifacts", action="store_true", help="Also log provenance artifacts (opt-in)")
    return parser.parse_args()


def prepare_run_metadata(
    existing_cfg: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Finalize proposed_metadata with any legacy discrepancy note and check if already synced."""
    proposed = dict(payload["proposed_metadata"])
    existing_prov = existing_cfg.get("provenance")
    existing_remediation = existing_cfg.get("provenance_remediation")

    legacy_exec_commit = existing_prov.get("execution_commit") if isinstance(existing_prov, dict) else None
    if legacy_exec_commit and legacy_exec_commit != proposed["source_code_base_commit"]:
        proposed["legacy_metadata_discrepancy_note"] = (
            f"Preserved legacy provenance.execution_commit='{legacy_exec_commit}'; "
            f"recorded actual summary.json commit in source_code_base_commit="
            f"'{proposed['source_code_base_commit']}'."
        )

    already_synced = (existing_remediation == proposed)
    return proposed, already_synced


def save_unique_backup_snapshot(
    pre_snapshots: dict[str, dict[str, Any]],
    entity: str,
    project: str,
    backup_dir: Path | None = None,
    now_utc: datetime | None = None,
) -> tuple[Path, str]:
    """Write an immutable, uniquely named backup snapshot with timestamp and SHA256."""
    target_dir = backup_dir if backup_dir is not None else BACKUP_DIR
    ts = now_utc or datetime.now(timezone.utc)
    created_at_utc = ts.isoformat()
    timestamp_slug = ts.strftime("%Y%m%dT%H%M%S%fZ")
    runs_sha256 = hashlib.sha256(
        json.dumps(pre_snapshots, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    backup_payload = {
        "created_at_utc": created_at_utc,
        "entity": entity,
        "project": project,
        "runs_sha256": runs_sha256,
        "runs": pre_snapshots,
    }
    serialized = json.dumps(backup_payload, indent=2, default=str)
    file_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    target_dir.mkdir(parents=True, exist_ok=True)
    base_stem = f"wandb_provenance_backup_{timestamp_slug}_{file_sha256[:12]}"
    candidate = target_dir / f"{base_stem}.json"
    counter = 1
    while True:
        try:
            with candidate.open("x", encoding="utf-8", newline="\n") as f:
                f.write(serialized)
            break
        except FileExistsError:
            candidate = target_dir / f"{base_stem}_{counter:02d}.json"
            counter += 1

    sha_sidecar = candidate.with_suffix(".json.sha256")
    with sha_sidecar.open("x", encoding="utf-8", newline="\n") as f:
        f.write(f"{file_sha256}  {candidate.name}\n")

    return candidate, file_sha256


def build_artifact_spec(spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Construct the artifact upload specification and metadata."""
    files_info = [
        {
            "path": fp,
            "arcname": fp.name,
            "repo_relative_path": fp.relative_to(ROOT).as_posix(),
            "size_bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
        }
        for fp in spec["artifact_files"]
    ]
    artifact_metadata = {
        "source_run_id": spec["run_id"],
        "provenance_classification": spec["provenance_classification"],
        "evidence_type": "verified_local_result_bundle",
        "uploaded_post_hoc": True,
        "uploaded_post_hoc_note": (
            "uploaded_post_hoc=True describes the post-hoc timing of artifact attachment "
            "and does not alter the run's provenance_classification."
        ),
        "source_archive": spec["source_archive"],
        "source_files": [
            {
                "filename": f["arcname"],
                "repo_relative_path": f["repo_relative_path"],
                "size_bytes": f["size_bytes"],
                "sha256": f["sha256"],
            }
            for f in files_info
        ],
        "bundle_sha256": payload["bundle_sha256"],
    }
    if spec["provenance_classification"] == "POST_HOC_IMPORT":
        artifact_metadata.update({
            "config_execution_verified": False,
            "config_execution_note": (
                "YAML configuration is verified present in the Colab source ZIP archive, "
                "but there is no direct runtime evidence (original CLI command/stdout log) "
                "proving it was the exact configuration executed at runtime; "
                "config_execution_verified=False means unverified, not proven incorrect."
            ),
            "source_worktree_dirty": True,
            "uncommitted_diff_status": "MISSING",
            "original_stdout_log": "MISSING",
            "original_cli_command": "MISSING",
        })
    return {
        "artifact_name": spec["artifact_name"],
        "artifact_type": spec.get("artifact_type", "provenance_results"),
        "files": files_info,
        "bundle_sha256": payload["bundle_sha256"],
        "artifact_metadata": artifact_metadata,
    }


def check_existing_wandb_artifact(
    api: Any,
    run: Any,
    entity: str,
    project: str,
    artifact_spec: dict[str, Any],
) -> dict[str, Any]:
    """Check if an identical artifact or collection type mismatch exists on the run or project."""
    art_name = artifact_spec["artifact_name"]
    expected_type = artifact_spec["artifact_type"]
    expected_bundle_sha = artifact_spec["bundle_sha256"]
    expected_filenames = sorted(f["arcname"] for f in artifact_spec["files"])

    run_matches: list[dict[str, Any]] = []
    for logged_art in run.logged_artifacts():
        base_name = logged_art.name.split(":", 1)[0]
        if base_name != art_name:
            continue
        art_files = sorted(f.name for f in logged_art.files())
        art_meta = dict(getattr(logged_art, "metadata", {}) or {})
        meta_bundle_sha = art_meta.get("bundle_sha256")
        if meta_bundle_sha == expected_bundle_sha and art_files == expected_filenames:
            run_matches.append({
                "name": logged_art.name,
                "version": getattr(logged_art, "version", None),
                "digest": getattr(logged_art, "digest", None),
                "files": art_files,
            })

    existing_collections: list[dict[str, Any]] = []
    type_mismatch: dict[str, str] | None = None
    try:
        for at in api.artifact_types(project=f"{entity}/{project}"):
            for col in at.collections():
                if col.name == art_name:
                    versions = [getattr(v, "version", None) for v in col.artifacts()]
                    existing_collections.append({
                        "type": at.name,
                        "name": col.name,
                        "versions": versions,
                    })
                    if at.name != expected_type:
                        type_mismatch = {
                            "expected_type": at.name,
                            "actual_type": expected_type,
                        }
    except Exception:
        pass

    project_latest: dict[str, Any] | None = None
    try:
        latest = api.artifact(f"{entity}/{project}/{art_name}:latest")
        latest_meta = dict(getattr(latest, "metadata", {}) or {})
        project_latest = {
            "name": latest.name,
            "type": getattr(latest, "type", None),
            "version": getattr(latest, "version", None),
            "digest": getattr(latest, "digest", None),
            "bundle_sha256": latest_meta.get("bundle_sha256"),
            "files": sorted(f.name for f in latest.files()),
        }
    except Exception:
        project_latest = None

    already_on_run = len(run_matches) > 0
    return {
        "already_on_run": already_on_run,
        "run_matches": run_matches,
        "existing_collections": existing_collections,
        "type_mismatch": type_mismatch,
        "project_latest": project_latest,
        "needs_upload": (not already_on_run) and (type_mismatch is None),
    }


def upload_artifact_via_resumed_run(
    entity: str,
    project: str,
    run_id: str,
    artifact_spec: dict[str, Any],
) -> None:
    """Attach an artifact to a finished W&B run via resume='must' without wandb.log() or summary writes."""
    import wandb

    art = wandb.Artifact(
        name=artifact_spec["artifact_name"],
        type=artifact_spec["artifact_type"],
        metadata=artifact_spec["artifact_metadata"],
    )
    for f in artifact_spec["files"]:
        art.add_file(str(f["path"]), name=f["arcname"])

    live_run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        reinit="finish_previous",
        settings=wandb.Settings(disable_code=True, disable_git=True, quiet=True),
    )
    try:
        logged = live_run.log_artifact(art)
        logged.wait()
    finally:
        live_run.finish()


def main() -> None:
    args = parse_args()
    is_apply = bool(args.apply) and not bool(args.dry_run)

    target_specs = RUN_SPECS
    if args.run_id:
        target_specs = [s for s in RUN_SPECS if s["run_id"] == args.run_id]
        if not target_specs:
            valid_ids = [s["run_id"] for s in RUN_SPECS]
            print(f"ERROR: Unknown --run-id '{args.run_id}'. Valid IDs: {valid_ids}")
            sys.exit(1)

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is not installed.")
        sys.exit(1)

    print("=" * 95)
    print("W&B PROVENANCE REMEDIATION")
    print(f"Target Project : {args.entity}/{args.project}")
    print(f"Target Run(s)  : {args.run_id if args.run_id else 'ALL (4 runs)'}")
    print(f"Include Artif. : {bool(args.include_artifacts)}")
    print(f"Execution Mode : {'APPLY (LIVE WRITE)' if is_apply else 'DRY RUN (READ-ONLY — NO WRITES TO W&B)'}")
    print("=" * 95)

    api = wandb.Api()
    all_ready = True
    pre_snapshots: dict[str, dict[str, Any]] = {}
    prepared_runs: list[tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], bool, dict[str, Any] | None, dict[str, Any] | None]] = []

    # Phase 1: Verify & Snapshot all target runs before any write
    for spec in target_specs:
        rid = spec["run_id"]
        run_path = f"{args.entity}/{args.project}/{rid}"
        ok, issues, payload = verify_and_build_payload(spec)
        if not ok:
            all_ready = False
            print(f"\n--- Run ID: {rid} | STATUS: BLOCKED -> {issues}")
            continue

        run = api.run(run_path)
        if run.name != spec["expected_name"]:
            all_ready = False
            print(f"\n--- Run ID: {rid} | STATUS: BLOCKED -> Name mismatch: {run.name} != {spec['expected_name']}")
            continue

        snap = snapshot_run_state(run)
        pre_snapshots[rid] = snap
        proposed_metadata, metadata_synced = prepare_run_metadata(snap["config"], payload)

        artifact_spec: dict[str, Any] | None = None
        artifact_check: dict[str, Any] | None = None
        if args.include_artifacts:
            artifact_spec = build_artifact_spec(spec, payload)
            artifact_check = check_existing_wandb_artifact(api, run, args.entity, args.project, artifact_spec)
            if artifact_check.get("type_mismatch"):
                all_ready = False

        prepared_runs.append((spec, run, payload, proposed_metadata, metadata_synced, artifact_spec, artifact_check))

    runs_needing_update = [
        spec["run_id"]
        for spec, _, _, _, metadata_synced, _, artifact_check in prepared_runs
        if (not metadata_synced) or (bool(args.include_artifacts) and bool(artifact_check and artifact_check["needs_upload"]))
    ]

    if is_apply:
        if not all_ready:
            print("ABORTING --apply: One or more runs are BLOCKED.")
            sys.exit(1)
        if runs_needing_update:
            backup_path, backup_sha256 = save_unique_backup_snapshot(
                pre_snapshots, args.entity, args.project
            )
            print(f"\n[BACKUP SAVED] Pre-apply snapshot written to {backup_path} (sha256={backup_sha256})")
        else:
            print("\n[BACKUP SKIPPED] All target runs (metadata & artifacts) are ALREADY_UP_TO_DATE; no pre-update backup needed.")

    # Phase 2: Report and (if --apply) Execute & Verify per run
    for spec, run, payload, proposed_metadata, metadata_synced, artifact_spec, artifact_check in prepared_runs:
        rid = spec["run_id"]
        run_path = f"{args.entity}/{args.project}/{rid}"
        snap = pre_snapshots[rid]

        print(f"\n--- Run ID: {rid} | Method: {spec['method']} ---")
        print(f"  Run Name              : {run.name} (state={run.state}, job_type={run.job_type})")
        print(f"  Snapshot Pre-State    : summary_keys={len(snap['summary'])}, history_len={snap['history_len']}, history_sha256={snap['history_sha256'][:16]}...")
        print(f"  Existing Logged Artif.: {snap['logged_artifacts']}")
        print(f"  source_code_base_commit: {proposed_metadata['source_code_base_commit']}")
        print(f"  source_worktree_dirty : {proposed_metadata['source_worktree_dirty']}")
        print(f"  uncommitted_diff_status: {proposed_metadata['uncommitted_diff_status']}")
        print(f"  source_commit_evidence: {proposed_metadata['source_commit_evidence']}")
        print(f"  import_ref_commit     : {proposed_metadata['import_script_reference_commit']}")
        print(f"  missing_evidence      : {proposed_metadata['missing_evidence']}")
        for af in payload["verified_artifacts"]:
            print(f"  Artifact File [OK]    : {af['path']} ({af['size_bytes']} B, sha256={af['sha256']})")

        if args.include_artifacts and artifact_spec and artifact_check:
            print(f"  Artifact Target       : {artifact_spec['artifact_name']} (type={artifact_spec['artifact_type']})")
            print(f"  Artifact Bundle SHA256: {artifact_spec['bundle_sha256']}")
            print(f"  Existing Collections  : {artifact_check.get('existing_collections', [])}")
            print(f"  Existing Project Artif: {artifact_check['project_latest']}")
            print(f"  Artifact On Run Match : {artifact_check['already_on_run']} (matches={artifact_check['run_matches']})")
            print(f"  Artifact Metadata Plan: source_run_id={artifact_spec['artifact_metadata']['source_run_id']}, "
                  f"provenance_classification={artifact_spec['artifact_metadata']['provenance_classification']}, "
                  f"evidence_type={artifact_spec['artifact_metadata']['evidence_type']}, "
                  f"uploaded_post_hoc={artifact_spec['artifact_metadata']['uploaded_post_hoc']}")
            print(f"  Artifact Metadata JSON: {json.dumps(artifact_spec['artifact_metadata'], sort_keys=True)}")
            print("  Resume Strategy       : wandb.init(id=run_id, resume='must', settings=disable_code/git) -> log_artifact().wait() -> finish() [NO wandb.log(), NO summary edit]")

        art_needs_upload = bool(args.include_artifacts and artifact_check and artifact_check["needs_upload"])

        if not is_apply:
            meta_status = "ALREADY_UP_TO_DATE" if metadata_synced else "READY (UPDATE_NEEDED)"
            print(f"  DRY-RUN META STATUS   : {meta_status}")
            if args.include_artifacts:
                tm = artifact_check.get("type_mismatch") if artifact_check else None
                if tm:
                    art_status = (
                        f"BLOCKED_ARTIFACT_TYPE_MISMATCH "
                        f"(expected={tm['expected_type']}, actual={tm['actual_type']})"
                    )
                elif art_needs_upload:
                    art_status = "READY (ARTIFACT_UPLOAD_NEEDED)"
                else:
                    art_status = "ALREADY_UP_TO_DATE"
                print(f"  DRY-RUN ARTIF STATUS  : {art_status}")
            continue

        if metadata_synced:
            print("  LIVE META STATUS      : ALREADY_UP_TO_DATE (no config write needed)")
        else:
            run.config["provenance_remediation"] = proposed_metadata
            run.update()
            re_run = api.run(run_path)
            verify_post_apply(snap, re_run, proposed_metadata)
            print("  LIVE META STATUS      : UPDATED_AND_VERIFIED (Summary/History/ID/Name/Legacy-Provenance intact)")

        if args.include_artifacts:
            if art_needs_upload and artifact_spec:
                upload_artifact_via_resumed_run(args.entity, args.project, rid, artifact_spec)
                re_run_art = api.run(run_path)
                verify_post_apply(snap, re_run_art, proposed_metadata)
                post_check = check_existing_wandb_artifact(api, re_run_art, args.entity, args.project, artifact_spec)
                if not post_check["already_on_run"]:
                    raise RuntimeError(f"Artifact upload verification failed on {rid}!")
                print("  LIVE ARTIFACT STATUS  : UPLOADED_AND_VERIFIED (Summary/History/ID/Name/Config intact)")
            else:
                print("  LIVE ARTIFACT STATUS  : ALREADY_UP_TO_DATE (no upload needed)")

    print("\n" + "=" * 95)
    print(f"OVERALL STATUS: {'READY' if all_ready else 'BLOCKED'}")
    print("=" * 95)


if __name__ == "__main__":
    main()
