"""Generate Google Colab notebook notebooks/foal_dtd_headline_final.ipynb.

Implements Section IX (Sections 12-15) Headline Comparison experiments (Adapted LwF,
Official-Protocol LwF, Adapted ER vs. Full F-OAL) on the Describable Textures Dataset
(DTD) with full Weights & Biases (wandb) tracking.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def make_md(source: str) -> dict:
    lines = source.splitlines()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }


def make_code(source: str) -> dict:
    lines = source.splitlines()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }


def build_notebook() -> dict:
    cells = []

    cells.append(
        make_md(
            """# F-OAL — DTD Headline Comparison Benchmark (ER & LwF vs. F-OAL, Section IX Methodology)

- **Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024, Page 7, Table 2 & Table 3)
- **Author Official Repository:** [`https://github.com/liuyuchen-cz/F-OAL`](https://github.com/liuyuchen-cz/F-OAL) (Pinned Ref Commit: `73e4ac1`)
- **Group Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL)
- **Target Branch:** `experiment/dtd-ablation`
- **Methodology Source:** Closely follows **Section IX (Sections 12–15)** of `docs/CIFAR100_COMPLETE_WORK_REPORT.pdf`, adapted strictly to the **DTD** benchmark protocol (`docs/DTD_COMPLETE_WORK_REPORT.md`).

### Compared Paradigms on DTD (10 Tasks $\\times$ 4 Classes, 40/47 Classes, Seed 42)
1. **F-OAL (Full, A0 / C3-D):** Exemplar-free ($M=0$) forward-only analytic Ridge RLS with 12-block Feature Fusion ($D=768$) + Smooth Projection ($d_p=1000$, Uniform). Canonical DTD Seed 42: $\\bar{A} = 85.08\\%$, $A_{10} = 76.06\\%$, $F_{\\text{past}} = 8.61\\%$, $F_{\\text{code}} = 7.75\\%$ (Paper DTD Table 2: **82.8%**).
2. **Adapted LwF (Section IX.13 — Recency Collapse Audit):** Exemplar-free ($M=0$) dynamic classifier head (`nn.Linear(768, seen_classes)`), static loss $\\mathcal{L} = \\mathcal{L}_{\\text{CE}} + \\mathcal{L}_{\\text{KD}}(T=2.0)$, `AdamW(lr=0.001, weight_decay=1e-4)`.
3. **Official-Protocol LwF (Section IX.15 — Catastrophic Rigidity Audit):** Exemplar-free ($M=0$) static 40-class head (`nn.Linear(768, 40)`), dynamic loss weighting $\\mathcal{L}_t = \\frac{1}{t+1}\\mathcal{L}_{\\text{new}} + \\left(1 - \\frac{1}{t+1}\\right)\\mathcal{L}_{\\text{old}}$ with full 40-class KD ($T=2.0$), pure `SGD(lr=0.001, momentum=0.0, weight_decay=0.0)`. Paper DTD Table 2: $\\bar{A}=55.8\\%$, $A_{10}=48.2\\%$, $F=12.6\\%$.
4. **Adapted ER (Section IX.14 — Episodic Memory Anchor):** Experience Replay with reservoir buffer ($M=5000$), dynamic classifier head, `AdamW(lr=0.001, weight_decay=1e-4)`. Paper DTD Table 2: $\\bar{A}=73.2\\%$, $A_{10}=68.1\\%$, $F=18.5\\%$.

### Controlled Protocol Parity
- **Backbone:** Frozen `vit_base_patch16_224.augreg2_in21k_ft_in1k` + 12-block CLS Feature Fusion + LayerNorm ($D=768$)
- **Preprocessing:** Bilinear resize $224 \\times 224$, raw $[0, 1]$ tensor (no normalization)
- **Data Split:** 40 selected classes out of 47 (`tanganke/dtd`), 10 tasks $\\times$ 4 classes, `samples_per_class: 40` (1,600 train / 1,600 test), streaming `batch_size=10`, strictly 1 pass, Seed `42` (`task_split_fingerprint`: `42b13803...`)
- **Experiment Tracking:** Public Weights & Biases (`wandb`) on CoreWeave Forge (`hcmus-foal-reproduction`, group `headline_comparison_dtd`)"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 1: GPU & Environment Preflight ===
!nvidia-smi
import os
import sys
import shutil
import platform
import subprocess
import torch

print(f"Python version   : {sys.version}")
print(f"Platform         : {platform.platform()}")
print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "FATAL: CUDA GPU is not available! Switch Colab runtime to GPU (T4 / L4 / A100) before proceeding."
    )

device_name = torch.cuda.get_device_name(0)
device_props = torch.cuda.get_device_properties(0)
vram_gb = device_props.total_memory / (1024 ** 3)
print(f"GPU Device       : {device_name}")
print(f"CUDA version     : {torch.version.cuda}")
print(f"Total VRAM       : {vram_gb:.2f} GB")
print(f"Free Disk Space  : {shutil.disk_usage('.').free / (1024 ** 3):.2f} GB")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 2: Clone Group Repository & Checkout Working Branch ===
import os
import shutil
import subprocess
import sys
from pathlib import Path

if Path("/content").exists():
    os.chdir("/content")
    REPO_DIR = Path("/content/F-OAL")
else:
    REPO_DIR = Path.cwd() / "F-OAL"

BRANCH_NAME = "experiment/dtd-ablation"

if (REPO_DIR / ".git").exists():
    print(f"Updating existing repository at {REPO_DIR} (branch {BRANCH_NAME})...")
    os.chdir(str(REPO_DIR))
    subprocess.run(["git", "fetch", "origin"], check=True)
    subprocess.run(["git", "checkout", "-B", BRANCH_NAME, f"origin/{BRANCH_NAME}"], check=True)
    subprocess.run(["git", "pull", "origin", BRANCH_NAME], check=True)
else:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    print(f"Cloning {BRANCH_NAME} into {REPO_DIR}...")
    subprocess.run(
        ["git", "clone", "-b", BRANCH_NAME, "https://github.com/caoban123/F-OAL.git", str(REPO_DIR)],
        check=True,
    )
    os.chdir(str(REPO_DIR))

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), text=True).strip()
print(f"Repository ready at commit {rev} (branch: {BRANCH_NAME})")
print("All DTD Section IX baseline files verified present in repository.")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 3: Install Dependencies & Setup Weights & Biases (wandb) ===
!pip install -q timm pyyaml pytest datasets matplotlib pandas
!pip install -q --upgrade wandb
!pip install -q -e . --no-deps

import os
import wandb

# Retrieve W&B API Key from Colab Secrets or environment (never hardcode credentials)
try:
    from google.colab import userdata
    try:
        secret_val = userdata.get("WANDB_API_KEY")
        if secret_val:
            os.environ["WANDB_API_KEY"] = secret_val
    except Exception:
        pass
except Exception:
    pass

WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://api.forge.coreweave.com")
os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL

WANDB_PROJECT = "hcmus-foal-reproduction"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm")
WANDB_GROUP = "headline_comparison_dtd"

print(f"W&B Host    : {WANDB_BASE_URL}")
print(f"W&B Target  : {WANDB_ENTITY}/{WANDB_PROJECT} (group: {WANDB_GROUP})")
print("Forge Key   : https://forge.coreweave.com/wandb/authorize")

USE_WANDB = False
try:
    if WANDB_BASE_URL:
        wandb.login(host=WANDB_BASE_URL)
    else:
        wandb.login()
    USE_WANDB = True
    print("SUCCESS: W&B cloud tracking authenticated and enabled.")
except Exception as exc:
    print(f"NOTICE: W&B interactive/cloud login unavailable ({exc}). Switching to WANDB_MODE=offline.")
    os.environ["WANDB_MODE"] = "offline"

def wandb_cli_args(group_name: str = WANDB_GROUP) -> list[str]:
    args = [
        "--wandb",
        "--wandb-project", WANDB_PROJECT,
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-group", group_name,
    ]
    if USE_WANDB:
        if WANDB_BASE_URL:
            args.extend(["--wandb-base-url", WANDB_BASE_URL])
    else:
        args.append("--wandb-offline")
    return args"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 4: Prepare DTD Dataset & Verify Canonical Split Fingerprint ===
import os
from src.foal.dtd_datasets import load_dtd, make_dtd_class_order, make_dtd_task_specs, DTD_CLASS_NAMES
from scripts.train import compute_task_split_fingerprint

os.environ["HF_DATASETS_OFFLINE"] = "0"
train_src, test_src = load_dtd(root="./data")
labels_train = list(train_src.fine_label)
labels_test = list(test_src.fine_label)

order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=42)
specs = make_dtd_task_specs(
    train_labels=labels_train,
    test_labels=labels_test,
    order=order,
    classes_per_task=4,
    samples_per_class=40,
    seed=42,
)
fp = compute_task_split_fingerprint(
    order=order,
    specs=specs,
    max_tasks=10,
    limit_train=None,
    limit_test=None,
    stream_seed=42,
    labels_test=labels_test,
)
EXPECTED_SPLIT_SHA = "42b13803af83abc522b7f814a6b9e450f884a4a7b675d77a0067491a72e79590"
assert fp["sha256"] == EXPECTED_SPLIT_SHA, (
    f"DTD Split SHA mismatch! Got {fp['sha256']}, expected {EXPECTED_SPLIT_SHA}"
)

print(f"DTD Dataset Ready   : train={len(train_src)}, test={len(test_src)}")
print(f"Selected 40 Classes : {[DTD_CLASS_NAMES[c] for c in order[:8]]} ... (total={len(order)})")
print(f"Split Fingerprint   : {fp['sha256']} (MATCHES CANONICAL A0)")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 5: Run Preflight Unit & Protocol Parity Test Suite ===
import subprocess
import sys

subprocess.run(
    [
        sys.executable, "-m", "pytest", "-v",
        "tests/test_dtd_headline_baselines.py",
        "tests/test_dataset_adapters.py",
        "tests/test_dtd_dataset.py",
    ],
    check=True,
)
print("All DTD baseline, adapter, and dataset preflight tests PASSED!")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 6: Fast 2-Task Pilot Smoke Test (Isolated Directories) ===
import shutil
import subprocess
import sys
from pathlib import Path

pilots = [
    ("lwf", "configs/headline/lwf_dtd.yaml", "runs/headline_dtd/pilot_lwf"),
    ("official_lwf", "configs/headline/lwf_dtd_official_protocol.yaml", "runs/headline_dtd/pilot_official_lwf"),
    ("er", "configs/headline/er_dtd.yaml", "runs/headline_dtd/pilot_er"),
]
for tag, cfg_path, pilot_dir_str in pilots:
    pilot_dir = Path(pilot_dir_str)
    if pilot_dir.exists():
        shutil.rmtree(pilot_dir)
    print(f"\\n>>> Running 2-task GPU pilot for {tag} in isolated dir {pilot_dir}...")
    subprocess.run(
        [
            sys.executable, "scripts/train_baseline.py",
            "--config", cfg_path,
            "--output-dir", str(pilot_dir),
            "--run-name", f"pilot_dtd_{tag}",
            "--max-tasks", "2",
            "--limit-train-per-task", "20",
            "--limit-test-per-task", "20",
            "--device", "cuda",
        ],
        check=True,
    )
print("\\nAll 3 2-task GPU pilot checks completed cleanly with zero production contamination.")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 7: Full 10-Task GPU Run — Adapted LwF (Section IX.13: Recency Collapse) ===
import json
import subprocess
import sys
from pathlib import Path

FEATURE_CACHE_DIR = "/content/feature_cache" if Path("/content").exists() else "scratch/feature_cache"
EXPECTED_SPLIT_SHA = "42b13803af83abc522b7f814a6b9e450f884a4a7b675d77a0067491a72e79590"

out_dir = Path("runs/headline_dtd/lwf")
summary_file = out_dir / "summary.json"

if not summary_file.exists():
    cmd = [
        sys.executable, "scripts/train_baseline.py",
        "--config", "configs/headline/lwf_dtd.yaml",
        "--output-dir", str(out_dir),
        "--run-name", "headline_dtd_lwf_adapted_adamw_seed42",
        "--feature-cache-dir", FEATURE_CACHE_DIR,
        "--device", "cuda",
    ] + wandb_cli_args()
    subprocess.run(cmd, check=True)
else:
    print(f"Found existing run at {out_dir}. Verifying integrity...")

data = json.loads(summary_file.read_text(encoding="utf-8"))
mat = data.get("task_matrix", [])
total_mat_elements = sum(len(r) for r in mat)
if len(mat) != 10 or total_mat_elements != 55:
    raise RuntimeError(f"FATAL: Invalid matrix shape ({len(mat)} stages, {total_mat_elements} elements; expected 10 and 55)!")
split_sha = data["metadata"]["task_split_fingerprint"]["sha256"]
if split_sha != EXPECTED_SPLIT_SHA:
    raise RuntimeError(f"FATAL: Split SHA mismatch ({split_sha} vs {EXPECTED_SPLIT_SHA})")

m = data["metrics_summary"]
print(f"\\n[VERIFIED ADAPTED LwF] A_avg={m['a_avg_pct']:.4f}%, A_10={m['a_last_pct']:.4f}%, F_past={m['f_past_pct']:.4f}%, F_code={m['f_code_pct']:.4f}%")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 8: Full 10-Task GPU Run — Official-Protocol LwF (Section IX.15: Catastrophic Rigidity) ===
import json
import subprocess
import sys
from pathlib import Path

out_dir = Path("runs/headline_dtd/lwf_official_dtd")
summary_file = out_dir / "summary.json"

if not summary_file.exists():
    cmd = [
        sys.executable, "scripts/train_baseline.py",
        "--config", "configs/headline/lwf_dtd_official_protocol.yaml",
        "--output-dir", str(out_dir),
        "--run-name", "headline_dtd_lwf_official_protocol_seed42",
        "--feature-cache-dir", FEATURE_CACHE_DIR,
        "--device", "cuda",
    ] + wandb_cli_args()
    subprocess.run(cmd, check=True)
else:
    print(f"Found existing run at {out_dir}. Verifying integrity...")

data = json.loads(summary_file.read_text(encoding="utf-8"))
mat = data.get("task_matrix", [])
total_mat_elements = sum(len(r) for r in mat)
if len(mat) != 10 or total_mat_elements != 55:
    raise RuntimeError(f"FATAL: Invalid matrix shape ({len(mat)} stages, {total_mat_elements} elements; expected 10 and 55)!")
split_sha = data["metadata"]["task_split_fingerprint"]["sha256"]
if split_sha != EXPECTED_SPLIT_SHA:
    raise RuntimeError(f"FATAL: Split SHA mismatch ({split_sha} vs {EXPECTED_SPLIT_SHA})")

m = data["metrics_summary"]
print(f"\\n[VERIFIED OFFICIAL-PROTOCOL LwF] A_avg={m['a_avg_pct']:.4f}%, A_10={m['a_last_pct']:.4f}%, F_past={m['f_past_pct']:.4f}%, F_code={m['f_code_pct']:.4f}%")
print("Diagonal New-Task Plasticity R_(j,j) vs. Loss Weight 1/(t+1):")
for j, row in enumerate(mat):
    print(f"  Task {j + 1:02d} (w_new = {1.0 / (j + 1):.2f}): R_({j+1},{j+1}) = {row[j] * 100:.2f}%")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 9: Full 10-Task GPU Run — Adapted ER (Section IX.14: Replay Buffer M=5000) ===
import json
import subprocess
import sys
from pathlib import Path

out_dir = Path("runs/headline_dtd/er")
summary_file = out_dir / "summary.json"

if not summary_file.exists():
    cmd = [
        sys.executable, "scripts/train_baseline.py",
        "--config", "configs/headline/er_dtd.yaml",
        "--output-dir", str(out_dir),
        "--run-name", "headline_dtd_er_replay_buffer5000_seed42",
        "--feature-cache-dir", FEATURE_CACHE_DIR,
        "--device", "cuda",
    ] + wandb_cli_args()
    subprocess.run(cmd, check=True)
else:
    print(f"Found existing run at {out_dir}. Verifying integrity...")

data = json.loads(summary_file.read_text(encoding="utf-8"))
mat = data.get("task_matrix", [])
total_mat_elements = sum(len(r) for r in mat)
if len(mat) != 10 or total_mat_elements != 55:
    raise RuntimeError(f"FATAL: Invalid matrix shape ({len(mat)} stages, {total_mat_elements} elements; expected 10 and 55)!")
split_sha = data["metadata"]["task_split_fingerprint"]["sha256"]
if split_sha != EXPECTED_SPLIT_SHA:
    raise RuntimeError(f"FATAL: Split SHA mismatch ({split_sha} vs {EXPECTED_SPLIT_SHA})")

m = data["metrics_summary"]
print(f"\\n[VERIFIED ADAPTED ER] A_avg={m['a_avg_pct']:.4f}%, A_10={m['a_last_pct']:.4f}%, F_past={m['f_past_pct']:.4f}%, F_code={m['f_code_pct']:.4f}%")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 10: Verify Full F-OAL (A0 / C3-D) Anchor & Cross-Method Split Parity ===
import json
import subprocess
import sys
from pathlib import Path

foal_dir = Path("results/dtd/dtd_a0_full")
if not (foal_dir / "metrics.json").exists():
    print("F-OAL A0 anchor not found on disk; executing canonical DTD F-OAL A0 run...")
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", "configs/dtd/dtd_a0_full.yaml",
        "--feature-cache-dir", FEATURE_CACHE_DIR,
        "--output-dir", str(foal_dir),
        "--run-name", "dtd_a0_full_seed42",
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)

foal_meta = json.loads((foal_dir / "metadata.json").read_text(encoding="utf-8"))
foal_metrics = json.loads((foal_dir / "metrics.json").read_text(encoding="utf-8"))
assert foal_meta["task_split_fingerprint"]["sha256"] == EXPECTED_SPLIT_SHA

for name, run_path in [
    ("F-OAL A0", foal_dir / "metadata.json"),
    ("Adapted LwF", Path("runs/headline_dtd/lwf/metadata.json")),
    ("Official LwF", Path("runs/headline_dtd/lwf_official_dtd/metadata.json")),
    ("Adapted ER", Path("runs/headline_dtd/er/metadata.json")),
]:
    sha = json.loads(run_path.read_text(encoding="utf-8"))["task_split_fingerprint"]["sha256"]
    assert sha == EXPECTED_SPLIT_SHA, f"{name} split SHA mismatch: {sha}"
    print(f"  [OK] {name:<14} split SHA256 = {sha}")

print(
    f"\\n[VERIFIED FULL F-OAL A0] A_avg={foal_metrics['average_incremental_accuracy']*100:.4f}%, "
    f"A_10={foal_metrics['last_accuracy']*100:.4f}%, F_past={foal_metrics['forgetting_past_tasks']*100:.4f}%"
)"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 11: Generate Section IX Comparative Analysis Tables & 6 Figures ===
import subprocess
import sys
from pathlib import Path
from IPython.display import Image, display

subprocess.run(
    [sys.executable, "scripts/generate_dtd_headline_analysis.py", "--strict"],
    check=True,
)

analysis_dir = Path("results/headline_dtd_analysis")
for fig_name in [
    "fig1_dtd_headline_task_matrices.png",
    "fig2_dtd_headline_foal_vs_baselines.png",
    "fig3_dtd_headline_adapted_vs_official_lwf.png",
    "fig4_dtd_headline_accuracy_progression.png",
    "fig5_dtd_headline_paper_vs_reproduced.png",
    "fig6_dtd_headline_forgetting_comparison.png",
]:
    fig_path = analysis_dir / fig_name
    if fig_path.exists():
        print(f"\\n--- {fig_name} ---")
        display(Image(filename=str(fig_path)))"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 12: Sync Comparative Figures & Tables to W&B Cloud ===
import subprocess
import sys

sync_cmd = [
    sys.executable, "scripts/import_dtd_headline_to_wandb.py",
    "--results-root", "runs/headline_dtd",
    "--analysis-dir", "results/headline_dtd_analysis",
    "--project", WANDB_PROJECT,
    "--entity", WANDB_ENTITY,
    "--group", WANDB_GROUP,
]
if WANDB_BASE_URL:
    sync_cmd.extend(["--base-url", WANDB_BASE_URL])

if USE_WANDB:
    subprocess.run(sync_cmd, check=True)
else:
    print("Offline W&B mode active — running dry-run verification and printing sync command:")
    subprocess.run(sync_cmd + ["--dry-run"], check=True)
    print("To upload offline W&B runs later, run: wandb sync wandb/offline-run-*")"""
        )
    )

    cells.append(
        make_code(
            """# === CELL 13: Package Results Archive with SHA256SUMS & Auto-Download ===
import hashlib
import zipfile
from pathlib import Path

zip_name = "foal_dtd_headline_baselines_results.zip"
files_to_pack: list[Path] = []

for root_dir in [Path("runs/headline_dtd"), Path("results/headline_dtd_analysis")]:
    if root_dir.exists():
        for p in sorted(root_dir.rglob("*")):
            if p.is_file() and not p.name.endswith((".pt", ".pth")) and "pilot_" not in p.as_posix():
                files_to_pack.append(p)

for cfg_p in sorted(Path("configs/headline").glob("*dtd*.yaml")):
    files_to_pack.append(cfg_p)

manifest_lines = []
for p in files_to_pack:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_lines.append(f"{h}  {p.as_posix()}")

manifest_file = Path("runs/headline_dtd/SHA256SUMS.txt")
manifest_file.parent.mkdir(parents=True, exist_ok=True)
manifest_file.write_text("\\n".join(manifest_lines) + "\\n", encoding="utf-8")
files_to_pack.append(manifest_file)

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in files_to_pack:
        zf.write(p, p.as_posix())

zip_bytes = Path(zip_name).read_bytes()
zip_sha = hashlib.sha256(zip_bytes).hexdigest()
print("=" * 80)
print(f"ARCHIVE GENERATED: {zip_name} ({len(zip_bytes) / 1024:.2f} KB)")
print(f"SHA256           : {zip_sha}")
print("=" * 80)

try:
    from google.colab import files
    print(f"Triggering browser download for {zip_name}...")
    files.download(zip_name)
except Exception as exc:
    print(f"Manual download notice ({exc}): download '{zip_name}' from the Colab Files pane.")"""
        )
    )

    return {
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {
            "accelerator": "GPU",
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "cells": cells,
    }


def main() -> None:
    nb = build_notebook()
    out_path = ROOT_DIR / "notebooks" / "foal_dtd_headline_final.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote notebook to {out_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
