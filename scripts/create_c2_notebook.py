"""Generate notebooks/foal_c2_backbone_controlled.ipynb for Google Colab GPU execution."""
import json
from pathlib import Path


def create_c2_notebook():
    cells = []

    # Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# F-OAL — Controlled Pretrained ViT Backbone Alignment Experiment (C2)\n",
            "\n",
            "- **Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024)\n",
            "- **Author Official Repository:** [`https://github.com/liuyuchen-cz/F-OAL`](https://github.com/liuyuchen-cz/F-OAL) (Pinned Commit: `73e4ac1b4c965030973a2b77f05c57c2f44f49f5`)\n",
            "- **Group Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL)\n",
            "- **Target Branch:** `experiment/c2-backbone-alignment`\n",
            "- **Research Question (C2):** Isolate the effect of Pretrained ViT Checkpoint (ImageNet-1K only vs. ImageNet-21K + 1K fine-tune) while holding all other factors fixed:\n",
            "  - **B0 Baseline Reference:** `vit_base_patch16_224.augreg_in1k` (ImageNet-1K AugReg, C1 P1 setup)\n",
            "  - **B1 Official Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21K pre-trained, fine-tuned on ImageNet-1K)\n",
            "- **Strict Controls:**\n",
            "  - Projection Equality: Identical random projection matrix $P$ (seed 42, $D=1000$, `official_uniform`, identical SHA256)\n",
            "  - Preprocessing: Bicubic 224x224, Normalize(0.5, 0.5) (C1 setup, not changing to Bilinear [0, 1] until C3)\n",
            "  - Dataset & Split: CIFAR-100, 10 tasks x 10 classes, seed 42\n",
            "  - Optimization: Online Ridge RLS, batch size 10, gamma 1.0, float64"
        ]
    })

    # CELL 1: Environment Preflight
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 1: Environment Preflight ===\n",
            "import os\n",
            "import sys\n",
            "import shutil\n",
            "import platform\n",
            "import subprocess\n",
            "import torch\n",
            "\n",
            "print(f\"Python version   : {sys.version}\")\n",
            "print(f\"Platform         : {platform.platform()}\")\n",
            "print(f\"PyTorch version  : {torch.__version__}\")\n",
            "print(f\"CUDA available   : {torch.cuda.is_available()}\")\n",
            "\n",
            "if not torch.cuda.is_available():\n",
            "    raise RuntimeError(\n",
            "        \"FATAL: CUDA GPU is not available! Please switch Colab runtime to GPU (T4 / L4 / A100) before running.\"\n",
            "    )\n",
            "\n",
            "device_name = torch.cuda.get_device_name(0)\n",
            "device_props = torch.cuda.get_device_properties(0)\n",
            "vram_gb = device_props.total_memory / (1024 ** 3)\n",
            "print(f\"GPU Device       : {device_name}\")\n",
            "print(f\"CUDA version     : {torch.version.cuda}\")\n",
            "print(f\"Total VRAM       : {vram_gb:.2f} GB\")\n",
            "print(f\"Compute Cap.     : {device_props.major}.{device_props.minor}\")\n",
            "\n",
            "free_bytes = shutil.disk_usage(\".\").free\n",
            "print(f\"Free Disk Space  : {free_bytes / (1024**3):.2f} GB\")\n"
        ]
    })

    # CELL 2: Repository Clone
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 2: Clone Group Repository & Checkout C2 Branch ===\n",
            "from pathlib import Path\n",
            "\n",
            "REPO_DIR = Path(\"/content/F-OAL\")\n",
            "BRANCH_NAME = \"experiment/c2-backbone-alignment\"\n",
            "\n",
            "if REPO_DIR.exists():\n",
            "    print(f\"Removing existing directory {REPO_DIR} for clean clone...\")\n",
            "    shutil.rmtree(REPO_DIR)\n",
            "\n",
            "subprocess.run([\n",
            "    \"git\", \"clone\", \"-b\", BRANCH_NAME,\n",
            "    \"https://github.com/caoban123/F-OAL.git\", str(REPO_DIR)\n",
            "], check=True)\n",
            "\n",
            "commit_hash = subprocess.check_output(\n",
            "    [\"git\", \"rev-parse\", \"HEAD\"], cwd=str(REPO_DIR), text=True\n",
            ").strip()\n",
            "print(f\"SUCCESS: Cloned {BRANCH_NAME} at commit {commit_hash}\")\n",
            "\n",
            "os.chdir(str(REPO_DIR))\n",
            "if str(REPO_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n"
        ]
    })

    # CELL 3: Dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 3: Install Project Dependencies ===\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", \"requirements.txt\"], cwd=str(REPO_DIR), check=True)\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-e\", \".\"], cwd=str(REPO_DIR), check=True)\n",
            "\n",
            "import timm\n",
            "import datasets\n",
            "import wandb\n",
            "print(f\"timm version     : {timm.__version__}\")\n",
            "print(f\"datasets version : {datasets.__version__}\")\n",
            "print(f\"wandb version    : {wandb.__version__}\")\n",
            "\n",
            "print(\"Running preflight unit tests...\")\n",
            "subprocess.run([sys.executable, \"-m\", \"pytest\", \"-v\", \"tests/test_c2_backbone.py\"], cwd=str(REPO_DIR), check=True)\n"
        ]
    })

    # CELL 4: Dataset
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 4: Prepare CIFAR-100 Dataset ===\n",
            "from src.foal.datasets import CIFAR100OnlineDataset\n",
            "\n",
            "DATA_DIR = REPO_DIR / \"artifacts/data\"\n",
            "DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(\"Initializing & verifying CIFAR-100 dataset...\")\n",
            "ds_handler = CIFAR100OnlineDataset(root=str(DATA_DIR), seed=42)\n",
            "train_src = ds_handler.get_train_source()\n",
            "test_src = ds_handler.get_test_source()\n",
            "print(f\"Train samples count: {len(train_src)}\")\n",
            "print(f\"Test samples count : {len(test_src)}\")\n",
            "assert len(train_src) == 50000, f\"Expected 50000 train samples, got {len(train_src)}\"\n",
            "assert len(test_src) == 10000, f\"Expected 10000 test samples, got {len(test_src)}\"\n"
        ]
    })

    # CELL 5: Configuration Validation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 5: Configuration & Projection Equality Validation ===\n",
            "import yaml\n",
            "from src.foal.projection import SmoothProjection\n",
            "\n",
            "cfg_b0_p = REPO_DIR / \"configs/c2_b0_augreg_in1k.yaml\"\n",
            "cfg_b1_p = REPO_DIR / \"configs/c2_b1_augreg2_in21k_ft_in1k.yaml\"\n",
            "\n",
            "with open(cfg_b0_p, \"r\", encoding=\"utf-8\") as f:\n",
            "    cfg_b0 = yaml.safe_load(f)\n",
            "with open(cfg_b1_p, \"r\", encoding=\"utf-8\") as f:\n",
            "    cfg_b1 = yaml.safe_load(f)\n",
            "\n",
            "print(f\"B0 Backbone: {cfg_b0['model']['backbone']}\")\n",
            "print(f\"B1 Backbone: {cfg_b1['model']['backbone']}\")\n",
            "\n",
            "# Verify Projection Equality\n",
            "proj_b0 = SmoothProjection(768, 1000, seed=42, initialization='official_uniform')\n",
            "proj_b1 = SmoothProjection(768, 1000, seed=42, initialization='official_uniform')\n",
            "assert torch.equal(proj_b0.weight, proj_b1.weight), \"FATAL: Projection weights differ between B0 and B1!\"\n",
            "p_sha = proj_b0.get_metadata()['matrix_sha256']\n",
            "print(f\"Projection Equality VERIFIED! Both use matrix P SHA256: {p_sha[:16]}...\")\n"
        ]
    })

    # CELL 6: Checkpoint Verification
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 6: Verify Both Pretrained ViT Checkpoints ===\n",
            "from src.foal.backbone import FrozenViT\n",
            "\n",
            "print(\"Loading & verifying B0 Checkpoint (augreg_in1k)...\")\n",
            "b0_vit = FrozenViT(\"vit_base_patch16_224.augreg_in1k\", pretrained=True, feature_fusion=True)\n",
            "print(f\"  B0 Tag: {b0_vit.pretrained_tag}\")\n",
            "print(f\"  B0 Feature Dim: {b0_vit.feature_dim}\")\n",
            "print(f\"  B0 Num Blocks: {len(b0_vit.network.blocks)}\")\n",
            "assert b0_vit.pretrained_tag == \"augreg_in1k\"\n",
            "del b0_vit\n",
            "\n",
            "print(\"\nLoading & verifying B1 Checkpoint (augreg2_in21k_ft_in1k)...\")\n",
            "b1_vit = FrozenViT(\"vit_base_patch16_224.augreg2_in21k_ft_in1k\", pretrained=True, feature_fusion=True)\n",
            "print(f\"  B1 Tag: {b1_vit.pretrained_tag}\")\n",
            "print(f\"  B1 Feature Dim: {b1_vit.feature_dim}\")\n",
            "print(f\"  B1 Num Blocks: {len(b1_vit.network.blocks)}\")\n",
            "assert b1_vit.pretrained_tag == \"augreg2_in21k_ft_in1k\"\n",
            "del b1_vit\n",
            "print(\"\nSUCCESS: Both pretrained ViT checkpoints verified on disk!\")\n"
        ]
    })

    # CELL 7: Diagnostics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 7: Matched-Input Backbone Diagnostics ===\n",
            "DIAG_OUT_DIR = REPO_DIR / \"results/c2_diagnostics\"\n",
            "DIAG_OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "subprocess.run([\n",
            "    sys.executable, \"-m\", \"scripts.run_c2_diagnostics\",\n",
            "    \"--output-dir\", str(DIAG_OUT_DIR),\n",
            "    \"--device\", \"cuda\",\n",
            "    \"--num-samples\", \"200\"\n",
            "], cwd=str(REPO_DIR), check=True)\n"
        ]
    })

    # CELL 8: Pilot
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 8: Level A Pilot (2 Tasks) for B0 and B1 ===\n",
            "PILOT_DIR = REPO_DIR / \"results/c2_pilot\"\n",
            "PILOT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(\"--- Running B0 Pilot (2 Tasks) ---\")\n",
            "subprocess.run([\n",
            "    sys.executable, \"-m\", \"scripts.train\",\n",
            "    \"--config\", \"configs/c2_b0_augreg_in1k.yaml\",\n",
            "    \"--output-dir\", str(PILOT_DIR / \"pilot_b0\"),\n",
            "    \"--max-tasks\", \"2\",\n",
            "    \"--limit-train-per-task\", \"100\",\n",
            "    \"--limit-test-per-task\", \"100\",\n",
            "    \"--device\", \"cuda\"\n",
            "], cwd=str(REPO_DIR), check=True)\n",
            "\n",
            "print(\"--- Running B1 Pilot (2 Tasks) ---\")\n",
            "subprocess.run([\n",
            "    sys.executable, \"-m\", \"scripts.train\",\n",
            "    \"--config\", \"configs/c2_b1_augreg2_in21k_ft_in1k.yaml\",\n",
            "    \"--output-dir\", str(PILOT_DIR / \"pilot_b1\"),\n",
            "    \"--max-tasks\", \"2\",\n",
            "    \"--limit-train-per-task\", \"100\",\n",
            "    \"--limit-test-per-task\", \"100\",\n",
            "    \"--device\", \"cuda\"\n",
            "], cwd=str(REPO_DIR), check=True)\n",
            "print(\"SUCCESS: Level A Pilot completed for both B0 and B1!\")\n"
        ]
    })

    # CELL 9: Full GPU Experiment
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 9: Full 10-Task CIFAR-100 GPU Experiment ===\n",
            "# NOTE: Ready for Colab execution. Set use_wandb flag and credentials as needed.\n",
            "FULL_OUT_DIR = REPO_DIR / \"results/c2_cifar100_full\"\n",
            "CACHE_DIR = REPO_DIR / \"results/c2_feature_cache\"\n",
            "FULL_OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "CACHE_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Setup W&B credentials if available in Colab secrets\n",
            "use_wandb = False\n",
            "try:\n",
            "    from google.colab import userdata\n",
            "    wb_key = userdata.get('WANDB_API_KEY')\n",
            "    if wb_key:\n",
            "        wandb.login(key=wb_key)\n",
            "        use_wandb = True\n",
            "except Exception:\n",
            "    pass\n",
            "if not use_wandb and os.environ.get('WANDB_API_KEY'):\n",
            "    wandb.login(key=os.environ['WANDB_API_KEY'])\n",
            "    use_wandb = True\n",
            "\n",
            "runs = [\n",
            "    (\"configs/c2_b0_augreg_in1k.yaml\", \"b0_augreg_in1k\", \"c2_cifar100_b0_augreg_in1k_seed42\"),\n",
            "    (\"configs/c2_b1_augreg2_in21k_ft_in1k.yaml\", \"b1_augreg2_in21k_ft_in1k\", \"c2_cifar100_b1_augreg2_in21k_ft_in1k_seed42\"),\n",
            "]\n",
            "\n",
            "for cfg_file, run_subdir, run_name in runs:\n",
            "    out_path = FULL_OUT_DIR / run_subdir\n",
            "    print(f\"\\n{'='*70}\\nRUNNING: {run_name} -> {out_path}\\n{'='*70}\")\n",
            "    cmd = [\n",
            "        sys.executable, \"-m\", \"scripts.train\",\n",
            "        \"--config\", cfg_file,\n",
            "        \"--output-dir\", str(out_path),\n",
            "        \"--device\", \"cuda\",\n",
            "        \"--run-name\", run_name,\n",
            "        \"--feature-cache-dir\", str(CACHE_DIR),\n",
            "    ]\n",
            "    if use_wandb:\n",
            "        cmd.extend([\n",
            "            \"--wandb\",\n",
            "            \"--wandb-project\", \"hcmus-foal-reproduction\",\n",
            "            \"--wandb-group\", \"c2_backbone_alignment\",\n",
            "        ])\n",
            "    else:\n",
            "        cmd.extend([\n",
            "            \"--wandb\", \"--wandb-offline\",\n",
            "            \"--wandb-project\", \"hcmus-foal-reproduction\",\n",
            "            \"--wandb-group\", \"c2_backbone_alignment\",\n",
            "        ])\n",
            "    subprocess.run(cmd, cwd=str(REPO_DIR), check=True)\n",
            "print(\"\\nSUCCESS: Both full 10-task experiments completed!\")\n"
        ]
    })

    # CELL 10: Comparative Results
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 10: Results Recalculation & Comparison Table ===\n",
            "import json\n",
            "import pandas as pd\n",
            "\n",
            "b0_metrics = json.loads((FULL_OUT_DIR / \"b0_augreg_in1k/metrics.json\").read_text(encoding='utf-8'))\n",
            "b1_metrics = json.loads((FULL_OUT_DIR / \"b1_augreg2_in21k_ft_in1k/metrics.json\").read_text(encoding='utf-8'))\n",
            "\n",
            "summary = [\n",
            "    {\n",
            "        \"Configuration\": \"B0: augreg_in1k (ImageNet-1K)\",\n",
            "        \"Avg Incr Acc (%)\": f\"{b0_metrics['average_incremental_accuracy']*100:.2f}%\",\n",
            "        \"Last Acc (%)\": f\"{b0_metrics['last_accuracy']*100:.2f}%\",\n",
            "        \"Forgetting Past (%)\": f\"{b0_metrics.get('forgetting_past_tasks', b0_metrics.get('forgetting'))*100:.2f}%\",\n",
            "        \"Forgetting Code (%)\": f\"{b0_metrics.get('forgetting_official_code', 0.0)*100:.2f}%\",\n",
            "    },\n",
            "    {\n",
            "        \"Configuration\": \"B1: augreg2_in21k_ft_in1k (Official Checkpoint)\",\n",
            "        \"Avg Incr Acc (%)\": f\"{b1_metrics['average_incremental_accuracy']*100:.2f}%\",\n",
            "        \"Last Acc (%)\": f\"{b1_metrics['last_accuracy']*100:.2f}%\",\n",
            "        \"Forgetting Past (%)\": f\"{b1_metrics.get('forgetting_past_tasks', b1_metrics.get('forgetting'))*100:.2f}%\",\n",
            "        \"Forgetting Code (%)\": f\"{b1_metrics.get('forgetting_official_code', 0.0)*100:.2f}%\",\n",
            "    },\n",
            "]\n",
            "df_res = pd.DataFrame(summary)\n",
            "print(\"\\n--- C2 EXPERIMENTAL RESULTS COMPARISON ---\")\n",
            "print(df_res.to_string(index=False))\n"
        ]
    })

    # CELL 11: Export & Download
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 11: Package Results into ZIP & Download ===\n",
            "import zipfile\n",
            "\n",
            "ZIP_PATH = Path(\"/content/foal_c2_backbone_results.zip\")\n",
            "if ZIP_PATH.exists():\n",
            "    ZIP_PATH.unlink()\n",
            "\n",
            "with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:\n",
            "    # Package results directory\n",
            "    for p in REPO_DIR.glob(\"results/c2_*/**/*\"):\n",
            "        if p.is_file() and not p.name.endswith(\".pyc\"):\n",
            "            zf.write(p, arcname=p.relative_to(REPO_DIR))\n",
            "    # Package wandb logs if present\n",
            "    for p in REPO_DIR.glob(\"wandb/offline-run-*/**/*\"):\n",
            "        if p.is_file():\n",
            "            zf.write(p, arcname=p.relative_to(REPO_DIR))\n",
            "\n",
            "print(f\"Packaged {ZIP_PATH} (Size: {ZIP_PATH.stat().st_size / (1024**2):.2f} MB)\")\n",
            "\n",
            "try:\n",
            "    from google.colab import files\n",
            "    files.download(str(ZIP_PATH))\n",
            "    print(\"Triggered automatic browser download.\")\n",
            "except Exception as e:\n",
            "    print(f\"Note: Manual download available at {ZIP_PATH}\")\n"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_p = Path("notebooks/foal_c2_backbone_controlled.ipynb")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated C2 notebook at: {out_p}")


if __name__ == "__main__":
    create_c2_notebook()
