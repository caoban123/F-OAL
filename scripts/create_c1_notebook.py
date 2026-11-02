"""Generate notebooks/foal_c1_projection_controlled.ipynb for Google Colab GPU execution."""
import json
from pathlib import Path


def create_c1_notebook():
    cells = []

    # Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# F-OAL — Controlled Smooth Projection Experiment (C1) & Metric Alignment (C0)\n",
            "\n",
            "- **Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024)\n",
            "- **Author Official Repository:** [`https://github.com/liuyuchen-cz/F-OAL`](https://github.com/liuyuchen-cz/F-OAL) (Pinned Commit: `73e4ac1b4c965030973a2b77f05c57c2f44f49f5`)\n",
            "- **Group Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL)\n",
            "- **Target Branch:** `experiment/c1-projection-initialization`\n",
            "- **Research Purpose:** Empirically isolate and quantify the impact of Smooth Projection initialization on the ~12% accuracy gap observed between our initial run (79.33%) and official code (91.13%):\n",
            "  - **P0 (`gaussian_std1`):** Historical Gaussian initialization $P_{ij} \\sim \\mathcal{N}(0, 1)$, variance = 1.0.\n",
            "  - **P1 (`official_uniform`):** Official author initialization $P_{ij} \\sim \\mathcal{U}(-1/\\sqrt{d}, +1/\\sqrt{d})$, variance = $1/(3d) \\approx 0.000434$.\n",
            "  - **P2 (`gaussian_matched_variance`):** Control Gaussian with population variance matched to P1 ($1/(3d)$).\n",
            "- **Scope & Constraints:** 10 tasks x 10 classes, online one-pass stream, batch size 10, deterministic class order (seed 42), frozen ViT backbone features."
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
            "total_disk, used_disk, free_disk = shutil.disk_usage(\"/\")\n",
            "print(f\"Free Disk Space  : {free_disk / (1024 ** 3):.2f} GB / {total_disk / (1024 ** 3):.2f} GB\")\n",
            "\n",
            "print(\"\\n--- nvidia-smi ---\")\n",
            "subprocess.run([\"nvidia-smi\"], check=False)\n"
        ]
    })

    # CELL 2: Clone Group Repository & Checkout Branch
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 2: Clone Group Repository & Checkout C1 Branch ===\n",
            "import pathlib\n",
            "import shutil\n",
            "import subprocess\n",
            "\n",
            "REPO_URL = \"https://github.com/caoban123/F-OAL.git\"\n",
            "TARGET_BRANCH = \"experiment/c1-projection-initialization\"\n",
            "BASE_DIR = pathlib.Path(\"/content\") if pathlib.Path(\"/content\").exists() else pathlib.Path.cwd()\n",
            "REPO_DIR = BASE_DIR / \"F-OAL\"\n",
            "\n",
            "if not (REPO_DIR / \".git\").exists():\n",
            "    if REPO_DIR.exists():\n",
            "        shutil.rmtree(REPO_DIR)\n",
            "    print(f\"Cloning {REPO_URL} into {REPO_DIR}...\")\n",
            "    subprocess.run([\"git\", \"clone\", REPO_URL, str(REPO_DIR)], check=True)\n",
            "else:\n",
            "    print(f\"Repository exists at {REPO_DIR}, fetching...\")\n",
            "    subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"fetch\", \"--all\"], check=True)\n",
            "\n",
            "subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"checkout\", TARGET_BRANCH], check=True)\n",
            "subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"pull\", \"origin\", TARGET_BRANCH], check=False)\n",
            "current_sha = subprocess.check_output([\"git\", \"-C\", str(REPO_DIR), \"rev-parse\", \"HEAD\"], text=True).strip()\n",
            "print(f\"Active Branch : {TARGET_BRANCH}\")\n",
            "print(f\"Commit SHA    : {current_sha}\")\n",
            "\n",
            "# Verify that C1 diagnostics and configs exist\n",
            "assert (REPO_DIR / \"src/foal/diagnostics.py\").exists(), \"Missing src/foal/diagnostics.py!\"\n",
            "assert (REPO_DIR / \"configs/c1_p0_gaussian_std1.yaml\").exists(), \"Missing configs/c1_p0_gaussian_std1.yaml!\"\n",
            "assert (REPO_DIR / \"configs/c1_p1_official_uniform.yaml\").exists(), \"Missing configs/c1_p1_official_uniform.yaml!\"\n",
            "assert (REPO_DIR / \"configs/c1_p2_gaussian_matched_variance.yaml\").exists(), \"Missing configs/c1_p2_gaussian_matched_variance.yaml!\"\n",
            "print(\"All required C1 modules and configurations are verified present.\")\n"
        ]
    })

    # CELL 3: Install & Verify Dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 3: Install & Verify Dependencies ===\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            "packages = [\n",
            "    \"timm>=0.9.7\",\n",
            "    \"scikit-learn\",\n",
            "    \"scipy\",\n",
            "    \"PyYAML\",\n",
            "    \"tqdm\",\n",
            "    \"wandb\",\n",
            "    \"matplotlib\",\n",
            "    \"pytest\",\n",
            "]\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", *packages], check=True)\n",
            "\n",
            "import timm\n",
            "import yaml\n",
            "import wandb\n",
            "import matplotlib\n",
            "\n",
            "print(f\"timm version       : {timm.__version__}\")\n",
            "print(f\"PyYAML version     : {yaml.__version__}\")\n",
            "print(f\"W&B version        : {wandb.__version__}\")\n",
            "print(f\"Matplotlib version : {matplotlib.__version__}\")\n"
        ]
    })

    # CELL 4: Prepare CIFAR-100 & Validate C1 Configs
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 4: Prepare CIFAR-100 & Validate C1 Configs ===\n",
            "import torchvision.datasets as tv_datasets\n",
            "import pathlib\n",
            "import yaml\n",
            "\n",
            "BASE_DIR = pathlib.Path(\"/content\") if pathlib.Path(\"/content\").exists() else pathlib.Path.cwd()\n",
            "REPO_DIR = BASE_DIR / \"F-OAL\"\n",
            "data_root = REPO_DIR / \"data\"\n",
            "data_root.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(\"Downloading / verifying CIFAR-100 dataset...\")\n",
            "train_set = tv_datasets.CIFAR100(root=str(data_root), train=True, download=True)\n",
            "test_set = tv_datasets.CIFAR100(root=str(data_root), train=False, download=True)\n",
            "\n",
            "assert len(train_set.data) == 50000\n",
            "assert len(test_set.data) == 10000\n",
            "assert len(set(train_set.targets)) == 100\n",
            "print(f\"CIFAR-100 ready at {data_root}.\")\n",
            "\n",
            "# Validate the three C1 configurations\n",
            "configs = {}\n",
            "for p_id in [\"p0_gaussian_std1\", \"p1_official_uniform\", \"p2_gaussian_matched_variance\"]:\n",
            "    cfg_file = REPO_DIR / f\"configs/c1_{p_id}.yaml\"\n",
            "    with open(cfg_file, \"r\", encoding=\"utf-8\") as f:\n",
            "        configs[p_id] = yaml.safe_load(f)\n",
            "    print(f\"Loaded {cfg_file.name}: projection.initialization = {configs[p_id]['projection']['initialization']}\")\n",
            "\n",
            "assert configs[\"p0_gaussian_std1\"][\"projection\"][\"initialization\"] == \"gaussian_std1\"\n",
            "assert configs[\"p1_official_uniform\"][\"projection\"][\"initialization\"] == \"official_uniform\"\n",
            "assert configs[\"p2_gaussian_matched_variance\"][\"projection\"][\"initialization\"] == \"gaussian_matched_variance\"\n",
            "print(\"All 3 C1 configurations validated successfully.\")\n"
        ]
    })

    # CELL 5: Class Order & RNG Isolation Verification
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 5: Class Order & RNG Isolation Verification ===\n",
            "import sys\n",
            "BASE_DIR = pathlib.Path(\"/content\") if pathlib.Path(\"/content\").exists() else pathlib.Path.cwd()\n",
            "REPO_DIR = BASE_DIR / \"F-OAL\"\n",
            "if str(REPO_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n",
            "\n",
            "from src.foal.datasets import make_class_order\n",
            "from src.foal.projection import SmoothProjection\n",
            "\n",
            "order_42 = make_class_order(100, 42)\n",
            "expected_first_10 = [59, 21, 56, 18, 33, 42, 50, 27, 92, 71]\n",
            "assert order_42[:10] == expected_first_10, f\"Expected {expected_first_10}, got {order_42[:10]}\"\n",
            "print(\"Class order verified bitwise identical to historical Seed 42 run.\")\n",
            "\n",
            "# Run unit test suite\n",
            "subprocess.run([sys.executable, \"-m\", \"pytest\", \"tests/test_projection.py\", \"tests/test_metrics.py\", \"-v\"], cwd=str(REPO_DIR), check=True)\n",
            "print(\"Unit tests passed 100%.\")\n"
        ]
    })

    # CELL 6: Matched-Input Projection Diagnostics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 6: Matched-Input Projection Diagnostics (Section 9) ===\n",
            "# Evaluates P0, P1, and P2 on the exact same 200 real CIFAR-100 ViT features\n",
            "import json\n",
            "import pandas as pd\n",
            "\n",
            "DIAG_OUT_DIR = REPO_DIR / \"results/c1_diagnostics\"\n",
            "DIAG_OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "subprocess.run([\n",
            "    sys.executable, \"-m\", \"scripts.run_c1_diagnostics\",\n",
            "    \"--output-dir\", str(DIAG_OUT_DIR),\n",
            "    \"--device\", \"cuda\",\n",
            "], cwd=str(REPO_DIR), check=True)\n",
            "\n",
            "diag_report_path = DIAG_OUT_DIR / \"c1_matched_input_diagnostics.json\"\n",
            "with open(diag_report_path, \"r\", encoding=\"utf-8\") as f:\n",
            "    diag_data = json.load(f)\n",
            "\n",
            "print(\"\\n--- DIAGNOSTICS SUMMARY TABLE ---\")\n",
            "summary_rows = []\n",
            "for init_name, cond in diag_data[\"configurations\"].items():\n",
            "    summary_rows.append({\n",
            "        \"Configuration\": init_name,\n",
            "        \"Weight Std\": f\"{cond['projection_metadata']['measured_std']:.6f}\",\n",
            "        \"u=hP Std\": f\"{cond['pre_sigmoid_u']['std']:.4f}\",\n",
            "        \"z=sig(u) Std\": f\"{cond['post_sigmoid_z']['std']:.4f}\",\n",
            "        \"Saturation (<0.01 | >0.99)\": f\"{cond['post_sigmoid_z']['total_saturation_fraction']*100:.2f}%\",\n",
            "        \"Effective Rank (G)\": f\"{cond['numerical_stability']['effective_rank_unregularized']:.2f}\",\n",
            "        \"Cond Number (G+gamma*I)\": f\"{cond['numerical_stability']['regularized_gram_condition_number']:.2f}\",\n",
            "    })\n",
            "df_diag = pd.DataFrame(summary_rows)\n",
            "display(df_diag)\n"
        ]
    })

    # CELL 7: Level A 2-Task Pilot Verification
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 7: Level A 2-Task Pilot Verification ===\n",
            "# Verifies pipeline functionality and feature cache equivalence on GPU\n",
            "PILOT_DIR = REPO_DIR / \"results/c1_pilot\"\n",
            "PILOT_CACHE = REPO_DIR / \"results/c1_pilot_feature_cache\"\n",
            "\n",
            "for p_id in [\"p0_gaussian_std1\", \"p1_official_uniform\", \"p2_gaussian_matched_variance\"]:\n",
            "    out_p = PILOT_DIR / p_id\n",
            "    cfg_file = REPO_DIR / f\"configs/c1_{p_id}.yaml\"\n",
            "    subprocess.run([\n",
            "        sys.executable, \"-m\", \"scripts.train\",\n",
            "        \"--config\", str(cfg_file),\n",
            "        \"--output-dir\", str(out_p),\n",
            "        \"--max-tasks\", \"2\",\n",
            "        \"--limit-train-per-task\", \"100\",\n",
            "        \"--limit-test-per-task\", \"100\",\n",
            "        \"--device\", \"cuda\",\n",
            "        \"--feature-cache-dir\", str(PILOT_CACHE),\n",
            "    ], cwd=str(REPO_DIR), check=True)\n",
            "\n",
            "print(\"Level A 2-task pilots completed successfully on GPU.\")\n"
        ]
    })

    # CELL 8: Level B Full 10-Task CIFAR-100 Controlled C1 Runs
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 8: Level B Full 10-Task CIFAR-100 Controlled C1 Runs ===\n",
            "# Runs full 10-task CIFAR-100 with shared feature cache across P0, P1, P2\n",
            "import os\n",
            "FULL_C1_RESULTS_DIR = REPO_DIR / \"results/c1_cifar100_full\"\n",
            "FULL_C1_CACHE_DIR = BASE_DIR / \"c1_full_feature_cache\"\n",
            "FULL_C1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
            "FULL_C1_CACHE_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Configure W&B login if key is available, else run offline\n",
            "use_wandb = False\n",
            "try:\n",
            "    from google.colab import userdata\n",
            "    wb_key = userdata.get('WANDB_API_KEY')\n",
            "    if wb_key:\n",
            "        wandb.login(key=wb_key)\n",
            "        use_wandb = True\n",
            "        print(\"W&B login successful via Colab Secrets.\")\n",
            "except Exception:\n",
            "    if os.environ.get('WANDB_API_KEY'):\n",
            "        wandb.login(key=os.environ['WANDB_API_KEY'])\n",
            "        use_wandb = True\n",
            "        print(\"W&B login successful via environment variable.\")\n",
            "    else:\n",
            "        print(\"No W&B API key detected; runs will log offline to disk.\")\n",
            "\n",
            "c1_experiments = [\n",
            "    (\"p0_gaussian_std1\", \"c1_cifar100_gaussian_std1_seed42\"),\n",
            "    (\"p1_official_uniform\", \"c1_cifar100_official_uniform_seed42\"),\n",
            "    (\"p2_gaussian_matched_variance\", \"c1_cifar100_gaussian_matched_variance_seed42\"),\n",
            "]\n",
            "\n",
            "for p_id, run_name in c1_experiments:\n",
            "    out_dir = FULL_C1_RESULTS_DIR / p_id\n",
            "    cfg_file = REPO_DIR / f\"configs/c1_{p_id}.yaml\"\n",
            "    print(f\"\\n{'='*70}\\nSTARTING FULL C1 EXPERIMENT: {run_name}\\n{'='*70}\")\n",
            "    \n",
            "    cmd = [\n",
            "        sys.executable, \"-m\", \"scripts.train\",\n",
            "        \"--config\", str(cfg_file),\n",
            "        \"--output-dir\", str(out_dir),\n",
            "        \"--run-name\", run_name,\n",
            "        \"--device\", \"cuda\",\n",
            "        \"--feature-cache-dir\", str(FULL_C1_CACHE_DIR),\n",
            "        \"--experiment-type\", \"controlled-c1\",\n",
            "    ]\n",
            "    if use_wandb:\n",
            "        cmd.extend([\n",
            "            \"--wandb\",\n",
            "            \"--wandb-project\", \"hcmus-foal-reproduction\",\n",
            "            \"--wandb-group\", \"c1_projection_initialization\",\n",
            "        ])\n",
            "    else:\n",
            "        cmd.extend([\n",
            "            \"--wandb\",\n",
            "            \"--wandb-offline\",\n",
            "            \"--wandb-project\", \"hcmus-foal-reproduction\",\n",
            "            \"--wandb-group\", \"c1_projection_initialization\",\n",
            "        ])\n",
            "    \n",
            "    subprocess.run(cmd, cwd=str(REPO_DIR), check=True)\n",
            "    print(f\"Completed {run_name}.\")\n"
        ]
    })

    # CELL 9: C0 Metric Alignment & C1 Discrepancy Comparative Analysis
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 9: Comparative Analysis & Visualization ===\n",
            "import json\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "results_summary = []\n",
            "runs_meta = {\n",
            "    \"P0 (Gaussian std=1.0)\": FULL_C1_RESULTS_DIR / \"p0_gaussian_std1\",\n",
            "    \"P1 (Official Uniform)\": FULL_C1_RESULTS_DIR / \"p1_official_uniform\",\n",
            "    \"P2 (Gaussian matched var)\": FULL_C1_RESULTS_DIR / \"p2_gaussian_matched_variance\",\n",
            "}\n",
            "\n",
            "curves = {}\n",
            "for label, r_dir in runs_meta.items():\n",
            "    metrics_f = r_dir / \"metrics.json\"\n",
            "    if metrics_f.exists():\n",
            "        data = json.loads(metrics_f.read_text(encoding=\"utf-8\"))\n",
            "        results_summary.append({\n",
            "            \"Condition\": label,\n",
            "            \"Avg Incr Acc (%)\": f\"{data['average_incremental_accuracy']*100:.2f}%\",\n",
            "            \"Last Acc (%)\": f\"{data['last_accuracy']*100:.2f}%\",\n",
            "            \"Fgt Past-Task Eq(17-18) (%)\": f\"{data['forgetting_past_tasks']*100:.2f}%\",\n",
            "            \"Fgt Official Code (%)\": f\"{data['forgetting_official_code']*100:.2f}%\",\n",
            "            \"Feat Extr Time (s)\": data.get('feature_extraction_seconds', 'N/A'),\n",
            "            \"RLS & Eval Time (s)\": data.get('rls_and_eval_seconds', 'N/A'),\n",
            "        })\n",
            "        curves[label] = data.get('stage_accuracies', [])\n",
            "\n",
            "# Add Official Seed 0 Reference for comparison\n",
            "results_summary.append({\n",
            "    \"Condition\": \"[Reference] Official Author Seed 0\",\n",
            "    \"Avg Incr Acc (%)\": \"91.13%\",\n",
            "    \"Last Acc (%)\": \"86.60%\",\n",
            "    \"Fgt Past-Task Eq(17-18) (%)\": \"5.17%\",\n",
            "    \"Fgt Official Code (%)\": \"4.69%\",\n",
            "    \"Feat Extr Time (s)\": \"—\",\n",
            "    \"RLS & Eval Time (s)\": \"~1450 s\",\n",
            "})\n",
            "\n",
            "df_results = pd.DataFrame(results_summary)\n",
            "print(\"\\n--- C1 CONTROLLED EXPERIMENT COMPARISON TABLE ---\")\n",
            "display(df_results)\n",
            "\n",
            "# Plot Stage Accuracy Curves\n",
            "plt.figure(figsize=(9, 5))\n",
            "for label, accs in curves.items():\n",
            "    plt.plot(range(1, len(accs) + 1), [a * 100 for a in accs], marker='o', label=label)\n",
            "plt.title(\"CIFAR-100 Incremental Accuracy across Stages (C1 Controlled Experiment)\")\n",
            "plt.xlabel(\"Task / Stage\")\n",
            "plt.ylabel(\"Accuracy (%)\")\n",
            "plt.grid(True, linestyle=\"--\", alpha=0.6)\n",
            "plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig(REPO_DIR / \"results/c1_stage_accuracy_comparison.png\", dpi=300)\n",
            "plt.show()\n"
        ]
    })

    # CELL 10: Artifact Validation & Packaging
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === CELL 10: Artifact Validation & Packaging ===\n",
            "import zipfile\n",
            "import pathlib\n",
            "\n",
            "ZIP_PATH = BASE_DIR / \"foal_c1_projection_results.zip\"\n",
            "dirs_to_package = [\n",
            "    REPO_DIR / \"results/c1_diagnostics\",\n",
            "    REPO_DIR / \"results/c1_pilot\",\n",
            "    REPO_DIR / \"results/c1_cifar100_full\",\n",
            "]\n",
            "\n",
            "print(f\"Creating {ZIP_PATH}...\")\n",
            "with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:\n",
            "    for target_dir in dirs_to_package:\n",
            "        if target_dir.exists():\n",
            "            for file_path in target_dir.rglob(\"*\"):\n",
            "                if file_path.is_file():\n",
            "                    arcname = file_path.relative_to(REPO_DIR)\n",
            "                    zipf.write(file_path, arcname=str(arcname))\n",
            "                    print(f\"  Added: {arcname}\")\n",
            "\n",
            "# Verify ZIP integrity\n",
            "with zipfile.ZipFile(ZIP_PATH, 'r') as zipf:\n",
            "    bad_file = zipf.testzip()\n",
            "    if bad_file:\n",
            "        raise RuntimeError(f\"Corrupt file in ZIP: {bad_file}\")\n",
            "    file_count = len(zipf.namelist())\n",
            "\n",
            "zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)\n",
            "print(f\"\\nSUCCESS! ZIP package verified: {ZIP_PATH}\")\n",
            "print(f\"Total files in ZIP : {file_count}\")\n",
            "print(f\"Total ZIP size     : {zip_size_mb:.2f} MB\")\n"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "foal_c1_projection_controlled.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = Path("notebooks/foal_c1_projection_controlled.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Generated {out_file} successfully.")


if __name__ == "__main__":
    create_c1_notebook()
