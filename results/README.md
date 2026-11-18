# F-OAL Experiment Results Directory Structure & Colab Result Ingestion Guide

This document describes the organization of experiment results across CIFAR-100 and DTD datasets, where to paste results downloaded from Google Colab notebooks, and how to execute post-experiment analysis scripts.

---

## 1. Directory Overview

| Directory / File | Dataset | Role & Contents | Status |
| :--- | :--- | :--- | :--- |
| `results/dtd_c1_full/` | DTD | **Target folder** for C1 Projection Controlled Experiment runs | Ready for pasting Colab outputs |
| `results/dtd_c2_full/` | DTD | **Target folder** for C2 Backbone Controlled Experiment runs | Ready for pasting Colab outputs |
| `results/dtd_c3_full/` | DTD | **Target folder** for C3 Preprocessing Controlled Experiment runs | Ready for pasting Colab outputs |
| `results/foal_full_dtd_seed42/` | DTD | **Target folder** for single-seed Full DTD Reproduction run (Seed 42) | Kept as test fixture; overwrite with Colab run |
| `results/official_foal_dtd_seed0/` | DTD | **Target folder** for Official F-OAL Reproduction run on DTD (Seed 0) | Ready for pasting Colab outputs |
| `results/dtd_c1_analysis/` | DTD | Destination for C1 analysis plots, comparison tables, and JSON summaries | Cleaned, ready for analysis script |
| `results/dtd_c2_analysis/` | DTD | Destination for C2 analysis plots, comparison tables, and JSON summaries | Cleaned, ready for analysis script |
| `results/dtd_c3_analysis/` | DTD | Destination for C3 analysis plots and tables; holds diagnostics JSONs | Cleaned stale files; diagnostics preserved |
| `results/dtd_c1_diagnostics/` | DTD | Matched-input diagnostics for C1 projection variants | Valid (created today) |
| `results/dtd_c2_diagnostics/` | DTD | Feature representation diagnostics for C2 backbone variants | Valid (created today) |
| `results/dtd/` | DTD | Table 4 10-stage ablation runs (A0, A1, A2, A3 and multi-seed) | Tracked canonical ablation runs |
| `results/dtd_table4_analysis/` | DTD | Table 4 factorial decomposition plots, manifests, and CSV tables | Tracked official DTD Table 4 analysis |
| `results/foal_dtd_ablation_results/` | DTD | Mirror copy of Table 4 ablation configs and run metrics | Tracked |
| `results/foal_multiseed_dtd_seed123/` | DTD | Multi-seed run fixture (Seed 123) for Table 4 forensic verification | Tracked test fixture (`test_dtd_forensic_verification.py`) |
| `results/foal_multiseed_dtd_seed2026/` | DTD | Multi-seed run fixture (Seed 2026) for Table 4 forensic verification | Tracked test fixture (`test_dtd_forensic_verification.py`) |
| `results/c1_analysis/` | CIFAR-100 | CIFAR-100 C1 projection analysis artifacts and plots | Tracked |
| `results/c1_diagnostics/` | CIFAR-100 | CIFAR-100 C1 input diagnostics | Tracked |
| `results/c2_analysis/` | CIFAR-100 | CIFAR-100 C2 backbone analysis artifacts and plots | Tracked |
| `results/c3_analysis/` | CIFAR-100 | CIFAR-100 C3 preprocessing factorial analysis artifacts | Tracked |
| `results/table4_analysis/` | CIFAR-100 | CIFAR-100 Table 4 ablation analysis artifacts and plots | Tracked |
| `results/foal_full_cifar100_seed42/` | CIFAR-100 | Full reproduction run on CIFAR-100 (Seed 42) | Tracked |
| `results/official_foal_cifar100_seed0/` | CIFAR-100 | Official author benchmark run on CIFAR-100 (Seed 0) | Tracked |
| `results/summary_runs.csv` | Both | Automated consolidated run registry | Generated via `summarize_runs.py` |
| `results/summary_aggregated.csv` | Both | Consolidated group-level statistics | Generated via `summarize_runs.py` |

---

## 2. Where to Paste Google Colab Results

When you download experiment output folders or zip files from Colab, extract them into the following exact subdirectories:

### A. DTD C1 Projection Controlled Experiment
* **Notebook:** `notebooks/foal_c1_projection_controlled_dtd.ipynb`
* **Colab zip/folder destination:** `results/dtd_c1_full/`
* **Expected folder structure:**
  ```text
  results/dtd_c1_full/
  ├── p0_gaussian_std1/
  │   ├── metadata.json
  │   ├── metrics.json
  │   └── task_accuracy.csv
  ├── p1_official_uniform/
  │   ├── metadata.json
  │   ├── metrics.json
  │   └── task_accuracy.csv
  └── p2_gaussian_matched_variance/
      ├── metadata.json
      ├── metrics.json
      └── task_accuracy.csv
  ```

### B. DTD C2 Backbone Controlled Experiment
* **Notebook:** `notebooks/foal_c2_backbone_controlled_dtd.ipynb`
* **Colab zip/folder destination:** `results/dtd_c2_full/`
* **Expected folder structure:**
  ```text
  results/dtd_c2_full/
  ├── b0_augreg_in1k/
  │   ├── metadata.json
  │   ├── metrics.json
  │   └── task_accuracy.csv
  └── b1_augreg2_in21k_ft_in1k/
      ├── metadata.json
      ├── metrics.json
      └── task_accuracy.csv
  ```

### C. DTD C3 Preprocessing Factorial Experiment (2x2 Design)
* **Notebook:** `notebooks/foal_c3_preprocessing_controlled_dtd.ipynb`
* **Colab zip/folder destination:** `results/dtd_c3_full/`
* **Expected folder structure:**
  ```text
  results/dtd_c3_full/
  ├── c3_a_bicubic_normalized/
  │   ├── metadata.json
  │   ├── metrics.json
  │   └── task_accuracy.csv
  ├── c3_b_bilinear_normalized/
  │   ├── metadata.json
  │   ├── metrics.json
  │   └── task_accuracy.csv
  ├── c3_c_bicubic_raw/
  │   ├── metadata.json
  │   ├── metrics.json
  │   └── task_accuracy.csv
  └── c3_d_bilinear_raw/
      ├── metadata.json
      ├── metrics.json
      └── task_accuracy.csv
  ```

### D. DTD Full Reproduction (Seed 42)
* **Notebook:** `notebooks/foal_dtd_full_reproduction.ipynb`
* **Colab zip/folder destination:** `results/foal_full_dtd_seed42/`
* **Expected files:**
  ```text
  results/foal_full_dtd_seed42/
  ├── metadata.json
  ├── metrics.json
  ├── task_accuracy.csv
  └── task_10.pt (optional checkpoint)
  ```

### E. Official F-OAL Reproduction on DTD (Author Code `liuyuchen-cz/F-OAL @ 73e4ac1`)
* **Notebook:** `notebooks/foal_official_dtd_reproduction.ipynb`
* **Colab zip/folder destination:** `results/official_foal_dtd_seed0/`
* **Expected files:**
  ```text
  results/official_foal_dtd_seed0/
  ├── metadata.json
  ├── metrics.json
  ├── task_accuracy.csv
  ├── official_stdout.log
  └── dtd_FOAL.csv
  ```

---

## 3. Analysis Commands (Run After Pasting Colab Results)

Once the run folders are pasted into place, execute the following commands from the project root to generate all comparison tables, statistical summaries, and publication-ready figures:

### Step 1: Run C1 Analysis
```bash
python -m scripts.generate_c1_dtd_analysis \
  --results-dir results/dtd_c1_full \
  --output-dir results/dtd_c1_analysis \
  --diag-file results/dtd_c1_diagnostics/c1_dtd_matched_input_diagnostics.json
```
**Outputs generated in `results/dtd_c1_analysis/`:**
* `c1_full_comparison.csv` & `c1_full_comparison.json`
* `c1_stage_accuracies.csv` & `c1_final_task_accuracies.csv` & `c1_task_forgetting.csv`
* `c1_dtd_stage_accuracy_comparison.png`
* `chart2_average_accuracy_comparison.png` & `chart4_projection_effects.png`

### Step 2: Run C2 Analysis
```bash
python -m scripts.generate_c2_dtd_analysis \
  --results-dir results/dtd_c2_full \
  --output-dir results/dtd_c2_analysis \
  --c1-p1-dir results/dtd_c1_full/p1_official_uniform \
  --diag-file results/dtd_c2_diagnostics/c2_dtd_matched_backbone_diagnostics.json
```
**Outputs generated in `results/dtd_c2_analysis/`:**
* `c2_full_comparison.csv` & `c2_full_comparison.json`
* `c2_stage_accuracies.csv` & `c2_final_task_accuracies.csv` & `c2_task_forgetting.csv`
* `c2_dtd_stage_accuracy_comparison.png`
* `c2_paper_official_ours_comparison.png` & `c2_backbone_effects.png`

### Step 3: Run C3 Analysis
```bash
python -m scripts.generate_c3_dtd_analysis \
  --results-dir results/dtd_c3_full \
  --output-dir results/dtd_c3_analysis \
  --table4-a0-dir results/dtd/dtd_a0_full \
  --diag-file results/dtd_c3_analysis/c3_dtd_matched_diagnostics.json
```
**Outputs generated in `results/dtd_c3_analysis/`:**
* `c3_full_comparison.csv` & `c3_full_comparison.json`
* `c3_stage_accuracies.csv` & `c3_final_task_accuracies.csv` & `c3_task_forgetting.csv`
* `c3_factorial_comparison.csv` & `c3_factorial_summary.json`
* `c3_dtd_stage_accuracy_comparison.png`
* `c3_main_effects_and_interaction.png` & `c3_factorial_interaction_plot.png` & `c3_representation_diagnostics.png`

### Step 4: Refresh Master Run Registry
```bash
python -m scripts.summarize_runs --results-dir results
```
**Updates:**
* `results/summary_runs.csv`
* `results/summary_aggregated.csv`

### Step 5: (Optional) Sync Everything to Weights & Biases
```bash
python -m scripts.sync_dtd_to_wandb
```

---

## 4. Audit & Cleanup Notes

During the repository audit, the following legacy and outdated items were identified and processed:

1. **`results/foal_seed42_pilot/` (REMOVED):**
   * Contained an outdated 1-task pilot run on CPU with an 11.1 MB checkpoint (`task_01.pt`) from Nov 9.
   * Confirmed not referenced by any production scripts or tests. Removed to eliminate clutter and save disk space.
2. **Stale comparison files in `results/dtd_c1_analysis/` and `results/dtd_c2_analysis/` (REMOVED):**
   * Contained outdated Nov 8 CSVs and figures that predated the aligned C1/C2 pipeline.
   * Cleaned up so fresh outputs from `generate_c1_dtd_analysis` and `generate_c2_dtd_analysis` can be written cleanly.
3. **Stale comparison files in `results/dtd_c3_analysis/` (REMOVED):**
   * Contained outdated Nov 8 files (`c3_full_comparison.*`, `c3_stage_accuracy_comparison.png`).
   * Cleaned up while **strictly preserving** today's verified diagnostics:
     * `c3_dtd_matched_diagnostics.json`
     * `c3_dtd_transform_verification.json`
4. **Preserved Test Fixtures:**
   * `results/foal_full_dtd_seed42/`, `results/foal_multiseed_dtd_seed123/`, and `results/foal_multiseed_dtd_seed2026/` are actively checked by `tests/test_dtd_forensic_verification.py` for Table 4 mathematical reconciliation and multi-seed stability. They must be preserved. When pasting fresh Colab results for Seed 42, `results/foal_full_dtd_seed42/` will be cleanly updated with complete GPU results.
5. **Feature Cache & Temporary Summaries:**
   * `results/dtd_feature_cache/` (~59 MB) and `results/dtd_multiseed_1600_summary.json` are preserved locally for fast CPU feature extraction if needed, and are safely excluded from git commits via `.gitignore`.
