# F-OAL — DTD Full Reproduction Plan (NeurIPS 2024 Table 2 & Table 4)

> **Status:** `SPECIFICATION_AND_CODE_COMPLETE`  
> **Benchmark Dataset:** Describable Textures Dataset (DTD) via HuggingFace `tanganke/dtd` / `torchvision.datasets.DTD`  
> **Paper Reference:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024, Table 2 & Table 4)  
> **Reference Notebook:** [`notebooks/foal_cifar100_full_reproduction.ipynb`](../notebooks/foal_cifar100_full_reproduction.ipynb)  
> **Target Notebook:** [`notebooks/foal_dtd_full_reproduction.ipynb`](../notebooks/foal_dtd_full_reproduction.ipynb)  
> **Target Config:** [`configs/dtd_full_reproduction.yaml`](../configs/dtd_full_reproduction.yaml) / [`configs/dtd.yaml`](../configs/dtd.yaml)  

---

## 1. Scientific Problem & Benchmark Targets

### Official Paper Targets (NeurIPS 2024 Table 2 & Table 4)
- **Average Incremental Accuracy ($A_{\text{avg}}$):** **82.8%**
- **Last Task Accuracy ($A_{\text{last}}$):** ~76.1%
- **Average Forgetting ($F$):** ~8.6%
- **GPU Training Duration:** ~8 seconds (NeurIPS 2024 Table 3)
- **Memory Footprint:** $\mathcal{M} = 0$ (Strictly Zero-Replay, zero buffer memory overhead)

---

## 2. Experimental Protocol Specifications

| Protocol Dimension | Specification | Verification & Implementation Detail |
|:---|:---|:---|
| **Dataset** | Describable Textures Dataset (DTD) | HuggingFace `tanganke/dtd` with `torchvision.datasets.DTD` fallback |
| **Classes Selected** | 40 classes out of 47 | Deterministic pseudo-random selection via `seed=42` (`make_dtd_class_order`) |
| **Task Partition** | 10 tasks $\times$ 4 classes | Zero-overlap partition across incremental stages |
| **Data Alignment** | Exactly 40 train & 40 test / class | 160 train / 160 test per task; 1,600 train / 1,600 test total (`samples_per_class: 40`) |
| **Pretrained Backbone** | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | ImageNet-21k pretraining + ImageNet-1k fine-tuning, 100% frozen (`param.requires_grad = False`) |
| **Preprocessing** | Bilinear, Raw $[0, 1]$ | Official author transform: `Resize((224, 224))` + `ToTensor()`, verified bitwise $\Delta \le 10^{-7}$ |
| **Feature Fusion** | Multi-layer CLS averaging | Mean of CLS tokens across all 12 blocks + final LayerNorm ($d = 768$) |
| **Smooth Projection** | Uniform $D = 1000$, Sigmoid | Kaiming Uniform $\mathcal{U}(-\frac{1}{\sqrt{3d}}, +\frac{1}{\sqrt{3d}})$, seed 42 |
| **Analytic Learner** | Online Woodbury RLS | Exact Sherman-Morrison-Woodbury update, $\gamma = 1.0$, `float64` precision |
| **Stream Processing** | $B = 10$, One-Pass | Sequential streaming without replay buffer ($\mathcal{M} = 0$) |
| **Evaluation Seeds** | Phase P1.2: Seed 42<br>Phase P1.3: Seeds 42, 123, 2026 | Full metric aggregation with mean $\pm$ sample standard deviation |

---

## 3. End-to-End Execution Pipeline

Following the structure of [`notebooks/foal_cifar100_full_reproduction.ipynb`](../notebooks/foal_cifar100_full_reproduction.ipynb):

1. **Hardware Verification:** Checks GPU availability, driver status via `nvidia-smi`, PyTorch version, and VRAM.
2. **Environment & Dependency Setup:** Installs `timm`, `datasets`, `wandb`, `pyyaml`, `pytest`, `matplotlib`, `pandas`.
3. **Preflight Test Suite:** Runs regression unit tests verifying Woodbury RLS updates, DTD dataset partitioning, and protocol isolation.
4. **Single-Seed Full Reproduction:**
   ```bash
   python -m scripts.train \
       --config configs/dtd_full_reproduction.yaml \
       --seed 42 \
       --device cuda \
       --run-name foal_full_dtd_seed42 \
       --wandb \
       --wandb-offline
   ```
   Saves task checkpoints `task_01.pt` ... `task_10.pt`, `metrics.json`, and `metadata.json`.
5. **Independent Checkpoint Evaluation:**
   ```bash
   python -m scripts.evaluate \
       --checkpoint results/foal_full_dtd_seed42/task_10.pt \
       --device cuda
   ```
   Loads `task_10.pt` into a fresh model instance and verifies all 10 seen tasks against disk test data.
6. **Official Results Table & Summary:**
   ```bash
   python -m scripts.summarize_runs --results-dir results
   ```
   Validates run completeness, isolates protocols, and exports summary CSVs and JSONs.
7. **Multi-Seed Full Reproduction (Optional — Phase P1.3):**
   Orchestrates seeds 42, 123, and 2026 using `scripts/run_seeds.py` with skip-completed logic.
8. **Export & Archive:** Creates a compressed `.zip` of all results, CSVs, metrics, and metadata.

---

## 4. Deliverables Checklist

- [x] Configuration file: [`configs/dtd_full_reproduction.yaml`](../configs/dtd_full_reproduction.yaml)
- [x] Standard configuration file: [`configs/dtd.yaml`](../configs/dtd.yaml)
- [x] Colab / Jupyter notebook: [`notebooks/foal_dtd_full_reproduction.ipynb`](../notebooks/foal_dtd_full_reproduction.ipynb)
- [x] Notebook generator script: [`scripts/create_dtd_notebook.py`](../scripts/create_dtd_notebook.py)
- [x] Dataset-aware checkpoint evaluation: [`scripts/evaluate.py`](../scripts/evaluate.py)
- [x] Protocol-isolated run summarizer: [`scripts/summarize_runs.py`](../scripts/summarize_runs.py)
- [x] Unit & regression test suite: [`tests/test_dtd_full_reproduction.py`](../tests/test_dtd_full_reproduction.py)
