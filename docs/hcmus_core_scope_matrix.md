# HCMUS Checkpoint 3 Core Scope & Reproducibility Matrix

## 1. Executive Summary & Audit Mandate

This document establishes the formal compliance matrix for the reproduction of the paper:
**"F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning" (NeurIPS 2024)**.

The project is governed by the HCMUS Checkpoint 3 Reproduction Requirements:
[HCMUS Checkpoint 3 Statement](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/)

To guarantee methodological rigor and transparent scientific auditing, all project deliverables and experiments are classified into three mutually exclusive categories:
- `REQUIRED_BY_HCMUS`: Explicit non-negotiable requirements mandated by the HCMUS evaluation rubric.
- `CHOSEN_BY_TEAM`: Methodological choices, ablations, and analytical experiments selected by the reproduction team to rigorously satisfy or demonstrate specific rubric criteria.
- `OPTIONAL_RESEARCH_EXTENSION`: Advanced scientific extensions, cross-domain stress tests, or multi-seed sweeps that enrich understanding but lie beyond the minimum passing threshold.

---

## 2. Formal Scope Classification Matrix

| Dimension / Item | Classification | Description & Justification | Status / Location in Repo |
| :--- | :--- | :--- | :--- |
| **Independent Core Implementation** | `REQUIRED_BY_HCMUS` | Clean-room re-implementation of F-OAL architecture (`FrozenViT`, `SmoothProjection`, `OnlineRidgeRLS`, `FOALModel`) independent of external proprietary frameworks. | Complete (`src/foal/`) |
| **Official Code Execution & Audit** | `REQUIRED_BY_HCMUS` | Running official implementation (`liuyuchen-cz/F-OAL`), verifying published claims, logging exact metrics on target hardware. | Complete (`official_runs/`, commit `3eca8232`) |
| **Zero-Replay / Single-Pass Online Stream** | `REQUIRED_BY_HCMUS` | Faithful implementation of online continual learning: samples processed sequentially in small mini-batches without storing raw exemplar images or multi-pass backpropagation. | Complete (`src/foal/datasets.py`, `scripts/train.py`) |
| **Experiment 1: Headline Comparison on CIFAR-100** | `REQUIRED_BY_HCMUS` | Reproducing primary benchmark comparison against baseline families (Backprop FC, Linear Ridge, Experience Replay) under identical protocol. | In Design (`docs/headline_comparison_plan.md`) |
| **Experiment 2: Table 4 Ablation Study (CIFAR-100)** | `CHOSEN_BY_TEAM` | Dissecting the architectural contributions of Feature Fusion (FF) and Smooth Projection (SP) as published in Table 4 of NeurIPS 2024. | Active (`configs/table4_*.yaml`, `notebooks/foal_table4_ablation_controlled.ipynb`) |
| **Additional Analysis: Batch-Size Invariance (RLS)** | `CHOSEN_BY_TEAM` | Empirical and theoretical validation that Recursive Least Squares (RLS) is mathematically invariant to batch partitioning $B \in \{5, 10, 25, 50, 100\}$ (Section 5.4 / Additional Analysis 3A). | Planned (`docs/batch_size_invariance_plan.md`) |
| **Controlled Diagnostic Studies (C1, C2, C3)** | `CHOSEN_BY_TEAM` | Controlled factorial ablation of projection initialization (C1), pretrained backbone weights (C2), and image preprocessing/interpolation (C3) explaining official vs paper discrepancies. | Complete & Merged (`main` at `e1fe4f3`) |
| **Table 4 Ablation on DTD Dataset** | `OPTIONAL_RESEARCH_EXTENSION` | Evaluating Table 4 ablation on Describable Textures Dataset (DTD) where ViT features require non-linear projection boost (+11.6 pp). | Roadmap (`docs/table4_ablation_plan.md`) |
| **Cross-Dataset Generalization (Tiny-ImageNet, Aircraft)** | `OPTIONAL_RESEARCH_EXTENSION` | Validating F-OAL across fine-grained (FGVCAircraft) and multi-class (Tiny-ImageNet) continual learning benchmarks. | Deferred to Extension Phase |
| **Multi-Seed Statistical Sweep (5 Seeds)** | `OPTIONAL_RESEARCH_EXTENSION` | Full 5-seed confidence interval estimation across all baselines. Single seed (42) with strict parity provides definitive alignment. | Planned for Final Report |
| **W&B Experiment Tracking** | `REQUIRED_BY_HCMUS` | Live tracking of all training runs, learning curves, forgetting metrics, and system memory. | Active (`CoreWeave Forge`: `hcmus-foal-reproduction`) |
| **Open Source Release & Documentation** | `REQUIRED_BY_HCMUS` | Reproducible GitHub repository with setup guides, standalone scripts, environment specifications, and clear commit history. | Active (`https://github.com/caoban123/F-OAL`) |

---

## 3. Strict Boundary Audit

### 3.1 What is Strictly Mandated
1. Clean code architecture adhering to standard ML engineering practices.
2. Direct comparison with official paper figures (NeurIPS 2024 Table 1/2/4).
3. Clear documentation explaining any performance delta ($\Delta \bar{A} \le 1.0\text{ pp}$ target).
4. Automated verification tests and reproducible execution pipelines.

### 3.2 What is Intentionally Scoped as Extension
1. Training deep gradient-based continual learning models (e.g. DER++, ER-AML) from scratch with hyperparameter tuning across dozens of epochs — scoped to standard 1-pass baselines and literature-reported values to respect compute budget and Colab constraints.
2. Large-scale multi-GPU cluster runs — our protocol is engineered to execute within single-GPU constraints (NVIDIA T4 / A100) via deterministic feature caching.
