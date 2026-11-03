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

## 2. Direct Source Verification from HCMUS Rubric

Directly verified from the official HCMUS syllabus text:
> "Choosing the scope: The team picks one of two scopes and states the choice and its reasoning in the deck:
> - **Full scope:** all main experiments reported in the paper, on the same or equivalent datasets.
> - **Core scope:** the two most important experiments at minimum, including the headline comparison of the paper, with the rest replaced by one additional analysis the team designs itself, such as a sensitivity sweep over the configuration the paper varies.
> The scope is a judgment call the team owns."
>
> "The rules of implementation:
> - Reimplement from the paper's description only. If the paper ships a public implementation, that code exists for verification and comparison... but never copy from it.
> - Python 3.9 or newer, with standard scientific stack.
> - Fix random seeds, document them, keep runs repeatable.
> - Every important function carries a docstring.
> - Every experiment lands in one public Weights & Biases project (W&B).
> - Datasets load programmatically from hosted sources (Hugging Face / torchvision / Kaggle)."

---

## 3. Formal Scope Classification Matrix

| Dimension / Item | Classification | Description & Justification | Status / Location in Repo |
| :--- | :--- | :--- | :--- |
| **Independent Re-Implementation** | `REQUIRED_BY_HCMUS` | Clean-room re-implementation from paper's mathematical description (`FrozenViT`, `SmoothProjection`, `OnlineRidgeRLS`, `FOALModel`) without copying code. | Complete (`src/foal/`) |
| **Official Code Execution & Deviation Analysis** | `REQUIRED_BY_HCMUS` | Running official implementation (`liuyuchen-cz/F-OAL`), comparing outputs against team's rebuild, localizing divergences. | Complete (`artifacts/official_cifar100/`, commit `3eca8232`) |
| **Public W&B Experiment Tracking** | `REQUIRED_BY_HCMUS` | Recording configuration, metrics, learning curves for every run in public W&B project. | Active (`CoreWeave Forge`: `hcmus-foal-reproduction`) |
| **Programmatic Dataset Loading** | `REQUIRED_BY_HCMUS` | End-to-end programmatic loading and preparation of CIFAR-100 without manual pre-processing steps. | Complete (`src/foal/datasets.py`, `notebooks/`) |
| **Experiment 1: Headline Comparison on CIFAR-100** | `REQUIRED_BY_HCMUS` (under Core Scope) | Mandated primary experiment: comparing F-OAL against key baseline families (Backprop FC, Linear Ridge, Literature benchmarks). | In Design (`docs/headline_comparison_plan.md`) |
| **Experiment 2: Table 4 Ablation Study (CIFAR-100)** | `CHOSEN_BY_TEAM` | Team's selection for the second mandatory experiment under Core Scope: isolating Feature Fusion (FF) and Smooth Projection (SP). | Completed & Verified (`results/table4_analysis/`, `TABLE4_SIGNED_OFF_WITH_LIMITATIONS`) |
| **Additional Analysis: Batch-Size Invariance (RLS)** | `CHOSEN_BY_TEAM` | Team's self-designed analysis replacing remaining paper datasets: empirical & numerical validation of Woodbury invariance across $B \in \{5, 10, 25, 50, 100\}$. | Planned (`docs/batch_size_invariance_plan.md`) |
| **Controlled Diagnostic Studies (C1, C2, C3)** | `CHOSEN_BY_TEAM` | Controlled factorial studies isolating projection initialization (C1), pretrained backbone weights (C2), and image preprocessing/interpolation (C3) explaining official vs paper discrepancies. | Complete & Merged (`main` at `e1fe4f3`) |
| **Table 4 Ablation on DTD Dataset** | `OPTIONAL_RESEARCH_EXTENSION` | Evaluating Table 4 ablation on Describable Textures Dataset (DTD) where ViT features require non-linear projection boost (+11.6 pp). | Roadmap (`docs/table4_ablation_plan.md`) |
| **Cross-Dataset Generalization (Tiny-ImageNet, Aircraft)** | `OPTIONAL_RESEARCH_EXTENSION` | Validating F-OAL across fine-grained (FGVCAircraft) and multi-class (Tiny-ImageNet) continual learning benchmarks. | Deferred to Extension Phase |
| **Multi-Seed Statistical Sweep (5 Seeds)** | `OPTIONAL_RESEARCH_EXTENSION` | Full 5-seed confidence interval estimation across all baselines. Single seed (42) with strict parity provides definitive alignment. | Planned for Final Report |
| **Open Source Release & Documentation** | `REQUIRED_BY_HCMUS` | Reproducible GitHub repository with setup guides, standalone scripts, environment specifications, and clear commit history. | Active (`https://github.com/caoban123/F-OAL`) |

---

## 4. Strict Boundary Audit

### 4.1 What is Strictly Mandated by HCMUS
1. Clean code architecture adhering to standard ML engineering practices (Python >=3.9, docstrings, type annotations).
2. Explicit statement of Core Scope justification in presentation deck and README.
3. Direct side-by-side comparison with published paper numbers and calculated relative deviation.
4. Deviation analysis explaining configuration omissions, environment divergence, and random initialization.
5. Official code comparison identifying divergences and bug fixes.
6. Public W&B project tracking all runs.

### 4.2 What is Intentionally Scoped as Extension
1. Full 6-dataset evaluation across all 14 baselines (justified under Core Scope due to GPU budget constraints).
2. Training complex multi-pass replay baselines (e.g. DER++, Co2L) from scratch across multiple epochs — scoped to standard 1-pass baselines and literature-reported values.
