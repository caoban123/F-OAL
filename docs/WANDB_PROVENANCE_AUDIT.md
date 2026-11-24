# W&B Provenance Audit Report — CIFAR-100 Benchmark Runs


> **Điều hướng nhanh:** [← Trang chủ `README.md`](../README.md) | [Báo cáo Khắc phục Provenance](WANDB_PROVENANCE_REMEDIATION_REPORT.md) | [W&B Report CIFAR-100](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/reports/F-OAL-T-i-l-p-k-t-qu-v-ph-n-t-ch-th-c-nghi-m-tr-n-b-d-li-u-CIFAR-100--VmlldzoxODI2NDExMA)

**Project:** `hcmus-foal-reproduction`
**Entity:** `caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm`
**W&B Server:** `https://forge.coreweave.com`
**Audit Role:** Research Reproducibility Auditor
**Scope:** Kiểm toán độc lập nguồn gốc (Provenance) **chỉ tập trung vào 4 source runs CIFAR-100 (`seed=42`)** (`3v53tmq5` — F-OAL C3-D, `xlcme1c3` — Experience Replay, `c5pjewk4` — Official-protocol LwF, `ifl3rz2u` — Adapted LwF) cùng 1 run tổng hợp `analysis-only` (`nceap43w`). Toàn bộ dự án còn bao gồm các thí nghiệm C1–C3 khác và Table 4 ablation với 3 seed (`42`, `123`, `2026`), cũng như các bộ dữ liệu khác của bài báo. Không chỉnh sửa Summary gốc, không sửa history, không tạo run mới, không chạy lại training.

---

## A. Run Inventory

| Run ID | W&B Run Name | Group | Job Type | State | Created At (UTC) | History Rows | W&B Artifacts |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| `3v53tmq5` | `c3_d_bilinear_raw` | `c3_preprocessing_alignment` | `null` | `finished` | `2026-11-03T03:15:32Z` | 11 | `run-3v53tmq5-history:v0` |
| `xlcme1c3` | `headline_er_replay_buffer5000_seed42` | `headline_comparison` | `null` | `finished` | `2026-11-06T13:20:45Z` | 13 | `run-xlcme1c3-task_accuracy_matrix:v0`, `run-xlcme1c3-history:v0` |
| `c5pjewk4` | `headline_lwf_official_protocol_seed42` | `headline_comparison` | `null` | `finished` | `2026-11-06T13:21:09Z` | 13 | `run-c5pjewk4-task_accuracy_matrix:v0`, `run-c5pjewk4-history:v0` |
| `ifl3rz2u` | `headline_lwf_adapted_adamw_seed42` | `headline_comparison` | `null` | `finished` | `2026-11-06T13:21:21Z` | 13 | `run-ifl3rz2u-task_accuracy_matrix:v0`, `run-ifl3rz2u-history:v0` |
| `nceap43w` | `analysis_baseline_accuracy_progression_seed42` | `headline_comparison` | `analysis` | `finished` | `2026-11-06T14:15:40Z` | 10 | `run-nceap43w-accuracy_progression_table:v0`, `run-nceap43w-history:v0` |

### Đối chiếu nhất quán Matrix vs Summary Metrics
Kiểm toán độc lập từ tệp ma trận gốc so với W&B Summary của cả 4 source runs xác nhận **khớp 100%** (sai số `< 1e-12`):
- **`3v53tmq5` (F-OAL):** $\bar{A} = 90.401635\%$, $A_{10} = 85.73\%$, $F_{\text{past}} = 5.155556\%$ (`summary/*` khớp `comparison/*` và `task_accuracy.csv`).
- **`xlcme1c3` (ER):** $\bar{A} = 86.670270\%$, $A_{10} = 79.79\%$, $F_{\text{past}} = 16.744444\%$ (`final/*` khớp `comparison/*` và `task_matrix.csv`).
- **`c5pjewk4` (Official LwF):** $\bar{A} = 46.293492\%$, $A_{10} = 23.34\%$, $F_{\text{past}} = 1.252856 \times 10^{-16}\%$ (unclipped Eq. 18; clipped = $0.411111\%$, $F_{\text{code}} = 0.370000\%$) — khớp hoàn toàn giữa W&B Summary và `task_matrix.csv`.
- **`ifl3rz2u` (Adapted LwF):** $\bar{A} = 29.629496\%$, $A_{10} = 11.21\%$, $F_{\text{past}} = 94.988889\%$ (`final/*` khớp `comparison/*` và `task_matrix.csv`).
- **`nceap43w` (Analysis-only):** 10 dòng history khớp 100% đường cong `stage_accuracy` theo từng `task` ($1 \dots 10$) của 4 runs trên.

---

## B. Provenance Classification & Verification Matrix

### Phân loại nguồn gốc & Phân biệt loại bằng chứng

Để tránh nhầm lẫn giữa quá trình tracking trực tiếp và dữ liệu bổ sung hậu kiểm, kiểm toán phân biệt 4 loại bằng chứng sau:

1. **Original training evidence:** Bằng chứng được thu thập trực tiếp trong quá trình training (`3v53tmq5` — `ONLINE_TRAINING`, chạy `scripts/train.py` trên Colab `NVIDIA A100-SXM4-40GB` ở chế độ `--wandb-offline` rồi sync nguyên bản lên W&B từ `offline-run-20261103_025546-3v53tmq5`, có `output.log` và `wandb-metadata.json` gốc).
2. **Post-hoc imported results:** Kết quả Colab được import vào W&B sau khi training đã kết thúc (`xlcme1c3`, `c5pjewk4`, `ifl3rz2u` — `POST_HOC_IMPORT`). Ba baseline này được huấn luyện trên Colab (`Tesla T4`) bằng `scripts/train_baseline.py`, lưu thành ZIP, rồi import lên W&B qua `scripts/import_headline_to_wandb.py` (tag `posthoc_import`). **Ba baseline không được W&B tracking trực tiếp trong quá trình training.**
3. **Post-hoc attached artifacts:** Các `wandb.Artifact` được gắn bổ sung vào những run đã `finished` sau khi kết thúc training (`c3_d_bilinear_raw_results:v0`, `headline_er_provenance_bundle:v0`, `headline_official_lwf_provenance_bundle:v0`, `headline_adapted_lwf_provenance_bundle:v0`, đều mang `uploaded_post_hoc=True`). Việc gắn artifact hậu kiểm không biến các baseline `POST_HOC_IMPORT` thành run được tracking trực tiếp, và cũng không làm thay đổi phân loại `ONLINE_TRAINING` của `3v53tmq5`.
4. **Analysis-only runs:** Các run chỉ phục vụ tổng hợp và trực quan hóa hậu thực nghiệm (`nceap43w` — `ANALYSIS_ONLY`, tạo bởi `scripts/create_wandb_accuracy_progression_run.py`, `job_type="analysis"`, `is_training_run=False`).

### Bảng mức độ xác minh từng trường (`VERIFIED` / `PARTIAL` / `MISSING`)

| Trường kiểm toán | `3v53tmq5` (F-OAL) | `xlcme1c3` (ER) | `c5pjewk4` (Official LwF) | `ifl3rz2u` (Adapted LwF) | `nceap43w` (Analysis) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Run ID & Name** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `VERIFIED` | `VERIFIED` |
| **2. Training Implementation** | `VERIFIED` (`src/foal/model.py`) | `VERIFIED` (`src/foal/baselines/er.py`) | `VERIFIED` (`src/foal/baselines/official_lwf.py`) | `VERIFIED` (`src/foal/baselines/lwf.py`) | `VERIFIED` (`N/A - Analysis`) |
| **3. Training Script** | `VERIFIED` (`scripts/train.py`) | `PARTIAL` (local `train_baseline.py`, thiếu trong W&B config) | `PARTIAL` (local `train_baseline.py`, thiếu trong W&B config) | `PARTIAL` (local `train_baseline.py`, thiếu trong W&B config) | `VERIFIED` (`scripts/create_wandb_accuracy_progression_run.py`) |
| **4. Training Command** | `VERIFIED` (`wandb-metadata.json` `args`) | `MISSING` (không lưu CLI command) | `MISSING` (không lưu CLI command) | `MISSING` (không lưu CLI command) | `PARTIAL` (trong `config.yaml` W&B) |
| **5. Git Commit SHA thực thi** | `VERIFIED` (`15b8afc71f6109efdbf3ff98a801ffcc83aa58d5`, `dirty: true`) | `VERIFIED` trong `summary.json` (`b0658c8837563d845253880da8efb1f0f11387fc`), nhưng `PARTIAL` trên W&B (thiếu key `git_commit` trong `run.config`) | `VERIFIED` trong `summary.json` (`c3dd3f23fbf2d8d94d481ec75daceb860190118e`), `PARTIAL` trên W&B | `VERIFIED` trong `summary.json` (`b0658c8837563d845253880da8efb1f0f11387fc`), `PARTIAL` trên W&B | `VERIFIED` (`940bc2b`) |
| **6. Dataset & Version** | `VERIFIED` (`uoft-cs/cifar100`, `datasets 4.8.5`) | `VERIFIED` (`cifar100`, `datasets 4.8.5`) | `VERIFIED` (`cifar100`, `datasets 4.8.5`) | `VERIFIED` (`cifar100`, `datasets 4.8.5`) | `VERIFIED` (derived from 4 runs) |
| **7. Class Order & Seed** | `VERIFIED` (`seed=42` + danh sách 100 lớp đầy đủ) | `PARTIAL` (`seed=42, class_order_seed=42`, không lưu list 100 lớp) | `PARTIAL` (`seed=42, class_order_seed=42`, không lưu list 100 lớp) | `PARTIAL` (`seed=42, class_order_seed=42`, không lưu list 100 lớp) | `VERIFIED` (`seed=42`) |
| **8. Pretrained Backbone** | `VERIFIED` (`vit_base_patch16_224.augreg2_in21k_ft_in1k` + `pretrained_cfg`) | `PARTIAL` (trong `summary.json` & YAML, thiếu trong W&B `run.config`) | `PARTIAL` (trong `summary.json` & YAML, thiếu trong W&B `run.config`) | `PARTIAL` (trong `summary.json` & YAML, thiếu trong W&B `run.config`) | `N/A` |
| **9. Image Preprocessing** | `VERIFIED` (`bilinear`, `normalize=false`, `224`) | `PARTIAL` (trong `summary.json` & YAML, thiếu trong W&B `run.config`) | `PARTIAL` (trong `summary.json` & YAML, thiếu trong W&B `run.config`) | `PARTIAL` (trong `summary.json` & YAML, thiếu trong W&B `run.config`) | `N/A` |
| **10. Hyperparameters** | `VERIFIED` | `PARTIAL` (thiếu `weight_decay`, `retrieve/update_policy` trong W&B `run.config`) | `PARTIAL` (thiếu `temperature`, `momentum`, `dynamic_loss_weighting` trong W&B `run.config`) | `PARTIAL` (thiếu `temperature`, `distill_lambda`, `weight_decay` trong W&B `run.config`) | `VERIFIED` |
| **11. Original Training Logs** | `VERIFIED` (`output.log` local & trên W&B) | `MISSING` (archive `headline_full_gpu_verified` không chứa stdout log) | `MISSING` (archive `headline_official_lwf_verified` không chứa stdout log) | `MISSING` (archive `headline_full_gpu_verified` không chứa stdout log) | `N/A` |
| **12. Task Accuracy Matrix** | `VERIFIED` (`task_accuracy.csv`, 55 ô) | `VERIFIED` (`task_matrix.csv` + W&B Table) | `VERIFIED` (`task_matrix.csv` + W&B Table) | `VERIFIED` (`task_matrix.csv` + W&B Table) | `VERIFIED` (`accuracy_progression_table`) |
| **13. Final Eval Summary** | `VERIFIED` | `VERIFIED` | `VERIFIED` | `VERIFIED` | `VERIFIED` |
| **14. Artifacts trên W&B** | `VERIFIED` (`c3_d_bilinear_raw_results:v0` gắn post-hoc, `type=evaluation`) | `VERIFIED` (`headline_er_provenance_bundle:v0` gắn post-hoc + `run_table`, 3 PNGs) | `VERIFIED` (`headline_official_lwf_provenance_bundle:v0` gắn post-hoc + `run_table`, 3 PNGs) | `VERIFIED` (`headline_adapted_lwf_provenance_bundle:v0` gắn post-hoc + `run_table`, 3 PNGs) | `VERIFIED` (`run_table`, `history`) |

---

## C. Verified Source Files & SHA256 Checksums

| Run ID | Local File Path | Size (Bytes) | SHA256 |
| :---: | :--- | :---: | :--- |
| `3v53tmq5` | `results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/metadata.json` | 10,058 | `0a7f098570893fc8f2954b4f3e5630d4e57bb55a39f9c5ac55dbafee51673282` |
| `3v53tmq5` | `results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/metrics.json` | 2,316 | `55a22febcf0cf18732fe87bf49b83b2fb3e4fce947833adea238501247557f52` |
| `3v53tmq5` | `results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/task_accuracy.csv` | 628 | `68684dfb3d348c5838815a63fa45dd874bd84b3a427cd29d06f4c320fa5de1d7` |
| `3v53tmq5` | `results/c3_full_gpu_verified/configs/c3_d_bilinear_raw.yaml` | 1,549 | `8fd465782930add0a46fcbfeb8f614ef09b5b8d5611a44a56c9bcd3213e73c31` |
| `3v53tmq5` | `results/c3_full_gpu_verified/wandb/offline-run-20261103_025546-3v53tmq5/files/output.log` | 3,100 | `bc7038e7df055412565ef448d42c39329ec19e058339287ad0e582413e9c1111` |
| `3v53tmq5` | `results/c3_full_gpu_verified/wandb/offline-run-20261103_025546-3v53tmq5/files/wandb-metadata.json` | 1,399 | `23a128fa594d538e14ddfde615f9ded9a2443a31fd3fd107b0d9c043d31ae3db` |
| `xlcme1c3` | `results/headline_full_gpu_verified/runs/headline/er/summary.json` | 4,630 | `ff90b02c2b46353ac4d558e8aaa66854ca0e99a981230d9638ec3991d53f7d2c` |
| `xlcme1c3` | `results/headline_full_gpu_verified/runs/headline/er/task_matrix.csv` | 505 | `319b9394d26020422e48611314317dc4a7b4be54b43a3ac09fb70446235f9900` |
| `xlcme1c3` | `configs/headline/er_cifar100.yaml` | 929 | `c71dae905933083f6183d0abb938ba4e0fa5f12d448e4ed691190a87c696289e` |
| `c5pjewk4` | `results/headline_official_lwf_verified/runs/headline/lwf_official_cifar100/summary.json` | 4,917 | `9de1f84e0a7a37042e9b334ce9af3ac514222e86f1ee1dd70237311ec497ae70` |
| `c5pjewk4` | `results/headline_official_lwf_verified/runs/headline/lwf_official_cifar100/task_matrix.csv` | 505 | `1e5c7ca267a78f86cff486957f775184551553d8cda96efdf9db9ff43f4f0998` |
| `c5pjewk4` | `configs/headline/lwf_cifar100_official_protocol.yaml` | 1,150 | `4dd4a126de89a93e596dbddc1a83b586c82a3484027aa82e5593572b830cb698` |
| `ifl3rz2u` | `results/headline_full_gpu_verified/runs/headline/lwf/summary.json` | 4,230 | `4413d62b935903957c7f2843cb3342b2882dec6ce50b52dcaa4000e37d62b9a8` |
| `ifl3rz2u` | `results/headline_full_gpu_verified/runs/headline/lwf/task_matrix.csv` | 505 | `5a10c765158df0bb2c7953091ef6855019cc8c707782e63b672776c77dc42f6d` |
| `ifl3rz2u` | `configs/headline/lwf_cifar100.yaml` | 826 | `6e1eb73336315ede165f0ffe5beca49ec430927ef87c0b4a519d6dee8806ecfb` |

---

## D. Missing Evidence & Discrepancies Found

1. **Sai lệch `execution_commit` trong `scripts/import_headline_to_wandb.py`:**
   - Script import ghi `execution_commit: "d333705"` cho ER (`xlcme1c3`) và Adapted LwF (`ifl3rz2u`), nhưng `summary.json` gốc trong archive ghi nhận commit thực thi thực tế là **`b0658c8837563d845253880da8efb1f0f11387fc`** (nhánh `experiment/headline-comparison`, `dirty: true`).
2. **Thiếu raw Colab training stdout logs & CLI commands cho 3 baselines (`xlcme1c3`, `c5pjewk4`, `ifl3rz2u`):**
   - Các archive kết quả (`foal_cifar100_headline_results.zip`, `foal_cifar100_official_lwf_results.zip`) chỉ xuất `summary.json`, `task_matrix.csv`, `cifar100_headline_progression.png`, các file `configs/headline/*.yaml` và `SHA256SUMS.txt` (với Official LwF), **không** lưu `original_stdout_log` hay `original_cli_command` của quá trình chạy Colab (`original_stdout_log="MISSING"`, `original_cli_command="MISSING"`).
   - **Lưu ý quan trọng:** File `output.log` (`220 B`) hiện có trên W&B của 3 baseline runs được tạo tự động bởi W&B SDK tại thời điểm chạy `scripts/import_headline_to_wandb.py` (post-hoc import), **không đồng nghĩa với stdout gốc của quá trình training**.
   - Ngoài ra, mặc dù các file YAML cấu hình tồn tại trong ZIP Colab và repository, cấu hình thực thi tại runtime chưa được xác minh trực tiếp (`config_execution_verified=False`, mang nghĩa *chưa xác minh*, không phải đã chứng minh cấu hình sai).
3. **Dirty worktree trên cả 4 source runs:**
   - Cả 4 runs đều có `source_worktree_dirty=True` và `uncommitted_diff_status="MISSING"` (không có bản diff của các thay đổi chưa commit). Do đó `source_code_base_commit` không chứng minh toàn bộ mã nguồn thực thi khớp 100% với commit đó.
4. **Trạng thái khắc phục hậu kiểm (Remediation Completed):**
   - Đã hoàn tất bổ sung `run.config['provenance_remediation']` và gắn post-hoc `wandb.Artifact` (`v0`) cho cả 4 runs (`3v53tmq5`, `xlcme1c3`, `c5pjewk4`, `ifl3rz2u`) mà không làm thay đổi `Summary`, `History` hay legacy `provenance`. Chi tiết xem tại [`docs/WANDB_PROVENANCE_REMEDIATION_REPORT.md`](WANDB_PROVENANCE_REMEDIATION_REPORT.md).

---

## E. End-to-End Provenance Mapping

| Experiment | Training Script | Execution Git Commit | Local Verified Artifact | W&B Run ID & Mode |
| :--- | :--- | :--- | :--- | :--- |
| **F-OAL (C3-D)** | `scripts/train.py` | `15b8afc71f6109efdbf3ff98a801ffcc83aa58d5` | `results/c3_full_gpu_verified/.../c3_d_bilinear_raw/` | `3v53tmq5` (`ONLINE_TRAINING`) |
| **ER (Buffer=5000)** | `scripts/train_baseline.py` | `b0658c8837563d845253880da8efb1f0f11387fc` | `results/headline_full_gpu_verified/runs/headline/er/` | `xlcme1c3` (`POST_HOC_IMPORT`) |
| **Official LwF** | `scripts/train_baseline.py` | `c3dd3f23fbf2d8d94d481ec75daceb860190118e` | `results/headline_official_lwf_verified/runs/headline/lwf_official_cifar100/` | `c5pjewk4` (`POST_HOC_IMPORT`) |
| **Adapted LwF** | `scripts/train_baseline.py` | `b0658c8837563d845253880da8efb1f0f11387fc` | `results/headline_full_gpu_verified/runs/headline/lwf/` | `ifl3rz2u` (`POST_HOC_IMPORT`) |
| **10-Task Progression** | `scripts/create_wandb_accuracy_progression_run.py` | `940bc2b` | Derived from above 4 W&B runs | `nceap43w` (`ANALYSIS_ONLY`) |

---

## F. Recommendations (Đáp ứng Lab Book mà không chạy lại training)

1. **Bổ sung metadata còn thiếu vào `run.config` (Read-only đối với Summary/History):**
   - Dùng W&B API (`run.config.update(...)` và `run.update()`) để ghi bổ sung các trường đã xác minh từ `summary.json` gốc (`git_commit`, `backbone`, `preprocessing`, `training_script`, `hyperparameters`) cho `xlcme1c3`, `c5pjewk4`, `ifl3rz2u`, đồng thời đính chính `execution_commit` của `xlcme1c3` & `ifl3rz2u` từ `d333705` sang `b0658c8837563d845253880da8efb1f0f11387fc`.
2. **Đính kèm file cấu hình YAML gốc lên W&B (`run.save`):**
   - Có thể bổ sung `configs/headline/er_cifar100.yaml`, `lwf_cifar100_official_protocol.yaml`, `lwf_cifar100.yaml` lên mục Files của 3 runs baseline, và `results/c3_full_gpu_verified/.../{metadata.json,metrics.json,task_accuracy.csv}` lên `3v53tmq5` nếu muốn 100% file truy xuất trực tiếp từ W&B UI.
3. **Ghi nhận trung thực về stdout log của 3 baselines:**
   - Trong Lab Book / Phụ lục báo cáo, ghi rõ `output.log` gốc của 3 baselines không được đóng gói trong ZIP Colab, nhưng toàn bộ 10-stage progression, thời gian chạy và môi trường đã được lưu trọn vẹn trong `summary.json` kèm SHA256.

---

## G. Suggested W&B `run.config` Updates (Chưa thực thi)

```python
# Mẫu cập nhật an toàn (chỉ bổ sung run.config, không chạm summary/history):
# 1. xlcme1c3 (ER)
{
    "git_commit": "b0658c8837563d845253880da8efb1f0f11387fc",
    "training_script": "scripts/train_baseline.py",
    "config_file": "configs/headline/er_cifar100.yaml",
    "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "preprocessing": {"interpolation": "bilinear", "normalize": False, "size": 224},
    "weight_decay": 0.0001,
    "retrieve_policy": "random",
    "update_policy": "random",
}
# 2. c5pjewk4 (Official LwF)
{
    "git_commit": "c3dd3f23fbf2d8d94d481ec75daceb860190118e",
    "training_script": "scripts/train_baseline.py",
    "config_file": "configs/headline/lwf_cifar100_official_protocol.yaml",
    "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "preprocessing": {"interpolation": "bilinear", "normalize": False, "size": 224},
    "temperature": 2.0,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "dynamic_loss_weighting": True,
    "full_output_head": 100,
}
# 3. ifl3rz2u (Adapted LwF)
{
    "git_commit": "b0658c8837563d845253880da8efb1f0f11387fc",
    "training_script": "scripts/train_baseline.py",
    "config_file": "configs/headline/lwf_cifar100.yaml",
    "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "preprocessing": {"interpolation": "bilinear", "normalize": False, "size": 224},
    "temperature": 2.0,
    "distill_lambda": 1.0,
    "weight_decay": 0.0001,
}
```
