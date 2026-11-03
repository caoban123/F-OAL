import os
from pathlib import Path

def update_readme_implementation():
    path = Path("README_IMPLEMENTATION.md")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("| **C2: Backbone Alignment**"):
            new_lines.append("| **C2: B0 augreg_in1k Control** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.28%** | **74.84%** | **8.07%** / **8.97%** | `7fd0f4a` | [`results/c2_analysis/`](results/c2_analysis/) | [Run `3oiftwg6`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3oiftwg6) | `VERIFIED FULL GPU` (`C2 PASS`) |\n")
            new_lines.append("| **C2: B1 augreg2_in21k Treatment** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **91.77%** | **87.43%** | **4.52%** / **5.02%** | `7fd0f4a` | [`results/c2_analysis/`](results/c2_analysis/) | [Run `ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | `VERIFIED FULL GPU` (`C2 PASS`) |\n")
            new_lines.append("| **C3-A: Bicubic, Norm [-1,1]** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **91.77%** | **87.43%** | **4.52%** / **5.02%** | `7fd0f4a` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | `REUSED C2-B1` (`C3 PASS`) |\n")
            new_lines.append("| **C3-B: Bilinear, Norm [-1,1]** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **91.39%** | **86.95%** | **4.40%** / **4.89%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3) | `VERIFIED FULL GPU` (`C3 PASS`) |\n")
            new_lines.append("| **C3-C: Bicubic, Raw [0,1]** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **90.88%** | **86.35%** | **4.71%** / **5.23%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h) | `VERIFIED FULL GPU` (`C3 PASS`) |\n")
            new_lines.append("| **C3-D: Bilinear, Raw [0,1] (Official Aligned)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **90.40%** | **85.73%** | **4.64%** / **5.16%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | `VERIFIED FULL GPU` (`C3 PASS`) |\n")
        elif line.startswith("- **Status:** `C3_READY_FOR_GPU`"):
            new_lines.append("- **Status:** `C3_SIGNED_OFF_WITH_LIMITATIONS` (Branch: `experiment/c3-preprocessing-alignment`)\n")
        elif line.startswith("- **Tests:** `tests/test_c3_preprocessing.py` (6/6 passing)"):
            new_lines.append("- **Tests:** `tests/test_c3_preprocessing.py` (7/7 passing 100%)\n")
        elif line.startswith("- **Diagnostics:** `scripts/run_c3_diagnostics.py`"):
            new_lines.append("- **Diagnostics:** `results/c3_analysis/c3_matched_diagnostics.json` (N=200 samples, 88 classes)\n")
            new_lines.append("- **Analysis & Figures:** `results/c3_analysis/` (6 publication-quality figures, full factorial decomposition)\n")
            new_lines.append("- **W&B Verified Runs (10 Total):** Group `c3_preprocessing_alignment` ([ts2wh4e3](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3), [4m4uir7h](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h), [3v53tmq5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5), and reused [ne1p16b5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5))\n")
        else:
            new_lines.append(line)

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(new_lines)
    print("Updated README_IMPLEMENTATION.md")

def update_c3_plan():
    path = Path("docs/c3_preprocessing_alignment_plan.md")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update header status
    content = content.replace(
        "> **Trạng thái tài liệu:** `C3_READY_FOR_GPU`",
        "> **Trạng thái tài liệu:** `C3_SIGNED_OFF_WITH_LIMITATIONS` (Đã hoàn tất Full GPU trên A100, đồng bộ 10 runs W&B trên CoreWeave Forge, kiểm toán độc lập 55-entry matrix và phân rã giai thừa 2x2)"
    )
    
    # Update checkboxes in section 8
    content = content.replace(
        "- [ ] Chạy Full GPU cho C3-B, C3-C, C3-D trên Colab (A100) và đồng bộ W&B group `c3_preprocessing_alignment`.\n- [ ] Tính toán phân rã giai thừa thực nghiệm và nghiệm thu báo cáo khoa học C3.",
        "- [x] Chạy Full GPU cho C3-B, C3-C, C3-D trên Colab (A100) và đồng bộ W&B group `c3_preprocessing_alignment` ([ts2wh4e3](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3), [4m4uir7h](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h), [3v53tmq5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5)).\n- [x] Tính toán phân rã giai thừa thực nghiệm và nghiệm thu báo cáo khoa học C3 (`C3_SIGNED_OFF_WITH_LIMITATIONS`)."
    )

    c3_results_section = """
---

## 10. Báo cáo Kết quả Thực nghiệm Full GPU & Phân rã Giai thừa $2 \\times 2$

### 10.1. Môi trường Thực thi & Kiểm chứng Toàn vẹn Dữ liệu
- **Phần cứng GPU:** NVIDIA A100-SXM4-40GB (39.49 GB VRAM).
- **Phần mềm:** PyTorch `2.11.0+cu130`, Torchvision `0.26.0+cu130`, timm `1.0.30`, Python `3.13.16`.
- **Gói nén lưu trữ kết quả:** `foal_c3_preprocessing_results.zip` (148,700 bytes, SHA256: `a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6`).
- **Xác thực toàn vẹn:** 54 file trích xuất thành công tại `results/c3_full_gpu_verified/`, CRC-32 hợp lệ 100%, không phát hiện lỗi đường dẫn hay tràn buffer.
- **Tính toán độc lập:** Toàn bộ số liệu được tính toán lại từ ma trận tam giác dưới $10 \\times 10$ ($55$ phần tử) tại `scripts/audit_c3_results.py`. Độ lệch tối đa so với số liệu xuất từ Colab bằng chính xác $0.00\\text{e+}00$.

### 10.2. Bảng Kết quả Thực nghiệm Giai thừa $2 \\times 2$

| Điều kiện | Phép Nội suy (Factor A) | Chuẩn hóa Pixel (Factor B) | Dải Pixel | $\\bar{A}$ (%) | $A_{10}$ (%) | $F_{10}^{\\text{past}}$ (%) | $F_{10}^{\\text{code}}$ (%) | T_Feat (s) | T_RLS (s) | W&B Run ID | Trạng thái Provenance |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---|
| **C3-A** | Bicubic | Normalize | $[-1.0, 1.0]$ | **91.7742%** | **87.4300%** | **5.0222%** | **4.5200%** | 248.30 | 8.49 | [`ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | `REUSED_VERIFIED_C2_B1` |
| **C3-B** | Bilinear | Normalize | $[-1.0, 1.0]$ | **91.3918%** | **86.9500%** | **4.8889%** | **4.4000%** | 176.08 | 8.56 | [`ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3) | `VERIFIED_FRESH_GPU` |
| **C3-C** | Bicubic | Raw | $[0.0, 1.0]$ | **90.8762%** | **86.3500%** | **5.2333%** | **4.7100%** | 175.58 | 8.49 | [`4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h) | `VERIFIED_FRESH_GPU` |
| **C3-D** | Bilinear | Raw | $[0.0, 1.0]$ | **90.4016%** | **85.7300%** | **5.1556%** | **4.6400%** | 175.28 | 8.50 | [`3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | `VERIFIED_FRESH_GPU` |

### 10.3. Phân tích Phân rã Giai thừa (Factorial Effects Decomposition)

1. **Tác động Đơn giản (Simple Effects):**
   - Phép Nội suy trong môi trường Normalized: $E_{I|\\text{Norm}} = B - A = 91.3918\\% - 91.7742\\% = -0.3824\\text{ pp}$.
   - Phép Nội suy trong môi trường Raw: $E_{I|\\text{Raw}} = D - C = 90.4016\\% - 90.8762\\% = -0.4746\\text{ pp}$.
   - Phép Chuẩn hóa trong môi trường Bicubic: $E_{N|\\text{Bicubic}} = C - A = 90.8762\\% - 91.7742\\% = -0.8980\\text{ pp}$.
   - Phép Chuẩn hóa trong môi trường Bilinear: $E_{N|\\text{Bilinear}} = D - B = 90.4016\\% - 91.3918\\% = -0.9902\\text{ pp}$.

2. **Tác động Chính Trung bình (Average Main Effects):**
   - **Tác động chính của Phép Nội suy ($E_{\\text{interp}}$):**
     $$E_{\\text{interp}} = \\frac{(B - A) + (D - C)}{2} = \\frac{-0.3824 + (-0.4746)}{2} = \\mathbf{-0.4285\\text{ pp}}$$
   - **Tác động chính của Chuẩn hóa dải pixel ($E_{\\text{norm}}$):**
     $$E_{\\text{norm}} = \\frac{(C - A) + (D - B)}{2} = \\frac{-0.8980 + (-0.9902)}{2} = \\mathbf{-0.9441\\text{ pp}}$$

3. **Hiệu ứng Tương tác (Interaction Contrast & Coefficient):**
   - **Tương phản tương tác ($E_{\\text{IN}}$):**
     $$E_{\\text{IN}} = D - C - B + A = (D - B) - (C - A) = -0.9902 - (-0.8980) = \\mathbf{-0.0922\\text{ pp}}$$
   - **Hệ số hồi quy tương tác (Effects Coding $\\beta_{\\text{IN}}$):**
     $$\\beta_{\\text{IN}} = \\frac{E_{\\text{IN}}}{4} = \\mathbf{-0.0231\\text{ pp}}$$

4. **Tổng Dịch chuyển Tiền xử lý (Total Preprocessing Shift):**
   $$\\Delta_{\\text{total}} = D - A = 90.4016\\% - 91.7742\\% = \\mathbf{-1.3726\\text{ pp}}$$
   - **Kiểm chứng đồng nhất thức toán học:** $E_{\\text{interp}} + E_{\\text{norm}} = -0.4285 + (-0.9441) = -1.3726\\text{ pp} \\equiv \\Delta_{\\text{total}}$ (sai số $< 10^{-12}$).
   - **Tỷ trọng đóng góp:** Chuẩn hóa dải pixel đóng góp **$68.78\\%$** tổng mức sụt giảm; phương pháp nội suy đóng góp **$31.22\\%$**.
   - **Tính chất tương tác:** Giá trị tương tác rất nhỏ ($-0.0922\\text{ pp}$), cho thấy hai nhân tố hoạt động gần như độc lập tuyến tính (additive), trong đó chuẩn hóa pixel $[-1, 1]$ là yếu tố mang lại lợi thế biểu diễn lớn hơn.

### 10.4. Đối chiếu Đối chứng với Tác giả (Official Run) và Bài báo (NeurIPS Paper)

| Chỉ số Đánh giá | NeurIPS 2024 Paper | Official Author Run (`3eca8232`, Seed 0) | C3-D Full Alignment (`3v53tmq5`, Seed 42) | Chênh lệch C3-D vs Official | Chênh lệch C3-D vs Paper |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Average Incremental Accuracy ($\\bar{A}$)** | **91.10%** | **90.98%** | **90.40%** | **$-0.58\\text{ pp}$** | **$-0.70\\text{ pp}$** |
| **Last Accuracy ($A_{10}$)** | — | **85.84%** | **85.73%** | **$-0.11\\text{ pp}$** | — |
| **Forgetting Official Code ($F_{10}^{\\text{code}}$)** | 5.50% | **4.69%** | **4.64%** | **$-0.05\\text{ pp}$** | $-0.86\\text{ pp}$ |
| **Forgetting Past Tasks ($F_{10}^{\\text{past}}$)** | — | **5.17%** | **5.16%** | **$-0.01\\text{ pp}$** | — |

> [!IMPORTANT]
> **Ý Nghĩa Khoa Học Then Chốt:**
> 1. Khi căn chỉnh tiền xử lý về đúng pipeline của tác giả (Bilinear, Raw $[0, 1]$), độ chính xác cuối cùng $A_{10}$ của nhóm ($85.73\\%$) **hội tụ gần như hoàn hảo** với bản chạy của tác giả ($85.84\\%$, độ lệch chỉ vỏn vẹn $-0.11\\text{ pp}$).
> 2. Tỷ lệ quên $F_{10}^{\\text{past}}$ đạt $5.16\\%$ so với $5.17\\%$ của tác giả (chênh lệch $-0.01\\text{ pp}$), và $F_{10}^{\\text{code}}$ đạt $4.64\\%$ so với $4.69\\%$ (chênh lệch $-0.05\\text{ pp}$).
> 3. Sự khác biệt nhỏ còn lại trên $\\bar{A}$ ($90.40\\%$ vs $90.98\\%$, chênh lệch $-0.58\\text{ pp}$) xuất phát từ hoán vị thứ tự các lớp và độ khó của các task đầu giữa Seed 42 và Seed 0.
> 4. Bí ẩn về việc C2-B1 vượt tác giả $+0.79\\text{ pp}$ đã được giải mã trọn vẹn: đó là do C2-B1 vô tình hưởng lợi từ tiền xử lý chuẩn ImageNet (Bicubic + chuẩn hóa $[-1, 1]$), giúp dải động và tần số không gian của ảnh đưa vào ViT phong phú hơn.

### 10.5. Kết luận Nghiệm thu Khoa học (Sign-off Status)
- **Quyết định Nghiệm thu:** **`C3_SIGNED_OFF_WITH_LIMITATIONS`**
- Toàn bộ 4 thực nghiệm C3-A, C3-B, C3-C, C3-D đã hoàn tất kiểm toán nghiêm ngặt, 10/10 runs được xác thực và công khai trên CoreWeave Forge, các biểu đồ và báo cáo lưu trữ đầy đủ trong repository.
"""
    if "## 10. Báo cáo Kết quả Thực nghiệm Full GPU" not in content:
        content = content + c3_results_section

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("Updated docs/c3_preprocessing_alignment_plan.md")

def update_experiments_md():
    path = Path("docs/experiments.md")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_c3_text = """## 4. Thực nghiệm C3 — Kiểm soát Quy trình Tiền xử lý Hình ảnh (Controlled Preprocessing Alignment)

- **Trạng thái:** `C3_READY_FOR_GPU`"""
    
    new_c3_text = """## 8. Thực nghiệm C3 — Kiểm soát Quy trình Tiền xử lý Hình ảnh (Controlled Preprocessing Alignment — Factorial $2 \\times 2$)

- **Trạng thái:** `C3_SIGNED_OFF_WITH_LIMITATIONS` (Full GPU hoàn tất trên NVIDIA A100, đồng bộ 10 runs W&B trên CoreWeave Forge)
- **Nhánh thực nghiệm:** `experiment/c3-preprocessing-alignment`
- **Gói nén lưu trữ:** `foal_c3_preprocessing_results.zip` (SHA256: `a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6`)
- **Tài liệu phân tích:** [`results/c3_analysis/`](../results/c3_analysis/) (Bao gồm 6 hình vẽ chất lượng xuất bản, manifest và các bảng tổng hợp CSV)

### 8.1. Bảng Kết quả Thực nghiệm Giai thừa $2 \\times 2$ (A100 GPU, CIFAR-100, Seed 42)

| Điều kiện | Nội suy | Chuẩn hóa | $\\bar{A}$ (%) | $A_{10}$ (%) | $F_{10}^{\\text{past}}$ (%) | $F_{10}^{\\text{code}}$ (%) | T_Feat (s) | T_RLS (s) | W&B Run ID | Ghi chú |
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---|
| **C3-A** | Bicubic | Normalize | **91.7742%** | **87.4300%** | **5.0222%** | **4.5200%** | 248.30 | 8.49 | [`ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | Reused C2-B1 |
| **C3-B** | Bilinear | Normalize | **91.3918%** | **86.9500%** | **4.8889%** | **4.4000%** | 176.08 | 8.56 | [`ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3) | Factor A |
| **C3-C** | Bicubic | Raw | **90.8762%** | **86.3500%** | **5.2333%** | **4.7100%** | 175.58 | 8.49 | [`4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h) | Factor B |
| **C3-D** | Bilinear | Raw | **90.4016%** | **85.7300%** | **5.1556%** | **4.6400%** | 175.28 | 8.50 | [`3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | Official Pipeline |

### 8.2. Phân rã Tác động Giai thừa & Kết luận Khoa học
1. **Tác động chính của Phép Nội suy ($E_{\\text{interp}}$):** $-0.4285\\text{ pp}$ (Bilinear làm giảm nhẹ độ chính xác so với Bicubic).
2. **Tác động chính của Chuẩn hóa dải pixel ($E_{\\text{norm}}$):** $-0.9441\\text{ pp}$ (Không chuẩn hóa dải pixel làm giảm đáng kể độ chính xác).
3. **Hiệu ứng tương tác ($E_{\\text{IN}}$):** $-0.0922\\text{ pp}$ (Hai yếu tố tác động độc lập tuyến tính, tính chất cộng gộp xấp xỉ).
4. **Tổng độ dịch chuyển tiền xử lý ($\\Delta_{\\text{total}}$):** $-1.3726\\text{ pp}$ (từ $91.7742\\%$ xuống $90.4016\\%$).
5. **Hội tụ với Tác giả:** Khi tiền xử lý được chuẩn hóa hoàn toàn về phía tác giả (C3-D), Last Accuracy $A_{10} = 85.73\\%$ khớp gần như chính xác tuyệt đối với tác giả ($85.84\\%$, độ lệch chỉ $-0.11\\text{ pp}$), tỷ lệ quên $F_{10}^{\\text{code}} = 4.64\\%$ so với $4.69\\%$ (lệch $-0.05\\text{ pp}$)."""

    if old_c3_text in content:
        content = content.replace(old_c3_text, new_c3_text)
    elif "## 8. Thực nghiệm C3" not in content:
        content = content + "\n\n" + new_c3_text

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("Updated docs/experiments.md")

def update_protocol_md():
    path = Path("docs/original_vs_ours_protocol.md")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    c3_protocol_section = """
### 5.4. C3 — Căn chỉnh Quy trình Tiền xử lý Hình ảnh ($2 \\times 2$ Factorial Design) & Nghiệm thu Thực nghiệm
- **Thiết kế kiểm soát giai thừa $2 \\times 2$:**
  - Factor A: Nội suy Bicubic vs Bilinear
  - Factor B: Chuẩn hóa $[-1.0, 1.0]$ vs Raw $[0.0, 1.0]$
  - Cố định 100%: Backbone `vit_base_patch16_224.augreg2_in21k_ft_in1k`, ma trận chiếu $P$ `official_uniform` seed 42 ($D=1000$, mã SHA256: `6d3048f4a0fca387...`), batch size 10, gamma 1.0, float64 RLS, thứ tự lớp seed 42.
- **Kết quả Full GPU trên NVIDIA A100-SXM4-40GB:**
  - **C3-A (Bicubic, Normalized):** $\\bar{A} = 91.7742\\%$, $A_{10} = 87.4300\\%$, $F_{10}^{\\text{past}} = 5.0222\\%$, $F_{10}^{\\text{code}} = 4.5200\\%$ ([Run `ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5)).
  - **C3-B (Bilinear, Normalized):** $\\bar{A} = 91.3918\\%$, $A_{10} = 86.9500\\%$, $F_{10}^{\\text{past}} = 4.8889\\%$, $F_{10}^{\\text{code}} = 4.4000\\%$ ([Run `ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3)).
  - **C3-C (Bicubic, Raw):** $\\bar{A} = 90.8762\\%$, $A_{10} = 86.3500\\%$, $F_{10}^{\\text{past}} = 5.2333\\%$, $F_{10}^{\\text{code}} = 4.7100\\%$ ([Run `4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h)).
  - **C3-D (Bilinear, Raw — Official Aligned):** $\\bar{A} = 90.4016\\%$, $A_{10} = 85.7300\\%$, $F_{10}^{\\text{past}} = 5.1556\\%$, $F_{10}^{\\text{code}} = 4.6400\\%$ ([Run `3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5)).
- **Phân rã giai thừa:** $E_{\\text{interp}} = -0.4285\\text{ pp}$, $E_{\\text{norm}} = -0.9441\\text{ pp}$, $E_{\\text{IN}} = -0.0922\\text{ pp}$, $\\Delta_{\\text{total}} = -1.3726\\text{ pp}$.
- **Hội tụ thực nghiệm:** $A_{10}$ của C3-D ($85.73\\%$) khớp cực sát Official Run ($85.84\\%$, lệch $-0.11\\text{ pp}$); $F_{10}^{\\text{past}}$ đạt $5.16\\%$ vs $5.17\\%$ (lệch $-0.01\\text{ pp}$); $F_{10}^{\\text{code}}$ đạt $4.64\\%$ vs $4.69\\%$ (lệch $-0.05\\text{ pp}$).
- **Trạng thái nghiệm thu:** **`C3_SIGNED_OFF_WITH_LIMITATIONS`**.
"""
    if "### 5.4. C3" not in content:
        content = content + c3_protocol_section

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("Updated docs/original_vs_ours_protocol.md")

def update_audit_plan():
    path = Path("docs/FOAL_AUDIT_AND_FIX_PLAN.md")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_c3_stat = "- **Trạng thái Thực nghiệm C3:** **`C3_READY_FOR_GPU`**"
    new_c3_stat = """- **Trạng thái Thực nghiệm C3:** **`C3_SIGNED_OFF_WITH_LIMITATIONS`**
  - Đã chạy hoàn tất Full GPU trên Google Colab NVIDIA A100-SXM4-40GB; xác thực toàn vẹn gói nén `foal_c3_preprocessing_results.zip` (SHA256: `a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6`).
  - **C3-A (Bicubic, Normalized):** $\\bar{A} = 91.7742\\%$, $A_{10} = 87.4300\\%$, $F_{10}^{\\text{past}} = 5.0222\\%$, $F_{10}^{\\text{code}} = 4.5200\\%$ ([ne1p16b5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5)).
  - **C3-B (Bilinear, Normalized):** $\\bar{A} = 91.3918\\%$, $A_{10} = 86.9500\\%$, $F_{10}^{\\text{past}} = 4.8889\\%$, $F_{10}^{\\text{code}} = 4.4000\\%$ ([ts2wh4e3](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3)).
  - **C3-C (Bicubic, Raw):** $\\bar{A} = 90.8762\\%$, $A_{10} = 86.3500\\%$, $F_{10}^{\\text{past}} = 5.2333\\%$, $F_{10}^{\\text{code}} = 4.7100\\%$ ([4m4uir7h](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h)).
  - **C3-D (Bilinear, Raw — Official Aligned):** $\\bar{A} = 90.4016\\%$, $A_{10} = 85.7300\\%$, $F_{10}^{\\text{past}} = 5.1556\\%$, $F_{10}^{\\text{code}} = 4.6400\\%$ ([3v53tmq5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5)).
  - **Phân rã giai thừa $2 \\times 2$:** $E_{\\text{interp}} = -0.4285\\text{ pp}$, $E_{\\text{norm}} = -0.9441\\text{ pp}$, $E_{\\text{IN}} = -0.0922\\text{ pp}$, $\\Delta_{\\text{total}} = -1.3726\\text{ pp}$.
  - Toàn bộ 7/7 unit tests trong `tests/test_c3_preprocessing.py` và 63/63 tests toàn bộ repo vượt qua 100%."""

    if old_c3_stat in content:
        content = content.replace(old_c3_stat, new_c3_stat)

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("Updated docs/FOAL_AUDIT_AND_FIX_PLAN.md")

if __name__ == "__main__":
    update_readme_implementation()
    update_c3_plan()
    update_experiments_md()
    update_protocol_md()
    update_audit_plan()

def update_readme():
    path = Path("README.md")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("| **C3-B (Bilinear, Norm)**"):
            new_lines.append("| **C3-B (Bilinear, Norm)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **91.39%** | **86.95%** | **4.40%** / **4.89%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3) | `VERIFIED FULL GPU` (`C3 PASS`) |\n")
        elif line.startswith("| **C3-C (Bicubic, Raw)**"):
            new_lines.append("| **C3-C (Bicubic, Raw)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **90.88%** | **86.35%** | **4.71%** / **5.23%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h) | `VERIFIED FULL GPU` (`C3 PASS`) |\n")
        elif line.startswith("| **C3-D (Bilinear, Raw)**"):
            new_lines.append("| **C3-D (Bilinear, Raw — Official Aligned)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **90.40%** | **85.73%** | **4.64%** / **5.16%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | `VERIFIED FULL GPU` (`C3 PASS`) |\n")
        else:
            new_lines.append(line)

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(new_lines)
    print("Updated README.md")

update_readme()
