# FGVC-Aircraft: tái lập F-OAL, ablation Table 4 và kiểm tra C1–C3

**Nhánh:** `experiment/fgvc-aircraft-ablation` · **Cập nhật:** 18/11/2026 · **Trạng thái:** `READY_FOR_INTEGRATION_REVIEW` (chưa merge vào `main`)

Báo cáo theo cùng khung với báo cáo DTD: ablation 3 seed → C1/C2/C3 → đối chiếu paper và code tác giả → forgetting → giới hạn, nguồn gốc, cách chạy lại. Mọi số dưới đây là của Aircraft; không chuyển số hay kết luận của CIFAR-100, DTD sang. Số % là accuracy; chênh lệch giữa hai accuracy tính bằng điểm (pp). SD là độ lệch chuẩn mẫu (`ddof=1`).

**Phát hiện chính**
- A0 đạt **64.22 ± 3.70%** trên 3 seed (paper 62.2). A0 > A1 > A2 > A3 ở cả 3 seed.
- Trên Aircraft, SP quan trọng hơn FF: hiệu ứng chính 4.52 so với 1.94.
- C1: đổi cách khởi tạo phép chiếu từ N(0,1) sang official_uniform tăng 18.78 điểm. C2: đổi checkpoint ViT tăng 14.64. C3: năm cách tiền xử lý chỉ chênh nhau 0.51.
- Chạy code tác giả không sửa trên Aircraft ra **61.91 / 53.67**, sát số paper (62.2 / 54.0).

## 1. Thiết lập chung

| Mục | Giá trị | Căn cứ |
|---|---|---|
| Dữ liệu | Split `train` chính thức (3.334 ảnh), `test` (3.333 ảnh), 100 lớp (mức variant) | Paper mục 4.1; code tác giả dùng `FGVCAircraft(split='train'/'test')` |
| Nguồn | Hugging Face `Multimodal-Fatima/FGVC_Aircraft_train` (revision `5b3589f`) và `_test` (`7aed01f`) | Đề môn: tải từ HF/Kaggle. Hai repo không đổi từ 2023 |
| Giao thức | 10 task × 10 lớp, học một lượt, batch 10, không replay | Paper mục 4.1 |
| Backbone | ViT-B/16 đóng băng, `vit_base_patch16_224.augreg2_in21k_ft_in1k` | C2 |
| Tiền xử lý | Resize **LANCZOS** về 224, ảnh thô [0, 1], không chuẩn hoá | Code tác giả `continuum/dataset_scripts/Aircraft.py` dòng 41, 54; `transforms_match['aircraft']` chỉ có `ToTensor` |
| FF | Trung bình CLS của 12 block | Paper |
| SP | 768 → 1000, `official_uniform` U(±1/√768), sigmoid | C1 |
| Bộ phân loại | RLS, γ = 1, float64 | Paper mục 4.7 |
| Seed | 42, 123, 2026; một seed đặt cùng lúc thứ tự lớp, phép chiếu, thứ tự dữ liệu | |
| Phần cứng | Kaggle Tesla T4, Python 3.13, torch 2.7.1+cu126, timm 1.0.30 | `metadata.json` |

LANCZOS là khác biệt riêng của Aircraft so với CIFAR-100: code tác giả dùng `transforms.Resize` mặc định (bilinear) cho CIFAR-100 nhưng resize LANCZOS bằng PIL cho Aircraft và DTD. Vì vậy cấu hình "sát code gốc" của Aircraft là LANCZOS thô, không phải bilinear thô như CIFAR.

## 2. Ablation Feature Fusion và Smooth Projection (Table 4, 3 seed)

Trong cùng một seed, bốn cấu hình dùng chung cách chia task và thứ tự lớp (kiểm bằng fingerprint trong `metadata.json`).

| Cấu hình | FF | SP | Paper | Nhóm $\bar A$ | Seed 42 / 123 / 2026 | Lệch so với paper |
|---|---|---|---|---|---|---|
| A0 đầy đủ | ✓ | ✓ | 62.2 | **64.22 ± 3.70** | 64.63 / 67.69 / 60.33 | +2.02 (+3.2%) |
| A1 bỏ FF | ✗ | ✓ | 60.9 | 62.98 ± 3.68 | 63.95 / 66.08 / 58.91 | +2.08 (+3.4%) |
| A2 bỏ SP | ✓ | ✗ | 58.7 | 60.40 ± 4.16 | 60.09 / 64.70 / 56.40 | +1.70 (+2.9%) |
| A3 bỏ cả hai | ✗ | ✗ | 56.0 | 57.75 ± 3.97 | 57.81 / 61.70 / 53.75 | +1.75 (+3.1%) |

**Mức tụt so với A0**, tính ghép cặp trong từng seed rồi lấy trung bình:

| Bỏ đi | Paper | Nhóm |
|---|---|---|
| FF | −1.3 | −1.24 ± 0.49 |
| SP | −3.5 | −3.82 ± 0.78 |
| Cả hai | −6.2 | −6.46 ± 0.43 |

**Hiệu ứng giai thừa 2×2** (cùng seed): $E_{FF}=\tfrac12[(A_0-A_1)+(A_2-A_3)]$, $E_{SP}=\tfrac12[(A_0-A_2)+(A_1-A_3)]$, $I=A_0-A_1-A_2+A_3$.

| Hiệu ứng | Paper | Nhóm (n = 3) |
|---|---|---|
| FF | 2.0 | 1.94 ± 0.42 |
| SP | 4.2 | 4.52 ± 0.83 |
| Tương tác FF × SP | −1.4 | −1.41 ± 0.18 |

Tương tác âm nghĩa là hai thành phần chồng lấn một phần: bỏ cả hai mất ít hơn tổng mức mất khi bỏ từng cái, không có nghĩa FF hay SP gây hại. Với n = 3, không khẳng định ý nghĩa thống kê.

**Dao động giữa các seed.** $\bar A$ dao động 60.33–67.69 trong khi $A_{last}$ gần như đứng yên (SD 0.01 ở A2, A3). A2 và A3 không có phép chiếu ngẫu nhiên, và nghiệm RLS sau mỗi task không phụ thuộc thứ tự dữ liệu trong task, nên với hai cấu hình này phần dao động của $\bar A$ đến từ thứ tự lớp. Paper không công bố seed cho Aircraft, nên một phần mức +2 so với paper có thể do khác thứ tự lớp.

## 3. Kiểm tra căn chỉnh C1, C2, C3 (seed 42)

Giống báo cáo DTD: mỗi phép kiểm chỉ đổi một yếu tố, chạy **một seed (42)**, không gộp với 12 run ablation ở mục 2. Cả 6 run mới chạy trên commit sạch `b0f3223` (`dirty=False`) và dùng đúng split của A0 seed 42. Các mốc dùng chung (C1-P1 = C2-B0, C2-B1 = C3-A, C3-E = A0 seed 42) là một run, không phải lần lặp độc lập.

| Điều kiện | $\bar A$ (%) | $A_{last}$ (%) | $F_{past}$ (pp) | $F_{code}$ (pp) |
|---|---|---|---|---|
| P0 Gaussian N(0,1) | 31.04 | 25.07 | 6.63 | 6.02 |
| P1 official_uniform | 49.82 | 40.58 | 12.41 | 11.17 |
| P2 Gaussian cùng phương sai | 49.95 | 41.21 | 12.05 | 10.84 |
| B0 augreg_in1k (= C1-P1) | 49.82 | 40.58 | 12.41 | 11.17 |
| B1 augreg2_in21k_ft_in1k | 64.46 | 54.48 | 12.58 | 11.32 |
| A bicubic, chuẩn hoá (= C2-B1) | 64.46 | 54.48 | 12.58 | 11.32 |
| B bilinear, chuẩn hoá | 64.97 | 55.17 | 12.58 | 11.32 |
| C bicubic, thô | 64.67 | 54.63 | 12.61 | 11.35 |
| D bilinear, thô | 64.88 | 54.66 | 12.58 | 11.32 |
| E LANCZOS, thô (= A0 seed 42) | 64.63 | 54.54 | 12.81 | 11.53 |
| Code tác giả, seed 0 | 61.91 | 53.67 | 11.82 | 10.64 |

**C1 – khởi tạo phép chiếu** (ViT `augreg_in1k`, bicubic, chuẩn hoá [-1, 1], giống DTD). P1 − P0 = +18.78; P2 − P0 = +18.91; P2 − P1 = +0.13. P2 có cùng phương sai lý thuyết với P1 và cho accuracy gần như bằng, nên điều quyết định là độ lớn phương sai (N(0,1) đẩy đầu vào sigmoid ra vùng bão hoà), không phải dạng phân phối uniform hay Gaussian.

**C2 – checkpoint ViT** (official_uniform, bicubic, chuẩn hoá). B1 − B0 = **+14.64** $\bar A$, +13.90 $A_{last}$. Mức này lớn gần gấp đôi DTD (+7.15). Paper chỉ ghi "ViT-B pre-trained on ImageNet-1K", nên phải ghi chính xác tag checkpoint mới tái lập được. Không quy toàn bộ mức tăng cho riêng ImageNet-21K, vì hai checkpoint còn khác cách huấn luyện và augmentation.

**C3 – tiền xử lý** (ViT `augreg2_in21k_ft_in1k`, official_uniform). Năm cách nằm trong khoảng 64.46–64.97, biên độ **0.51**, nhỏ hơn nhiều so với dao động giữa các seed (SD 3.70). Cách cao nhất là B (bilinear, chuẩn hoá). Nhóm vẫn chọn **E (LANCZOS thô)** vì đó là cách code tác giả làm, không chọn theo accuracy. Đây là kết quả một seed, chưa đủ để xếp hạng năm cách.

## 4. Đối chiếu với paper và với code tác giả

| | $\bar A$ | $A_{last}$ | Forgetting |
|---|---|---|---|
| Paper (Table 2) | 62.2 | 54.0 | 10.0 |
| Code tác giả, chạy lại, seed 0 | 61.91 | 53.67 | $F_{code}$ 10.64 · $F_{past}$ 11.82 |
| Code nhóm, A0, 3 seed | 64.22 ± 3.70 | 54.46 ± 0.29 | $F_{code}$ 10.50 ± 1.27 · $F_{past}$ 11.67 ± 1.42 |
| Code nhóm, A0, seed 42 | 64.63 | 54.54 | $F_{code}$ 11.53 · $F_{past}$ 12.81 |

Code tác giả chạy bằng lệnh gốc `python general_main.py --data aircraft --cl_type nc --agent FOAL --num_runs 1 --seed 0`, repo `liuyuchen-cz/F-OAL` commit `73e4ac1`, không sửa file nào, `timm==0.9.7` như `requirements.txt` của tác giả. Chỉ thêm một file `loss.py` ngoài repo vì code tác giả import nó cho agent DVC (F-OAL không dùng), giống cách nhóm đã chạy code tác giả trên CIFAR-100.

- Code tác giả ra 61.91 / 53.67, cách paper 0.3 điểm. $F_{code}$ (cách tính của code tác giả) 10.64 gần số 10.0 của paper hơn $F_{past}$ (công thức 17 của paper) 11.82, nên số forgetting trong paper nhiều khả năng sinh từ code.
- Code nhóm cao hơn code tác giả 2.72 điểm ở seed 42 so với seed 0. Hai lần chạy khác seed nên khác thứ tự lớp, mà mục 2 cho thấy thứ tự lớp làm $\bar A$ dao động tới ±3.7. Vì vậy đây là "gần nhau về số đo", không phải so khớp từng mẫu, và không kết luận code nào tốt hơn.
- Paper có ba số khác nhau cho cùng cấu hình Aircraft đầy đủ: 66.2 (mục 4.4), 62.2 (Table 2, 4), 61.1 (Table 6, γ = 1). Code tác giả chạy lại ra 61.91, gần 62.2 và 61.1 hơn.

## 5. Forgetting và ma trận độ chính xác từng task

| Cấu hình (3 seed) | $\bar A$ | $A_{last}$ | $F_{past}$ | $F_{code}$ |
|---|---|---|---|---|
| A0 đầy đủ | 64.22 ± 3.70 | 54.46 ± 0.29 | 11.67 ± 1.42 | 10.50 ± 1.27 |
| A1 bỏ FF | 62.98 ± 3.68 | 53.48 ± 0.23 | 11.49 ± 0.74 | 10.34 ± 0.67 |
| A2 bỏ SP | 60.40 ± 4.16 | 54.09 ± 0.01 | 8.26 ± 0.48 | 7.47 ± 0.41 |
| A3 bỏ cả hai | 57.75 ± 3.97 | 52.41 ± 0.01 | 7.33 ± 1.00 | 6.62 ± 0.90 |

$F_{past}$ theo công thức (17)–(18) của paper: đỉnh trước giai đoạn cuối, trung bình trên 9 task cũ. $F_{code}$ theo `experiment/metrics.py` của tác giả: đỉnh đến hết giai đoạn 10, trung bình trên cả 10 task. Hai số khác mẫu số, không thay cho nhau.

**Forgetting thấp không có nghĩa là học tốt hơn.** A3 quên ít nhất (7.33) nhưng $\bar A$ và $A_{last}$ thấp nhất: mô hình yếu từ đầu thì ít accuracy để mất. Bỏ SP làm $\bar A$ tụt 3.82 nhưng $A_{last}$ chỉ tụt 0.37, tức lợi ích của SP tập trung ở các giai đoạn đầu khi mới thấy ít lớp. Muốn giải thích cần thêm thí nghiệm.

Ma trận seed 42 (hàng = giai đoạn học, cột = task được đánh giá, %). Không lấy trung bình ma trận giữa các seed vì thứ tự lớp khác nhau.


#### A0 đầy đủ

| Giai đoạn | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86.3 | — | — | — | — | — | — | — | — | — |
| 2 | 80.3 | 71.0 | — | — | — | — | — | — | — | — |
| 3 | 75.8 | 63.4 | 65.7 | — | — | — | — | — | — | — |
| 4 | 72.2 | 61.9 | 58.7 | 61.1 | — | — | — | — | — | — |
| 5 | 69.9 | 53.2 | 54.2 | 58.4 | 67.1 | — | — | — | — | — |
| 6 | 70.4 | 49.2 | 52.1 | 53.0 | 65.3 | 77.8 | — | — | — | — |
| 7 | 68.7 | 46.2 | 52.7 | 50.3 | 62.9 | 73.1 | 65.3 | — | — | — |
| 8 | 67.8 | 45.3 | 51.2 | 49.1 | 61.1 | 71.0 | 65.0 | 64.9 | — | — |
| 9 | 64.8 | 44.7 | 50.0 | 46.1 | 60.2 | 68.6 | 65.9 | 61.3 | 50.8 | — |
| 10 | 64.5 | 43.5 | 48.8 | 44.6 | 59.9 | 62.3 | 64.0 | 59.8 | 47.7 | 50.3 |

#### A1 bỏ FF

| Giai đoạn | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86.6 | — | — | — | — | — | — | — | — | — |
| 2 | 78.8 | 71.0 | — | — | — | — | — | — | — | — |
| 3 | 75.8 | 62.5 | 63.3 | — | — | — | — | — | — | — |
| 4 | 71.3 | 62.5 | 58.1 | 57.5 | — | — | — | — | — | — |
| 5 | 70.4 | 54.4 | 56.3 | 56.9 | 64.7 | — | — | — | — | — |
| 6 | 69.3 | 50.5 | 53.3 | 53.6 | 62.9 | 75.4 | — | — | — | — |
| 7 | 67.8 | 47.7 | 51.5 | 52.1 | 60.8 | 70.4 | 64.0 | — | — | — |
| 8 | 67.2 | 47.4 | 51.8 | 48.2 | 59.6 | 69.5 | 65.0 | 62.5 | — | — |
| 9 | 63.9 | 42.9 | 49.1 | 45.5 | 56.6 | 67.1 | 63.4 | 58.6 | 50.5 | — |
| 10 | 63.9 | 43.2 | 48.2 | 44.6 | 58.4 | 62.3 | 61.6 | 59.5 | 45.9 | 49.4 |

#### A2 bỏ SP

| Giai đoạn | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 84.2 | — | — | — | — | — | — | — | — | — |
| 2 | 71.0 | 60.7 | — | — | — | — | — | — | — | — |
| 3 | 69.3 | 52.0 | 56.0 | — | — | — | — | — | — | — |
| 4 | 68.1 | 48.6 | 53.3 | 49.4 | — | — | — | — | — | — |
| 5 | 69.3 | 43.2 | 51.5 | 50.6 | 60.5 | — | — | — | — | — |
| 6 | 69.3 | 42.3 | 50.6 | 47.6 | 61.4 | 76.0 | — | — | — | — |
| 7 | 67.5 | 41.1 | 50.6 | 46.7 | 58.4 | 71.3 | 63.7 | — | — | — |
| 8 | 67.5 | 39.0 | 47.0 | 45.2 | 58.1 | 71.6 | 63.1 | 66.1 | — | — |
| 9 | 65.7 | 39.0 | 49.1 | 42.5 | 58.7 | 68.9 | 62.5 | 64.0 | 51.7 | — |
| 10 | 64.5 | 40.8 | 48.5 | 43.4 | 57.2 | 65.0 | 61.0 | 62.8 | 48.0 | 49.7 |

#### A3 bỏ cả hai

| Giai đoạn | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 83.0 | — | — | — | — | — | — | — | — | — |
| 2 | 64.2 | 56.8 | — | — | — | — | — | — | — | — |
| 3 | 62.7 | 46.8 | 48.2 | — | — | — | — | — | — | — |
| 4 | 65.1 | 45.3 | 47.9 | 48.2 | — | — | — | — | — | — |
| 5 | 66.9 | 42.6 | 48.8 | 52.1 | 57.8 | — | — | — | — | — |
| 6 | 68.1 | 43.8 | 50.0 | 47.0 | 60.8 | 74.6 | — | — | — | — |
| 7 | 66.3 | 43.2 | 47.6 | 48.5 | 56.3 | 71.9 | 62.2 | — | — | — |
| 8 | 67.8 | 40.8 | 47.6 | 44.6 | 57.8 | 69.8 | 60.1 | 62.2 | — | — |
| 9 | 66.9 | 41.4 | 45.5 | 42.8 | 56.6 | 66.5 | 59.2 | 59.8 | 49.2 | — |
| 10 | 66.9 | 39.0 | 46.1 | 42.2 | 55.7 | 63.8 | 57.4 | 58.6 | 45.6 | 48.8 |

## 6. Nguồn gốc code và dữ liệu

**13 run Table 4 ghi `dirty=True` và commit `e917492`.** Kernel Kaggle chạy chúng (`docs/aircraft_provenance/kaggle_table4_kernel.py`) làm theo thứ tự: clone repo, `git checkout e917492`, `git apply` một patch nhúng sẵn trong kernel (`kaggle_table4_kernel.patch`), rồi chạy. Patch làm git báo dirty. Kiểm lại:
- Dựng `e917492` + patch rồi so với commit `63cf9c0`: giống hệt từng file, chỉ thiếu `scripts/run_aircraft_ablation.py` (kernel tự viết vòng chạy, gọi `scripts/train.py` với đúng các tham số của script này).
- Log kernel (`kaggle_table4_kernel.log`) in `git status` khớp: sửa `train.py`, `fgvc_aircraft.py`, `datasets.py`, thêm `configs/aircraft/`.
- Số trong log khớp `metrics.json` của cả 13 run tới mọi chữ số.

Kết luận: code thực thi = `e917492` + patch = `63cf9c0` (trừ file runner). Không cần chạy lại.

**C1–C3** chạy bằng kernel thứ hai (`kaggle_c1c3_kernel.py`) trên commit sạch `b0f3223`, `dirty=False`. Run code tác giả: W&B run `9ko84fxj` được tạo **sau** khi job xong, từ ma trận đọc lại trong `official_stdout.log` (numpy in mỗi hàng thành hai dòng nên kernel không đọc được). Ma trận này khớp với số `general_main.py` tự in (avg_acc 0.61906, avg_end_acc 0.53667, fgt 0.10636), được kiểm trong test. Loại run: `POST_HOC_IMPORT`; các run còn lại: `ONLINE_TRAINING` (ghi offline rồi sync).

**Dữ liệu.**
- 3.334 ảnh HF `_train` trùng `torchvision.FGVCAircraft(split="train")` cả nhãn lẫn pixel (3.334/3.334).
- 3.333 ảnh HF `_test` trùng `split="test"` 3.333/3.333 và trùng `split="val"` 0/3.333. Vậy `_test` đúng là tập test, không phải val (log `kaggle_c1c3_kernel.log`, dòng `CROSSCHECK_TEST`).
- Danh sách 100 tên lớp của `_train` và `_test` giống hệt, cùng thứ tự.
- Mỗi run: tổng 3.334 ảnh train, 3.333 ảnh test, mỗi lớp xuất hiện ở đúng một task.
- Loader `src/foal/dataset_adapters/fgvc_aircraft.py` không còn đường tự chuyển sang dữ liệu giả; tải lỗi thì dừng.

Mã SHA-256 của mọi file kết quả và bằng chứng: `docs/aircraft_provenance/SHA256SUMS`.

## 7. Giới hạn

- Ablation chính n = 3; C1–C3 và code tác giả chỉ một seed mỗi điều kiện.
- Một seed đổi cùng lúc thứ tự lớp, phép chiếu và thứ tự dữ liệu, nên chưa tách được từng nguồn dao động (trừ A2, A3 như mục 2).
- Code tác giả chạy seed 0, code nhóm seed 42: khác thứ tự lớp, không so được từng mẫu.
- Môi trường khác lượt CIFAR-100 (torch 2.7.1 so với 2.11) vì bản torch có sẵn trên Kaggle lỗi trên T4.
- Run A2, A3 mang tag W&B `proj_gaussian_std1` do `train.py` tự gắn; hai cấu hình này không có phép chiếu nên tag không đúng.
- Run bổ sung `aircraft_a0_full_bilinear_seed42` (`mb7h3mz9`) hiện vẫn nằm trong group W&B `aircraft_table4_ablation`, tag `table4`. Đây là run C3-D, không thuộc 12 run Table 4; **cần chuyển sang group `c3_preprocessing_alignment_aircraft`** (chưa làm).

## 8. Cách chạy lại

```bash
git clone https://github.com/caoban123/F-OAL.git && cd F-OAL
git checkout experiment/fgvc-aircraft-ablation
pip install -r requirements.txt
# Kiểm kết quả đã lưu (CPU, không cần GPU, mạng, trọng số hay W&B)
python -m pytest -q tests/test_aircraft_ablation_results.py tests/test_dataset_adapters.py
# Chạy thử nhanh: 1 task, 20 ảnh/task
python scripts/train.py --config configs/aircraft/aircraft_a0_full.yaml --seed 42 --max-tasks 1 --limit-train-per-task 20 --limit-test-per-task 20 --output-dir results/smoke_aircraft
# Một run A0 đầy đủ (đổi a0_full thành a1_without_ff / a2_without_sp / a3_without_both, đổi seed thành 123 / 2026)
python scripts/train.py --config configs/aircraft/aircraft_a0_full.yaml --seed 42 --class-order-seed 42 --projection-seed 42 --stream-seed 42 --feature-cache-dir .cache/aircraft --output-dir results/aircraft_ablation/aircraft_a0_full_seed42
# C1-C3: cùng lệnh, đổi config sang configs/aircraft/c1_*.yaml, c2_*.yaml, c3_*.yaml
```

Mỗi run ghi `metrics.json`, `metadata.json`, `task_accuracy.csv`. Thêm `--wandb --wandb-offline` để ghi W&B offline.

| Lệnh | Trạng thái |
|---|---|
| `scripts/train.py` với `configs/aircraft/aircraft_a*.yaml` | Đã chạy thật (Kaggle, 07/11) |
| `scripts/train.py` với `configs/aircraft/c1_*.yaml`, `c2_*.yaml`, `c3_b/c_*.yaml` | Đã chạy thật (Kaggle, 18/11); config là đúng file kernel đã dùng |
| `configs/aircraft/c3_d_bilinear_raw.yaml` | Tương đương cấu hình đã chạy (A0 đổi `interpolation: bilinear`, kernel tạo bằng code); file này chưa được chạy trực tiếp |
| Lệnh chạy thử nhanh (`--max-tasks 1`) | Chỉ hỗ trợ trong code, chưa chạy với Aircraft |
| `scripts/run_aircraft_ablation.py` | Chỉ hỗ trợ trong code, **chưa từng chạy** (kernel tự viết vòng chạy với cùng tham số) |
| `tests/test_aircraft_ablation_results.py` | Đã chạy: 70 test qua (cùng `test_dataset_adapters.py`), CPU |

## 9. Run W&B và file kết quả

Project `hcmus-foal-reproduction`. Link: `https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/<id>`.

| Nhóm | Run | W&B | dirty | Code thực thi | Backbone | Tiền xử lý | FF, SP | Seed | $\bar A$ | SHA-256 `metrics.json` |
|---|---|---|---|---|---|---|---|---|---|---|
| Bổ sung (C3-D) | `aircraft_a0_full_bilinear_seed42` | `mb7h3mz9` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | bilinear, thô | FF, official_uniform | 42 | 64.88 | `e232eaa8f553` |
| Table 4 | `aircraft_a0_full_seed123` | `m1kze328` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | FF, official_uniform | 123 | 67.69 | `b2223e7e7644` |
| Table 4 | `aircraft_a0_full_seed2026` | `2qqecl0m` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | FF, official_uniform | 2026 | 60.33 | `a749dcc162b2` |
| Table 4 | `aircraft_a0_full_seed42` | `2qksp3kv` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | FF, official_uniform | 42 | 64.63 | `4dab4ed7b83f` |
| Table 4 | `aircraft_a1_without_ff_seed123` | `0n4xc8ap` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | không FF, official_uniform | 123 | 66.08 | `5889f724d7df` |
| Table 4 | `aircraft_a1_without_ff_seed2026` | `no2junlu` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | không FF, official_uniform | 2026 | 58.91 | `ce6885009133` |
| Table 4 | `aircraft_a1_without_ff_seed42` | `4fsupq4x` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | không FF, official_uniform | 42 | 63.95 | `44f52a16183b` |
| Table 4 | `aircraft_a2_without_sp_seed123` | `37254a6g` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | FF, không SP | 123 | 64.70 | `d7bda46c33dd` |
| Table 4 | `aircraft_a2_without_sp_seed2026` | `tp4fd91q` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | FF, không SP | 2026 | 56.40 | `0b8827bb4d61` |
| Table 4 | `aircraft_a2_without_sp_seed42` | `yezjox78` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | FF, không SP | 42 | 60.09 | `87863c589b08` |
| Table 4 | `aircraft_a3_without_both_seed123` | `ergopmva` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | không FF, không SP | 123 | 61.70 | `6a6cf3a778f2` |
| Table 4 | `aircraft_a3_without_both_seed2026` | `n8s1hjzc` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | không FF, không SP | 2026 | 53.75 | `4e5d2663db2f` |
| Table 4 | `aircraft_a3_without_both_seed42` | `gclk75xs` | True | e917492 + patch kernel (= 63cf9c0) | augreg2_in21k_ft_in1k | lanczos, thô | không FF, không SP | 42 | 57.81 | `c6bbbb89217a` |
| C1 | `aircraft_c1_p0_gaussian_std1_seed42` | `do82b0k3` | False | b0f3223 | augreg_in1k | bicubic, chuẩn hoá | FF, gaussian_std1 | 42 | 31.04 | `d0d16f2f11f4` |
| C1 | `aircraft_c1_p1_official_uniform_seed42` | `bemgqzjf` | False | b0f3223 | augreg_in1k | bicubic, chuẩn hoá | FF, official_uniform | 42 | 49.82 | `31b219f240fe` |
| C1 | `aircraft_c1_p2_gaussian_matched_variance_seed42` | `72oidkep` | False | b0f3223 | augreg_in1k | bicubic, chuẩn hoá | FF, gaussian_matched_variance | 42 | 49.95 | `dc70d9a5bc20` |
| C2 | `aircraft_c2_b1_augreg2_in21k_ft_in1k_seed42` | `rkgjxdvs` | False | b0f3223 | augreg2_in21k_ft_in1k | bicubic, chuẩn hoá | FF, official_uniform | 42 | 64.46 | `82f5bdd35fdf` |
| C3 | `aircraft_c3_b_bilinear_normalized_seed42` | `k0m9djvw` | False | b0f3223 | augreg2_in21k_ft_in1k | bilinear, chuẩn hoá | FF, official_uniform | 42 | 64.97 | `2b2197e71c6b` |
| C3 | `aircraft_c3_c_bicubic_raw_seed42` | `z64dog21` | False | b0f3223 | augreg2_in21k_ft_in1k | bicubic, thô | FF, official_uniform | 42 | 64.67 | `ac0a1575f299` |
| Code tác giả | `official_foal_aircraft_seed0` | `9ko84fxj` | không áp dụng | repo tác giả `73e4ac1`, không sửa | mặc định timm 0.9.7 | LANCZOS, thô | FF, SP | 0 | 61.91 | `0a53079e00dd` |

Group W&B: `aircraft_table4_ablation` (12 run Table 4 + run C3-D chưa chuyển), `c1_projection_initialization_aircraft`, `c2_backbone_alignment_aircraft`, `c3_preprocessing_alignment_aircraft`, `official_reproduction_aircraft`.

## 10. File thay đổi trên nhánh

| File | Lý do |
|---|---|
| `configs/aircraft/c1_*.yaml`, `c2_*.yaml`, `c3_*.yaml` | Config C1–C3 |
| `results/aircraft_c1_full/`, `aircraft_c2_full/`, `aircraft_c3_full/` | Kết quả C1–C3 |
| `results/official_foal_aircraft_seed0/` | Kết quả code tác giả (log gốc, csv tác giả, ma trận) |
| `docs/aircraft_provenance/` | Mã hai kernel Kaggle, patch, log, `SHA256SUMS` |
| `tests/test_aircraft_ablation_results.py` | Kiểm kết quả Aircraft, chạy CPU |
| `docs/aircraft_ablation_report_vi.md` | Báo cáo này |
| `.gitignore` | Cho phép commit các thư mục kết quả Aircraft |

Không sửa `src/`, `scripts/train.py`, kết quả cũ trong `results/aircraft_ablation/`, nhánh CIFAR-100, DTD hay `main`.
