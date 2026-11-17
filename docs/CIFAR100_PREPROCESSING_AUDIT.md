# F-OAL CIFAR-100 Preprocessing Pipeline Forensic Audit (Official vs. C3-D)

**Audit Role:** Research Reproducibility Auditor  
**Target Benchmark:** CIFAR-100 (100 classes, 10 tasks × 10 classes)  
**C3 Reference Results:** C3-A (Bicubic + Normalized) = `91.77%` | C3-B (Bilinear + Normalized) = `91.39%` | C3-C (Bicubic + Raw) = `90.88%` | **C3-D (Bilinear + Raw) = `90.40%`**  
**Final Audit Verdict:** **`MATCH`** *(C3-D produces bitwise-identical `[0.0, 1.0]` input tensors to the Official F-OAL CIFAR-100 pipeline)*

---

## 1. Audited Repository Revisions

| Repository | Role | Exact Commit SHA |
| :--- | :--- | :--- |
| [`https://github.com/liuyuchen-cz/F-OAL`](https://github.com/liuyuchen-cz/F-OAL) | Official Author Implementation | `73e4ac1b4c965030973a2b77f05c57c2f44f49f5` |
| [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL) | Group Reproduction (C3-D Execution) | `15b8afc71f6109efdbf3ff98a801ffcc83aa58d5` |
| [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL) | Group Reproduction (Current Audited HEAD) | `86ed263739ac561d029ef227573a43de58c4f4a9` |

*(Xác minh `git_commit` thực thi của C3-D từ `results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/metadata.json`.)*

---

## 2. Official Implementation Evidence (`liuyuchen-cz/F-OAL`)

| Pipeline Stage | File & Lines | GitHub URL | Behavior & Meaning |
| :--- | :--- | :--- | :--- |
| **1. Raw Image Load & RGB** | `continuum/dataset_scripts/cifar100.py:19-24` | [cifar100.py#L19-L24](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/continuum/dataset_scripts/cifar100.py#L19-L24) | Tải `torchvision.datasets.CIFAR100`; lưu `train_data`/`test_data` dạng NumPy `uint8` shape `(N, 32, 32, 3)` kênh màu RGB `[0, 255]`. |
| **2. PIL Conversion** | `utils/setup_elements.py:50` | [setup_elements.py#L50](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/utils/setup_elements.py#L50) | `transforms.ToPILImage()` chuyển mảng `uint8` `(32, 32, 3)` sang `PIL.Image` RGB. |
| **3. Resize & Interpolation** | `utils/setup_elements.py:51` | [setup_elements.py#L51](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/utils/setup_elements.py#L51) | `transforms.Resize((224,224))` — không truyền `interpolation`, nên `torchvision` dùng mặc định **`InterpolationMode.BILINEAR`**. |
| **4. Tensor & Pixel Range** | `utils/setup_elements.py:52` | [setup_elements.py#L52](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/utils/setup_elements.py#L52) | `transforms.ToTensor()` chuyển sang `torch.float32` shape `(3, 224, 224)`, chia cho `255.0` đưa miền giá trị về **`[0.0, 1.0]`**. |
| **5. Explicit Normalization** | `utils/setup_elements.py:49-53` | [setup_elements.py#L49-L53](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/utils/setup_elements.py#L49-L53) | **Không có `transforms.Normalize(...)`**. Không áp dụng ImageNet mean/std hay ViT `(0.5, 0.5, 0.5)`. |
| **6. Train Preprocessing** | `agents/FOAL.py:21` | [FOAL.py#L21](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/agents/FOAL.py#L21) | Huấn luyện 1-pass gọi `dataset_transform(..., transform=transforms_match[self.data])`. Không dùng `transforms_aug` (vốn chỉ dành cho phương pháp replay/augment khác). |
| **7. Eval Preprocessing** | `continuum/data_utils.py:129` | [data_utils.py#L129](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/continuum/data_utils.py#L129) | Đánh giá gọi `dataset_transform(..., transform=transforms_match[params.data])` — giống hệt lúc train. |
| **8. Backbone Implicit Prep** | `models/pretrained.py:134,142-162` | [pretrained.py#L134-L162](https://github.com/liuyuchen-cz/F-OAL/blob/73e4ac1b4c965030973a2b77f05c57c2f44f49f5/models/pretrained.py#L134-L162) | `timm.create_model('vit_base_patch16_224', pretrained=True)` đưa thẳng `x` vào `self.vit.patch_embed(x)`; `timm` **không** normalize ngầm bên trong `forward`/`patch_embed`. |

---

## 3. Reproduction Implementation Evidence (`caoban123/F-OAL`, C3-D)

| Pipeline Stage | File & Lines | GitHub URL | Behavior & Meaning |
| :--- | :--- | :--- | :--- |
| **1. C3-D Config** | `configs/c3_d_bilinear_raw.yaml:24-28` | [c3_d_bilinear_raw.yaml#L24-L28](https://github.com/caoban123/F-OAL/blob/86ed263739ac561d029ef227573a43de58c4f4a9/configs/c3_d_bilinear_raw.yaml#L24-L28) | Khai báo `image_size: 224`, `interpolation: bilinear`, `normalize: false`, `mean: null`, `std: null`. |
| **2. Raw Image Load & RGB** | `src/foal/datasets.py:63-89,128-133` | [datasets.py#L128-L133](https://github.com/caoban123/F-OAL/blob/86ed263739ac561d029ef227573a43de58c4f4a9/src/foal/datasets.py#L128-L133) | `load_cifar100()` tải `uoft-cs/cifar100` (hoặc fallback `torchvision`); `IndexedImageDataset` ép `sample['img'].convert('RGB')` ra `PIL.Image` RGB. |
| **3. Resize & Interpolation** | `src/foal/datasets.py:105-107` | [datasets.py#L105-L107](https://github.com/caoban123/F-OAL/blob/86ed263739ac561d029ef227573a43de58c4f4a9/src/foal/datasets.py#L105-L107) | `transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)` khi `interpolation='bilinear'`. |
| **4. Tensor & Pixel Range** | `src/foal/datasets.py:108` | [datasets.py#L108](https://github.com/caoban123/F-OAL/blob/86ed263739ac561d029ef227573a43de58c4f4a9/src/foal/datasets.py#L108) | `transforms.ToTensor()` chuyển sang `torch.float32` shape `(3, 224, 224)` trên miền **`[0.0, 1.0]`**. |
| **5. Explicit Normalization** | `src/foal/datasets.py:110-111` | [datasets.py#L110-L111](https://github.com/caoban123/F-OAL/blob/86ed263739ac561d029ef227573a43de58c4f4a9/src/foal/datasets.py#L110-L111) | `if normalize:` bị bỏ qua khi `normalize=False`, nên **không** thêm `transforms.Normalize(...)`. |
| **6. Train & Eval Parity** | `scripts/train.py:288-291,452,507,535` | [train.py#L288-L291](https://github.com/caoban123/F-OAL/blob/86ed263739ac561d029ef227573a43de58c4f4a9/scripts/train.py#L288-L291) | Cả train (`IndexedImageDataset` dòng 452) và eval (dòng 507, 535) dùng chung một `trans = cifar100_transforms(...)`. |
| **7. Backbone Implicit Prep** | `src/foal/backbone.py:25,84-96` | [backbone.py#L84-L96](https://github.com/caoban123/F-OAL/blob/86ed263739ac561d029ef227573a43de58c4f4a9/src/foal/backbone.py#L84-L96) | `FrozenViT.forward` gọi thẳng `self.network.forward_features(images)`; `pretrained_cfg` (`mean=[0.5]*3, std=[0.5]*3, bicubic`) chỉ là metadata, **không** chạy ngầm. |

---

## 4. Comparison Matrix & Clarifying "Raw"

| Thành phần | Official F-OAL | Group C3-D | Kết quả đối chiếu |
| :--- | :--- | :--- | :--- |
| **Không gian màu** | RGB (`uint8` → `PIL`) | RGB (`.convert('RGB')`) | **Khớp 100%** |
| **Kích thước đầu vào** | `(224, 224)` | `(224, 224)` | **Khớp 100%** |
| **Nội suy (Interpolation)** | `Resize((224, 224))` → mặc định `BILINEAR` | `InterpolationMode.BILINEAR` | **Khớp 100%** |
| **Miền giá trị pixel** | `ToTensor()` → `float32` `[0.0, 1.0]` | `ToTensor()` → `float32` `[0.0, 1.0]` | **Khớp 100%** |
| **Normalization** | Không có (`Normalize` vắng mặt) | `normalize: false` (Không có `Normalize`) | **Khớp 100%** |
| **Preprocessing ngầm của timm** | Không (`forward` không gọi `create_transform`) | Không (`forward_features` không chuẩn hóa) | **Khớp 100%** |
| **Train / Eval Parity** | Dùng chung `transforms_match['cifar100']` | Dùng chung `cifar100_transforms(...)` | **Khớp 100%** |

### Làm rõ khái niệm "Raw"
Trong dự án này, **"Raw"** có nghĩa là **không áp dụng bất kỳ bước `transforms.Normalize(mean, std)` nào** (cả ImageNet lẫn ViT `0.5/0.5`), chỉ giữ phép scale chuẩn `[0, 255] → [0.0, 1.0]` của `torchvision.transforms.ToTensor()`. Cả mã nguồn gốc lẫn `timm` backbone đều không thực hiện chuẩn hóa ngầm sang `[-1.0, 1.0]`.

### Kiểm thử so sánh Tensor trực tiếp (Không huấn luyện lại)
Đã chạy kiểm thử trên cùng một ảnh RGB `32×32×3` (`uint8`) giữa `transforms_match['cifar100']` (Official) và `cifar100_transforms(size=224, interpolation='bilinear', normalize=False)` (C3-D):
- `t_official`: shape `(3, 224, 224)`, `torch.float32`, range `[0.0, 1.0]`
- `t_c3d`: shape `(3, 224, 224)`, `torch.float32`, range `[0.0, 1.0]`
- **`Max abs diff (Official vs C3-D)` = `0.0`** (`torch.equal(t_official, t_c3d) == True`)
- Đối chứng với **C3-A** (Bicubic + `Normalize(0.5, 0.5)`): range `[-1.0, 1.0]`, **`Max abs diff (Official vs C3-A)` = `1.1568627`**!

---

## 5. Remaining Non-Preprocessing Differences
1. **Nguồn tải dataset:** Official dùng `torchvision.datasets.CIFAR100`, nhóm ưu tiên `uoft-cs/cifar100` (HuggingFace) với fallback `torchvision`; cả hai cung cấp cùng tập 50,000 ảnh train và 10,000 ảnh test CIFAR-100 ở định dạng RGB 32×32.
2. **DataLoader `drop_last` ở Eval:** Official `setup_test_loader` (`continuum/data_utils.py:130`) đặt `drop_last=True` và `shuffle=True`, trong khi bản tái lập đặt `drop_last=False, shuffle=False` để đánh giá đầy đủ 100% mẫu test một cách tất định. Đây là khác biệt của DataLoader, không phải preprocessing ảnh.

---

## 6. Final Audit Verdict

**VERDICT: `MATCH`**

Cấu hình **C3-D (Bilinear + Raw `[0.0, 1.0]`)** khớp **chính xác tuyệt đối (bitwise)** với pipeline tiền xử lý CIFAR-100 trong mã nguồn chính thức của tác giả (`liuyuchen-cz/F-OAL`, commit `73e4ac1`). Việc không chọn **C3-A** (`91.77%`, vốn dùng Bicubic + Normalized `(0.5, 0.5, 0.5)`) làm cấu hình tái lập chính là hoàn toàn đúng đắn về mặt phương pháp luận, tránh thổi phồng độ chính xác thêm $+1.37\text{ pp}$ từ một pipeline tiền xử lý không có trong mã nguồn gốc.

---

## 7. Đoạn Kết Luận Tiếng Việt Cho W&B Report

> **Kiểm toán Tiền xử lý Ảnh (C3-D Parity Audit):** Đối chiếu từng dòng giữa mã nguồn chính thức của tác giả (`liuyuchen-cz/F-OAL`, commit `73e4ac1`, tệp `utils/setup_elements.py#L49-L53`) và pipeline tái lập của nhóm xác nhận mã nguồn gốc chỉ sử dụng `ToPILImage() → Resize((224, 224)) → ToTensor()` (nội suy mặc định **Bilinear**, đưa pixel về `[0.0, 1.0]` và **không** gọi `transforms.Normalize`, cũng không có bước chuẩn hóa ngầm trong `timm`). Kiểm thử trực tiếp trên cùng một ảnh RGB xác nhận tensor đầu ra của cấu hình **C3-D (Bilinear + Raw)** trùng khớp tuyệt đối với mã nguồn gốc (`max_abs_diff = 0.0`, `torch.equal = True`), trong khi C3-A (`91.77%`) sai lệch tới `1.157` do chuẩn hóa về `[-1.0, 1.0]`. Do đó, việc lựa chọn **C3-D (`90.40%`)** làm cấu hình tái lập chuẩn thay cho C3-A (`91.77%`) là bắt buộc để bảo đảm tính trung thực với cài đặt gốc của bài báo.
