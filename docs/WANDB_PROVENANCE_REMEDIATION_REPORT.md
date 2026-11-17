# W&B Provenance Remediation Report — CIFAR-100 Runs

## 1. Thông tin chung & Tiến trình thực thi

- **W&B Project:** `caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction`
- **W&B Server:** `https://forge.coreweave.com`
- **Phạm vi (Scope):** Chỉ áp dụng cho 4 source runs CIFAR-100 (`seed=42`): `3v53tmq5` (F-OAL C3-D), `xlcme1c3` (Experience Replay), `c5pjewk4` (Official-protocol LwF), `ifl3rz2u` (Adapted LwF). Toàn bộ dự án còn có các thí nghiệm C1–C3 và Table 4 ablation với 3 seed (`42`, `123`, `2026`).
- **Script thực thi:** `scripts/backfill_wandb_provenance.py`
- **Các mốc thời gian chính (UTC / UTC+7):**
  - Bổ sung `run.config['provenance_remediation']` (cả 4 runs): `2026-11-16T14:34:24Z` (`21:34:24+07:00`)
  - Kiểm tra idempotency metadata: `2026-11-16T14:35:03Z` (`21:35:03+07:00`)
  - Nâng cấp cơ chế backup bất biến & kiểm thử local: `2026-11-16T14:45:47Z` (`21:45:47+07:00`)
  - Upload Artifact `3v53tmq5` (lần 1 — thất bại do type mismatch): `2026-11-16T14:59:31Z` (`21:59:31+07:00`)
  - Upload Artifact `3v53tmq5` (lần 2 — thành công `v0`): `2026-11-16T15:11:51Z` (`22:11:51+07:00`)
  - Upload Artifact `xlcme1c3` (thành công `v0`): `2026-11-16T15:22:16Z` (`22:22:16+07:00`)
  - Upload Artifact `c5pjewk4` (thành công `v0`): `2026-11-16T15:25:46Z` (`22:25:46+07:00`)
  - Upload Artifact `ifl3rz2u` (thành công `v0`): `2026-11-16T15:30:36Z` (`22:30:36+07:00`)

---

## 2. Tổng hợp 4 W&B Runs, Commit & Dirty Worktree

| Run ID | Run Name | Method | Provenance Classification | Verification Status | `source_code_base_commit` | `source_worktree_dirty` | `uncommitted_diff_status` | `import_script_reference_commit` |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `3v53tmq5` | `c3_d_bilinear_raw` | F-OAL (C3-D) | `ONLINE_TRAINING` | `VERIFIED_WITH_MINOR_ARTIFACT_GAP` (nay đã bổ sung `v0` post-hoc) | `15b8afc71f6109efdbf3ff98a801ffcc83aa58d5` | `True` | `MISSING` | `None` |
| `xlcme1c3` | `headline_er_replay_buffer5000_seed42` | Experience Replay (`B=5000`) | `POST_HOC_IMPORT` | `VERIFIED_LOCAL_PARTIAL_ON_WANDB` | `b0658c8837563d845253880da8efb1f0f11387fc` | `True` | `MISSING` | `8e977037dc3fb3e4ae231d111cf0e634deebcdee` |
| `c5pjewk4` | `headline_lwf_official_protocol_seed42` | Official-Protocol LwF (`SGD + Dynamic`) | `POST_HOC_IMPORT` | `VERIFIED_LOCAL_PARTIAL_ON_WANDB` | `c3dd3f23fbf2d8d94d481ec75daceb860190118e` | `True` | `MISSING` | `8e977037dc3fb3e4ae231d111cf0e634deebcdee` |
| `ifl3rz2u` | `headline_lwf_adapted_adamw_seed42` | Adapted LwF (`AdamW + Static`) | `POST_HOC_IMPORT` | `VERIFIED_LOCAL_PARTIAL_ON_WANDB` | `b0658c8837563d845253880da8efb1f0f11387fc` | `True` | `MISSING` | `8e977037dc3fb3e4ae231d111cf0e634deebcdee` |

> **Lưu ý về `source_code_base_commit`:** Do `source_worktree_dirty=True` và `uncommitted_diff_status="MISSING"` trên cả 4 runs, `source_code_base_commit` chỉ là base commit được ghi nhận trong metadata/summary gốc, **không** phải bằng chứng tái hiện chính xác toàn bộ mã nguồn đã thực thi tại runtime.

---

## 3. Provenance Artifacts đã upload & Nguồn gốc từng file

Tất cả các Artifact đều được gắn vào các run đã `finished` bằng `wandb.init(id=..., resume="must", settings=disable_code/git)` -> `live_run.log_artifact(art).wait()` -> `live_run.finish()`, gắn cờ `uploaded_post_hoc=True` để mô tả thời điểm gắn artifact (không làm thay đổi phân loại `ONLINE_TRAINING` của `3v53tmq5`).

| Run ID | Artifact Name | Type | Version | Digest | `bundle_sha256` |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `3v53tmq5` | `c3_d_bilinear_raw_results` | `evaluation` | `v0` | `3ff0b5a080b9d2364ba97451d8fe1343` | `903456d64538af7c5550ac4468efd844149753ef7a405ab9cde6836e939dca05` |
| `xlcme1c3` | `headline_er_provenance_bundle` | `provenance_results` | `v0` | `aed8812c3b9d747c5ce32d1ec319ad6a` | `190a5870a34471ed417e9a7800d1c978b79b0f756973870cc0497872db998491` |
| `c5pjewk4` | `headline_official_lwf_provenance_bundle` | `provenance_results` | `v0` | `fa15bbcd1453231b43fd8e56c59e5b73` | `d47e57555e4b1367de0363e08aaab47ed5612557410416dd134244f721f6a628` |
| `ifl3rz2u` | `headline_adapted_lwf_provenance_bundle` | `provenance_results` | `v0` | `1db2d7964f499ddd4069e18b676d8d01` | `9b612893419141bdd911d4fb7eff77a9d6dd5b15e9fd8fd73c9f08b42093d2f8` |

### Danh sách file được đóng gói & Nguồn gốc từng file

| Run ID | File trong Artifact | Size (B) | SHA256 | Nguồn gốc & Mức xác minh |
| :--- | :--- | :---: | :--- | :--- |
| `3v53tmq5` | `metrics.json` | 2,316 | `55a22febcf0cf18732fe87bf49b83b2fb3e4fce947833adea238501247557f52` | Xuất trực tiếp từ Colab offline run (`foal_c3_preprocessing_results.zip`) |
| `3v53tmq5` | `metadata.json` | 10,058 | `0a7f098570893fc8f2954b4f3e5630d4e57bb55a39f9c5ac55dbafee51673282` | Xuất trực tiếp từ Colab offline run (`foal_c3_preprocessing_results.zip`) |
| `3v53tmq5` | `task_accuracy.csv` | 628 | `68684dfb3d348c5838815a63fa45dd874bd84b3a427cd29d06f4c320fa5de1d7` | Xuất trực tiếp từ Colab offline run (`foal_c3_preprocessing_results.zip`) |
| `xlcme1c3` | `er_cifar100.yaml` | 929 | `c71dae905933083f6183d0abb938ba4e0fa5f12d448e4ed691190a87c696289e` | Có mặt trong `foal_cifar100_headline_results.zip` & repo; `config_execution_verified=False` |
| `xlcme1c3` | `summary.json` | 4,630 | `ff90b02c2b46353ac4d558e8aaa66854ca0e99a981230d9638ec3991d53f7d2c` | Kết quả Colab từ `foal_cifar100_headline_results.zip` |
| `xlcme1c3` | `task_matrix.csv` | 505 | `319b9394d26020422e48611314317dc4a7b4be54b43a3ac09fb70446235f9900` | Kết quả Colab từ `foal_cifar100_headline_results.zip` |
| `c5pjewk4` | `lwf_cifar100_official_protocol.yaml` | 1,150 | `4dd4a126de89a93e596dbddc1a83b586c82a3484027aa82e5593572b830cb698` | Có mặt trong `foal_cifar100_official_lwf_results.zip` & repo; `config_execution_verified=False` |
| `c5pjewk4` | `summary.json` | 4,917 | `9de1f84e0a7a37042e9b334ce9af3ac514222e86f1ee1dd70237311ec497ae70` | Kết quả Colab từ `foal_cifar100_official_lwf_results.zip` |
| `c5pjewk4` | `task_matrix.csv` | 505 | `1e5c7ca267a78f86cff486957f775184551553d8cda96efdf9db9ff43f4f0998` | Kết quả Colab từ `foal_cifar100_official_lwf_results.zip` |
| `ifl3rz2u` | `lwf_cifar100.yaml` | 826 | `6e1eb73336315ede165f0ffe5beca49ec430927ef87c0b4a519d6dee8806ecfb` | Có mặt trong `foal_cifar100_headline_results.zip` & repo; `config_execution_verified=False` |
| `ifl3rz2u` | `summary.json` | 4,230 | `4413d62b935903957c7f2843cb3342b2882dec6ce50b52dcaa4000e37d62b9a8` | Kết quả Colab từ `foal_cifar100_headline_results.zip` |
| `ifl3rz2u` | `task_matrix.csv` | 505 | `5a10c765158df0bb2c7953091ef6855019cc8c707782e63b672776c77dc42f6d` | Kết quả Colab từ `foal_cifar100_headline_results.zip` |

> **Ý nghĩa của `config_execution_verified=False`:** Mang nghĩa **chưa được xác minh trực tiếp tại runtime** (do thiếu `original_cli_command` và `original_stdout_log`), **không** đồng nghĩa với việc đã chứng minh cấu hình YAML bị sai.

---

## 4. Danh sách Snapshot Backup Bất Biến & SHA256

Thư mục lưu trữ: `results/headline_analysis/wandb_provenance_backups/` (mỗi file đều có sidecar `.json.sha256`):

### 4.1. Năm snapshot pre-upload đã xác minh SHA256 (`sidecar_ok=True`, được commit)

| Snapshot File | Thời điểm (UTC) | Mục đích | SHA256 |
| :--- | :--- | :--- | :--- |
| `wandb_provenance_backup_20261116T145917380400Z_ce11c44f7fea.json` | `14:59:17Z` | Trước lần upload Artifact `3v53tmq5` (lần 1 — thất bại) | `ce11c44f7fea49e742014c18efa9afadeab1a20290bfc1fcc8a5e7242aed53e8` |
| `wandb_provenance_backup_20261116T151100609703Z_483847db3c06.json` | `15:11:00Z` | Trước lần upload Artifact `3v53tmq5` (lần 2 — thành công) | `483847db3c06641228807ddd6a009d7c94728c57d5b48f3e2f973e6f67dd8dc4` |
| `wandb_provenance_backup_20261116T152111807420Z_b52c5a16f2f4.json` | `15:21:11Z` | Trước upload Artifact `xlcme1c3` | `b52c5a16f2f4dbf15eb0f395265d693d6eaf48e13e605c18c2b8993d286eba45` |
| `wandb_provenance_backup_20261116T152510749456Z_952fff47df1b.json` | `15:25:10Z` | Trước upload Artifact `c5pjewk4` | `952fff47df1b09867cb9b8590ddbc056d7affa3e3c42a7c6dae34652eb8fe3cc` |
| `wandb_provenance_backup_20261116T152946295146Z_a57fba574670.json` | `15:29:46Z` | Trước upload Artifact `ifl3rz2u` | `a57fba574670d9f1e32e1ce5e36ae968efebb85dcd18671d99fefb90ac55bd94` |

### 4.2. Snapshot thử nghiệm sớm giữ local (không commit)
- `wandb_provenance_backup_20261116T144516814274Z_588dcb9a8a98.json` (`1,734 B`, `14:45:16Z`): có sidecar `.sha256` không khớp (`sidecar_ok=False`; SHA256 thực tế trên đĩa là `440314f9d7fb9498519dc0fabf3b2afd99f546515e7d01d117b0e41a8298e97b` so với sidecar `588dcb9a8a98...`) do khác biệt line endings (`CRLF` trên Windows) trong quá trình tạo file ban đầu trước khi chuẩn hóa `newline="\n"`. File này được giữ nguyên vẹn ở local và không đưa vào commit.

---

## 5. Kết quả BEFORE / AFTER & Kiểm tra Idempotency

Kiểm tra độc lập qua W&B API sau toàn bộ các bước cho thấy không có bất kỳ sai lệch ngoài ý muốn nào:

| Run ID | `id` / `name` / `state` / `job_type` | `run.config` & legacy `provenance` | `summary` keys | `history_len` | `history_sha256` | Dry-Run Idempotency (sau cùng) |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| `3v53tmq5` | Khớp (`finished`, `None`) | Khớp (legacy `None` + `provenance_remediation`) | 40 | 11 | `e935eb5c3805cc28...` | `ALREADY_UP_TO_DATE` (Meta & Artifact) |
| `xlcme1c3` | Khớp (`finished`, `None`) | Khớp (legacy + `provenance_remediation`) | 24 | 13 | `7579561ca2f97683...` | `ALREADY_UP_TO_DATE` (Meta & Artifact) |
| `c5pjewk4` | Khớp (`finished`, `None`) | Khớp (legacy + `provenance_remediation`) | 24 | 13 | `108ec4487877e09c...` | `ALREADY_UP_TO_DATE` (Meta & Artifact) |
| `ifl3rz2u` | Khớp (`finished`, `None`) | Khớp (legacy + `provenance_remediation`) | 24 | 13 | `0fe584258bee1288...` | `ALREADY_UP_TO_DATE` (Meta & Artifact) |

Các Artifact gốc (`run-*-history:v0`, `run-*-task_accuracy_matrix:v0`) và toàn bộ `run.files()` gốc (bao gồm `output.log`, `wandb-metadata.json`, 4 ảnh/bảng media của 3 baselines) đều được bảo toàn.

---

## 6. Giới hạn về bằng chứng còn tồn tại

1. **Dirty worktree (`source_worktree_dirty=True`, `uncommitted_diff_status="MISSING"`):** Áp dụng cho cả 4 runs; bản diff chưa commit tại thời điểm chạy Colab không được lưu lại.
2. **Thiếu log & CLI gốc của 3 baselines (`POST_HOC_IMPORT`):** `original_stdout_log="MISSING"` và `original_cli_command="MISSING"` trên `xlcme1c3`, `c5pjewk4`, `ifl3rz2u`. File `output.log` (`220 B`) trên W&B của 3 runs này được tạo bởi `scripts/import_headline_to_wandb.py` khi import hậu kiểm, **không** phải stdout huấn luyện gốc.
3. **Cấu hình YAML của 3 baselines:** Có mặt trong ZIP Colab và repository, nhưng `config_execution_verified=False` (chưa xác minh trực tiếp cấu hình runtime, không đồng nghĩa cấu hình sai).
4. **Phạm vi dữ liệu & thực nghiệm:** Cuộc kiểm toán và remediation này **chỉ bao gồm 4 source runs CIFAR-100 (`seed=42`)** (`3v53tmq5`, `xlcme1c3`, `c5pjewk4`, `ifl3rz2u`), trong khi toàn bộ dự án còn có các thí nghiệm C1–C3 và Table 4 ablation với 3 seed (`42`, `123`, `2026`); đồng thời các kết quả này không đại diện cho khả năng tái lập trên toàn bộ các dataset khác của bài báo gốc.

---

## 7. Minh bạch hai sự cố trong quá trình thực thi

### Sự cố 1 — Snapshot backup đầu tiên bị ghi đè trong lần kiểm tra idempotency
- Trong lần chạy `--apply` đầu tiên (`2026-11-16T14:34:24Z`), script ghi snapshot trước cập nhật vào đường dẫn cố định `results/headline_analysis/wandb_provenance_backup_before_apply.json`.
- Ngay sau đó, khi chạy `--apply` lần thứ hai (`14:35:03Z`) để kiểm tra idempotency, script ghi đè vào chính đường dẫn cố định đó, làm mất bản snapshot gốc trước lần `--apply` đầu tiên. File `wandb_provenance_backup_before_apply.json` hiện tại **không** còn là bản snapshot gốc ban đầu (và chúng tôi không tái tạo giả bản gốc).
- **Khắc phục:** Sửa `scripts/backfill_wandb_provenance.py` sang `save_unique_backup_snapshot()` (tên file duy nhất theo UTC microsecond + SHA256, mở file chế độ exclusive `"x"`, kèm sidecar `.json.sha256`, và `[BACKUP SKIPPED]` khi tất cả runs đã `ALREADY_UP_TO_DATE`), được kiểm thử bởi `tests/test_backfill_wandb_provenance_backup.py`.

### Sự cố 2 — Lần upload Artifact `3v53tmq5` đầu tiên thất bại do Artifact Type Mismatch
- Lần chạy `--apply --include-artifacts --run-id 3v53tmq5` đầu tiên (`14:59:31Z`) thất bại ở bước hậu kiểm (`RuntimeError: Artifact upload verification failed on 3v53tmq5!`, `0 artifact file(s)`).
- **Nguyên nhân xác nhận từ `debug-internal.log` & W&B API:** Collection `c3_d_bilinear_raw_results` đã tồn tại từ trước trên project với `type="evaluation"` (0 version), trong khi script gửi `type="provenance_results"`, khiến GraphQL `createArtifact` trả HTTP `400` (`artifact type mismatch: expected evaluation but got provenance_results`). Vì `live_run.log_artifact(art)` chạy bất đồng bộ trong `wandb 0.31.0` chưa có `.wait()`, lỗi chỉ lộ ở bước hậu kiểm.
- **Khắc phục:** Đặt `artifact_type="evaluation"` riêng cho `3v53tmq5` trong `RUN_SPECS`, bổ sung `logged.wait()` trong `upload_artifact_via_resumed_run()`, và kiểm tra collection type mismatch ngay trong preflight. Lần retry duy nhất lúc `15:11:51Z` đã upload thành công `c3_d_bilinear_raw_results:v0` (`digest=3ff0b5a080b9d2364ba97451d8fe1343`).
