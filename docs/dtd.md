# F-OAL — Yêu cầu hậu kiểm và hoàn thiện nhánh DTD

**Repository:** https://github.com/caoban123/F-OAL  
**Branch:** `experiment/dtd-ablation`  
**Commit được kiểm toán:** `cd0843e`  
**Dataset:** Describable Textures Dataset (DTD)

## 1. Mục tiêu

Nhánh DTD đã có mã nguồn, các cấu hình thực nghiệm C1–C3, Table 4 ablation và kết quả đa seed (42, 123, 2026).

Tuy nhiên, quá trình kiểm toán độc lập phát hiện một số điểm chưa nhất quán giữa **báo cáo, dữ liệu thực nghiệm được commit và nguồn gốc các run**.

Yêu cầu của bước này là hoàn thiện nhánh DTD để bảo đảm:

- Kết quả trong báo cáo khớp với dữ liệu nguồn.
- Mỗi chỉ số đều truy vết được đến source run tương ứng.
- Các giao thức thực nghiệm khác nhau được phân biệt rõ.
- Có hướng dẫn chạy lại thí nghiệm từ môi trường mới.
- Không làm thay đổi kết quả khoa học hoặc thuật toán chỉ để đạt được các con số mong muốn.

## 2. P0 — Giải quyết sự không nhất quán của kết quả

File `docs/dtd_ablation_report.md` hiện ghi:

| Cấu hình | Giá trị trong báo cáo |
|---|---:|
| A0 — Full F-OAL | 87,20% |
| A1 — Without FF | 85,01% |
| A2 — Without SP | 84,35% |
| A3 — Without Both | 81,10% |

Trong khi đó, `results/dtd/` đang lưu kết quả seed 42:

| Cấu hình | Giá trị từ `metrics.json` |
|---|---:|
| A0 | 85,08% |
| A1 | 82,08% |
| A2 | 77,81% |
| A3 | 74,58% |

Ngoài ra, báo cáo ghi kết quả đa seed A0 là **85,13 ± 0,11%**, nhưng tính lại từ các run A0 trong `results/dtd/` cho **85,85 ± 0,92%**.

Repository cũng lưu một nhóm kết quả khác tại `results/foal_full_dtd_seed42/` và `results/foal_multiseed_dtd_seed*/`, với thống kê A0 khoảng **85,07 ± 0,02%**.

### Yêu cầu

1. Xác định nguồn gốc của từng nhóm kết quả.
2. Giải thích nguyên nhân các nhóm kết quả khác nhau.
3. Chỉ định rõ bộ kết quả **chính thức dùng trong báo cáo cuối cùng**.
4. Tạo bảng đối chiếu cho A0–A3 trên ba seed.
5. Tính lại mean, sample standard deviation và các hiệu ứng FF/SP từ đúng bộ kết quả.
6. Đồng bộ toàn bộ bảng, hình ảnh và diễn giải trong báo cáo.
7. Nếu kết quả 87,20% vẫn được sử dụng, phải chỉ rõ source run, giao thức, seed và đường dẫn đến dữ liệu chứng minh.

**Không được tự ý sửa các giá trị trong `metrics.json` để khớp báo cáo.**

## 3. P0 — Kiểm tra giao thức dữ liệu

Bài báo F-OAL sử dụng:

- 40 trong 47 lớp DTD.
- 10 incremental tasks, mỗi task 4 lớp.
- 40 ảnh training và 40 ảnh testing cho mỗi lớp.
- One-pass training, không replay.

Trong repository có đề cập cả những thực nghiệm sử dụng 80 ảnh training/lớp.

Yêu cầu:

1. Phân biệt rõ protocol 40 ảnh/lớp và 80 ảnh/lớp.
2. Kiểm tra `class_order`, sampling seed và số mẫu mỗi task.
3. Xác minh train/test không bị trùng mẫu.
4. Bảo đảm các cấu hình A0–A3 sử dụng cùng class order và data split khi so sánh trong cùng seed.
5. Ghi lại dataset source, split và version/revision nếu có.
6. Kiểm tra sự nhất quán giữa `src/foal/dtd_datasets.py` và `src/foal/dataset_adapters/dtd.py`, đặc biệt là danh sách và ánh xạ tên lớp.

Nếu không có bằng chứng đầy đủ để xác nhận một điều kiện, cần ghi rõ giới hạn thay vì tuyên bố đã kiểm chứng tuyệt đối.

## 4. P1 — Hoàn thiện provenance

Kiểm tra các tệp:

- `metadata.json`
- `metrics.json`
- `task_accuracy.csv`
- `matrix.json`
- `summary.json`

Với từng source run, lập bảng gồm:

| Trường | Nội dung cần xác minh |
|---|---|
| Run ID | Tên và W&B ID nếu có |
| Seed | Base, class order, projection, stream |
| Git commit | Commit lúc training |
| Git dirty | Có thay đổi chưa commit không |
| Config | YAML thực sự sử dụng |
| Dataset | Source, split, revision |
| Preprocessing | Resize, interpolation, normalization |
| Backbone | Model ID và pretrained weights |
| Results | Đường dẫn các file kết quả |
| Verification | Đã xác minh trực tiếp hay chưa |

Nếu run có `dirty=True` mà không còn diff, cần ghi nhận giới hạn này.

## 5. P1 — Hoàn thiện khả năng chạy lại

Kiểm tra các entry point:

- `scripts/train_dtd.py`
- `scripts/run_dtd_suite.py`
- `scripts/prepare_dtd_cache.py`
- Các notebook Colab liên quan.

Yêu cầu cung cấp một **Quick Start dành cho người chấm** với các bước:

1. Clone đúng branch.
2. Cài dependencies.
3. Chuẩn bị hoặc tải dữ liệu.
4. Chạy một smoke test nhỏ.
5. Chạy full reproduction hoặc Table 4 ablation.
6. Xác định vị trí các kết quả đầu ra.

Phải phân biệt:

- `SMOKE_VERIFIED`: Đã chạy kiểm tra nhanh.
- `FULL_EXECUTION_VERIFIED`: Đã chạy đầy đủ.
- `NUMERICAL_REPRODUCTION_VERIFIED`: Đã đối chiếu kết quả số.

Không được gọi một chế độ là verified nếu chưa thực sự kiểm tra.

## 6. P1 — Kiểm thử và kiểm tra báo cáo

Cần có kiểm thử tự động xác nhận:

- 10 giai đoạn và 55 giá trị độ chính xác trong mỗi run.
- Average Incremental Accuracy tính lại khớp `metrics.json`.
- Final Accuracy khớp dòng cuối của ma trận.
- Forgetting được tính theo định nghĩa đã công bố.
- Các cấu hình cùng seed dùng chung fingerprint dữ liệu.
- Hình ảnh được tạo từ đúng nhóm kết quả chính thức.

Kiểm tra lại những kết luận mạnh về ý nghĩa thống kê. Chỉ sử dụng khi đã xác định đúng tập run và phương pháp kiểm định phù hợp.

## 7. Giới hạn phạm vi chỉnh sửa

- Chỉ làm việc trên `experiment/dtd-ablation`.
- Không sửa nhánh CIFAR-100, Aircraft hoặc `main`.
- Không sửa công thức F-OAL nếu chưa có bằng chứng về lỗi.
- Không xóa các kết quả lịch sử.
- Không thay đổi W&B hoặc ghi đè Artifact khi chưa được duyệt.
- Không chạy lại full GPU nếu chưa xác định rõ lý do và chi phí.

## 8. Deliverables

Khi hoàn tất, gửi lại:

1. **Final Results Table:** A0–A3 × 3 seed, mean ± std.
2. **Provenance Mapping:** Bảng ánh xạ từng kết quả đến source run.
3. **Protocol Clarification:** Giải thích các nhóm kết quả 40/80 ảnh.
4. **Updated Report:** Báo cáo và hình ảnh đã đồng bộ.
5. **Reproduction Instructions:** Lệnh chạy smoke/full.
6. **Test Results:** Danh sách test đã chạy, số passed/failed/skipped.
7. **Git Diff Summary:** Danh sách file đã sửa và lý do.

**Tiêu chí nghiệm thu:** Không còn mâu thuẫn không được giải thích giữa báo cáo và kết quả nguồn; các số liệu đưa lên slide đều có bằng chứng truy vết.

Sau đó báo cáo trạng thái `READY_FOR_INTEGRATION_REVIEW`. Không tự merge vào `main`.