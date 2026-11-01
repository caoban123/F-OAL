# Ngữ cảnh đồ án Nhập môn Machine Learning — HCMUS

## Paper đã chọn

**F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning**, NeurIPS 2024. Người dùng xác nhận nhóm đã đăng ký và được phê duyệt; GitHub nhóm: https://github.com/caoban123/F-OAL.

**Paper gốc:** https://proceedings.neurips.cc/paper_files/paper/2024/file/48ffa38c13078d6ce26b328e7f373243-Paper-Conference.pdf

**Repository đối chiếu của tác giả:** https://github.com/liuyuchen-cz/F-OAL

## Các deadline Moodle do nhóm cung cấp

| Đầu việc | Thời hạn (theo Moodle) | Sản phẩm |
|---|---|---|
| Nộp link repository | **07/11/2026 23:59** | URL GitHub công khai; tiếp tục commit sau ngày này được |
| Slide giữa kỳ | **21/11/2026 23:59** | `ID.pptx`: Reading + Reproduction |
| Poster cuối kỳ | **05/12/2026 23:59** | `ID.pptx`: Poster A2 dọc |

`ID` là ID nhóm, không phải MSSV.

## Tiêu chí project đã thống nhất

- Theme **Never-Ending Learning**; giới hạn công bố trên Moodle được cung cấp là **60 tháng**.
- Giữa kỳ: phân tích paper sâu, tự triển khai thuật toán cốt lõi, thực hiện reproduction với chứng cứ W&B, so sánh với paper.
- Nếu làm **Core Scope**: tối thiểu 2 thí nghiệm quan trọng (bao gồm headline comparison) và 1 phân tích bổ sung do nhóm tự thiết kế; xác nhận lại hướng dẫn chính thức khi lập kết quả cuối.
- Slide: 20–40 trang, speaker notes tối đa 500 từ/slide, citation IEEE tại slide, hình minh họa tự vẽ lại, file nhỏ hơn 20 MB.
- Cuối kỳ: phân tích 3 strengths, 2 theoretical limitations, 2 future directions; thực nghiệm chứng minh hạn chế, poster A2 dọc.
- Python >=3.9 theo quy định; starter này yêu cầu Python >=3.10 để sử dụng type hints (`|`).
- Dataset phải tải được bằng code từ nguồn được phép; không rò rỉ dữ liệu; mô hình online chỉ học một lượt.
- Không sao chép mã nguồn thuật toán chính của tác giả; không bịa số liệu hoặc W&B URLs.

Nguồn project: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/

## Trạng thái tại thời điểm tạo starter (01/11/2026)

- Xác nhận từ nhóm: paper phê duyệt, đã đọc paper, repo GitHub công khai.
- Repo ban đầu chỉ có README.md theo lời nhóm; bộ starter tạo riêng để merge vào repo, chưa push.
- Chưa xác minh bất kỳ thí nghiệm full-scale, download dataset, pretrained weights hay W&B run nào.
