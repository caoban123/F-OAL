# TỔNG QUAN ĐỒ ÁN NHẬP MÔN MÁY HỌC — HCMUS

> **Bản tổng hợp ngày 25/10/2026**. Tài liệu này dùng để tổ chức công việc và tự đối chiếu yêu cầu. Nếu Moodle hoặc giảng viên cập nhật quy định, nguồn mới nhất được ưu tiên.


## 1. Bức tranh toàn cảnh

Đồ án thực hiện **theo nhóm**. Mỗi nhóm đăng ký **một bài báo khoa học Machine Learning đã công bố**, rồi dành cả học kỳ để nghiên cứu bài báo đó. Mục tiêu không chỉ là giải thích nội dung hoặc chạy lại code: nhóm phải **hiểu phương pháp, tự triển khai đóng góp cốt lõi, tái hiện thí nghiệm, phân tích các giới hạn về lý thuyết và tự thiết kế thí nghiệm để kiểm chứng các điểm yếu**.

Đồ án gồm **3 milestones** và **2 labs**, nhưng được gom thành **hai sản phẩm đánh giá chính**:

| Giai đoạn | Hoạt động | Sản phẩm |
| --- | --- | --- |
| **Milestone 1** | Selecting the Paper | Form chọn paper trên Moodle, chờ giảng viên duyệt |
| **Milestone 2** | Reading the Paper | Cùng Lab 1 tạo **slide deck giữa kỳ** |
| **Lab 1** | Experimental Reproduction | Code và kết quả tái hiện, W&B runs, phần thực nghiệm trong slide |
| **Milestone 3** | Critical Analysis | Cùng Lab 2 tạo **poster cuối kỳ** |
| **Lab 2** | Empirical Demonstration | Thí nghiệm kiểm chứng các hạn chế, W&B runs, bằng chứng trong poster |
| **Kết thúc** | Final Oral Exam | **Poster workshop**: trình bày và bảo vệ trực tiếp trước giám khảo |

### Cách chấm điểm

| Thành phần | Tỷ trọng |
| --- | ---: |
| Form đăng ký paper | **Điều kiện bắt buộc**, không có điểm riêng |
| Slide deck — Reading + Reproduction | **45%** |
| Poster — Critical Analysis + Empirical Demonstration | **50%** |
| Chất lượng public repository, theo dõi W&B, định dạng bài nộp | **5%** |
| **Tổng** | **100%** |

**Đặc biệt:** nhóm chưa đăng ký paper sẽ **không được chấm điểm project**. Slide tốt phải giải thích và trực quan hóa paper đủ rõ, đồng thời phân tích các sai lệch khi tái hiện. Poster tốt phải nối được **một hạn chế có lập luận lý thuyết** với **một thí nghiệm được thiết kế đúng để kiểm chứng hạn chế đó**.

## 2. Chủ đề nghiên cứu và tiêu chí chọn paper

Theo thông báo đăng ký chủ đề trong Moodle, **chủ đề bắt buộc của đợt đồ án là _Never-Ending Learning_**. Ba nhóm hướng nghiên cứu được giới thiệu gồm:

| Nhánh | Các ví dụ trong thông báo |
| --- | --- |
| **Continual Learning** | Task Adaptation, Task Generalizability, Grateful Forgetting, Task-Free Continual Learning |
| **Online Learning** | Learning on Data Streams, Regret Minimization, Memory-Based Systems, Concept Drift |
| **Lifelong Learning** | Curriculum Learning, Persistent Learning, Self-Reflection Systems |

**Ràng buộc độ mới:** bài báo được chọn **không được công bố quá 5 năm (60 tháng)**. Cần kiểm tra **ngày công bố chính xác**, nhất là bài báo có nhiều bản arXiv và bản hội nghị/tạp chí. Ảnh Moodle không xác định rõ ngày dùng làm mốc tính ngược 60 tháng, nên trường hợp sát ngưỡng cần hỏi giảng viên/form.

Moodle **khuyến nghị** chọn paper có phương pháp dựa trên **những mô hình ML đã học ở lớp lý thuyết**. Đây là khuyến nghị để dễ hiểu, tự triển khai và phản biện; không nên diễn giải thành lệnh cấm mọi phương pháp khác.

Nhóm cần đọc **Google Form đăng ký** và **Topics Sheet** được liên kết trong Moodle, sau đó cử đại diện điền form. Thông tin đăng ký xuất hiện trong Google Sheet để giảng viên xét duyệt. **Khi paper đã được chấp thuận, nhóm gắn với paper đó cho cả học kỳ; muốn đổi cần có chấp thuận rõ ràng của giảng viên.** Chặng này chưa yêu cầu code hay báo cáo riêng.

**Tiêu chí thực tế nên xét trước khi đăng ký:** paper có mô tả phương pháp đủ rõ để tự code, dataset có thể tải tự động, chi phí máy phù hợp, ít nhất hai thí nghiệm chính có thể tái hiện, và có ít nhất hai giả định/giới hạn lý thuyết có thể kiểm chứng bằng thí nghiệm. Đây là **khuyến nghị lập kế hoạch**, không phải rubric bắt buộc được chép từ đề.

## 3. Các hạn nộp từ Moodle

| Bài nộp | Deadline hiển thị | Cách nộp |
| --- | --- | --- |
| **Đăng ký paper** | **Chưa thấy trong 4 ảnh** | Google Form trên Moodle, kiểm tra trạng thái được duyệt trong Topics Sheet |
| **Link GitHub repository** | **Thứ Bảy 07/11/2026, 23:59** | Paste **URL repo công khai** vào textbox Moodle |
| **Slide giữa kỳ** | **Thứ Bảy 21/11/2026, 23:59** | Upload file PowerPoint **`ID.pptx`** |
| **Poster cuối kỳ** | **Thứ Bảy 05/12/2026, 23:59** | Upload file PowerPoint **`ID.pptx`** |

**`ID` là số nguyên ID nhóm trong sheet lập nhóm**. Đừng tự thay bằng MSSV, tên thành viên, tên nhóm hoặc tên paper. Cùng mẫu tên `ID.pptx` được dùng cho slide và poster nhưng nộp ở **hai hoạt động Moodle khác nhau**.

**Phân biệt cực kỳ quan trọng:** 07/11 là **hạn nộp đường dẫn GitHub**, **không phải hạn commit cuối cùng**. Moodle cho phép tiếp tục commit đến trước hạn nộp poster cuối kỳ. Tuy nhiên nên đẩy code và tạo README có thể xem được từ trước 07/11.

Các mốc trên được chép từ ảnh Moodle ở định dạng 11:59 PM (= 23:59). **Ảnh không nêu rõ múi giờ cấu hình tài khoản Moodle**; hãy xem lại trên chính tài khoản khi đến gần hạn. Trạng thái “chưa nộp” trong ảnh chỉ là trạng thái tại thời điểm chụp, không chứng minh trạng thái hiện tại.

## 4. Milestone 2 — Reading the Paper

Mục tiêu là làm chủ nội dung bài báo đến mức có thể **giải thích đầy đủ mà không cần mở PDF**. Mọi nội dung phải được trình bày bằng cách hiểu và cách diễn đạt của nhóm, không sao chép nguyên văn.

**Thứ tự đọc nhanh được khuyên:** Abstract → Introduction → Conclusion → Background/Related Work → Experiments → Method. Cách đọc này giúp hiểu bài toán, lời hứa và kết luận trước khi đào sâu vào phần kỹ thuật.

### Nội dung nhóm phải phân tích

1. **Context & Motivation:** paper giải quyết bài toán gì, vì sao quan trọng và **một ví dụ thực tế do nhóm tự nghĩ**. Trình bày **ít nhất 2 phương pháp tiền nhiệm**; với từng phương pháp, nêu ý tưởng, hạn chế và hậu quả thực tiễn của hạn chế.
2. **Contributions & Novelty:** xác định các đóng góp thực sự mới so với tiền nhiệm, phân loại thành **theoretical**, **technical** và **empirical contributions**.
3. **Notation & Formal Problem:** thống nhất ký hiệu, giải thích từng đại lượng và viết bài toán dưới dạng toán học ở những chỗ paper có phát biểu hình thức.
4. **Overall Pipeline:** **tự vẽ sơ đồ** xuyên suốt từ input đến output.
5. **Components & Equations:** giải thích từng thành phần làm gì, tại sao cần và hoạt động thế nào; nêu công thức chính với toàn bộ biến được giải nghĩa.
6. **Theoretical Properties:** tóm tắt các lemma, theorem, proof sketch, ý nghĩa trực giác và giới hạn của chúng. **Không bắt buộc chứng minh lại toàn bộ**.
7. **Complexity:** phân tích độ phức tạp **thời gian và bộ nhớ**, so sánh với các phương pháp tiền nhiệm.

### Slide deck giữa kỳ — yêu cầu định dạng bắt buộc

| Hạng mục | Quy định |
| --- | --- |
| File | **Một file `.pptx`**, tên **`ID.pptx`** |
| Số slide | **20–40 slides**, tính cả appendix |
| Dung lượng | **Dưới 20 MB**, bao gồm speaker notes |
| Speaker notes | **Mỗi slide phải có**, tối đa **500 từ/slide** |
| Cách trình bày | Ưu tiên **hình tự vẽ, sơ đồ, biểu đồ, bảng**; không nhồi đoạn văn hoặc source code thô |
| Trích dẫn | **IEEE đánh số tại chân chính slide có trích dẫn**; **không dùng references slide riêng** |
| Hình của paper | **Không được copy/dán**; phải tự vẽ hoặc tái tạo |
| Kết quả thực nghiệm | So sánh kết quả paper và nhóm, có phân tích chênh lệch và liên kết tới **W&B run** |
| Kênh nộp | PPTX qua Moodle; **link repo nộp riêng** ở hoạt động tương ứng |

Ở checkpoint này, **không có buổi thuyết trình miệng**: giám khảo đọc file cùng speaker notes. Vì vậy, **speaker notes chính là phần phân tích và lời thuyết trình được chấm**; slide thiếu notes bị xem là chưa hoàn chỉnh.

## 5. Lab 1 — Experimental Reproduction

Nhóm phải **tự tái hiện thí nghiệm và kết quả của paper**, sau đó đo mức chênh lệch và giải thích tại sao kết quả khác nhau. Điểm quan trọng không phải khớp tuyệt đối với tác giả, mà là thí nghiệm chạy thật, có khả năng tái lập và **giải thích được sai lệch**.

### Phạm vi tái hiện — chọn một trong hai

- **Full Scope:** làm lại **tất cả thí nghiệm chính** của paper trên dataset gốc hoặc tương đương.
- **Core Scope:** làm lại **ít nhất 2 thí nghiệm quan trọng nhất**, **bao gồm headline comparison** (so sánh kết quả nổi bật nhất của paper), đồng thời thực hiện **ít nhất một phân tích bổ sung do nhóm tự thiết kế** thay cho các thí nghiệm còn lại, ví dụ sensitivity sweep.

Nhóm phải **ghi rõ lựa chọn phạm vi và lý do trong slide deck**. Với paper có thí nghiệm tốn kém, Core Scope có thể hợp lý, nhưng phân tích bổ sung phải có chất lượng.

### Quy định triển khai và tái lập

- **Tự code đóng góp cốt lõi dựa vào mô tả trong bài báo**. Nếu có repository chính thức của tác giả, nhóm **được đọc, chạy và so sánh** để xác minh, nhưng **không được copy code hoặc nộp lại code đó**.
- Được dùng các thư viện cho **phần phụ trợ** (data loader, metrics, standard model blocks...), nhưng phải ghi rõ, trích dẫn và giải thích trong README.
- Dùng **Python 3.9 trở lên**. Stack chuẩn: `numpy`, `scipy`, `scikit-learn`; dùng `torch`, `tensorflow` hoặc `jax` khi bài toán cần deep learning. Thư viện ngoài stack cần lý giải trong README.
- **Cố định và ghi lại random seeds**. Mỗi run phải đủ thông tin để tái thực hiện. Hàm quan trọng cần **docstring** nêu mục đích, tham số và giá trị trả về.
- Dataset phải **được tải bằng code từ Hugging Face hoặc Kaggle**, dữ liệu và cấu trúc đầu vào được chuẩn bị tự động khi chạy. Nếu dataset cần dùng chưa có công khai, nhóm tự chuẩn bị rồi upload lên một trong hai nền tảng. **Không dựa vào bước chuẩn bị thủ công ngoài notebook/script**.
- Máy thực nghiệm có thể là máy nhóm, cloud instance hoặc hosted notebook, miễn chạy **thật, hoàn chỉnh và được ghi nhận**.
- **Tất cả run** được ghi trên **một public Weights & Biases (W&B) project**, gồm cấu hình, metric, đồ thị và đường dẫn run.
- README có bảng ánh xạ **thí nghiệm → script/notebook → W&B URL → kết luận ngắn**, cộng hướng dẫn cài đặt và chạy.

### Kết quả cần báo cáo

Đặt **kết quả trong paper và kết quả tự tái hiện cạnh nhau** bằng bảng, đồ thị, đường cong hoặc cặp input–output phù hợp. Nếu là số, trình bày **relative deviation**. Phân tích sai khác do hyperparameters/config không được công bố đầy đủ, phiên bản thư viện, random initialization, phần cứng hoặc nguyên nhân kỹ thuật có bằng chứng.

**Khi paper có code chính thức**, nhóm còn phải **chạy code gốc**, so sánh với bản tự code và tìm nơi hai pipeline cho kết quả khác nhau. Nếu phát hiện vấn đề trong code gốc, cần chứng minh và giải thích rõ.

## 6. Milestone 3 — Critical Analysis

Phần này tập trung **phản biện về lý thuyết**, không chỉ nhận xét benchmark. Những hạn chế phát hiện ở đây sẽ trở thành câu hỏi thực nghiệm cho Lab 2.

Các hướng phản biện chính gồm:

- **Hidden Assumptions:** giả định về phân phối dữ liệu, tính độc lập, sự sẵn có của nhãn, đặc tính hàm tối ưu hoặc chi phí tính toán ẩn.
- **Soundness of Claims:** lemma/theorem có đúng với điều kiện ghi trong paper không; chứng minh có thiếu giả thiết; số liệu benchmark có thực sự hỗ trợ mọi tuyên bố không.
- **Problem Formulation & Objective:** objective/loss có thực sự đo đúng mục tiêu, có surrogate thiếu phù hợp hoặc nghiệm suy biến không.
- **Complexity & Generality:** tốc độ/bộ nhớ tăng thế nào trong thực tế; phương pháp có còn tốt ở quy mô hay phân phối ngoài thí nghiệm của paper không.

### Chỉ tiêu nội dung tối thiểu

| Loại phân tích | Yêu cầu |
| --- | --- |
| **Strengths** | **Ít nhất 3 ưu điểm**, có bằng chứng từ paper hoặc reproduction |
| **Limitations** | **Ít nhất 2 giới hạn lý thuyết**, có lập luận và có thể thiết kế thí nghiệm kiểm chứng |
| **Future Directions** | **Ít nhất 2 hướng phát triển**, mỗi hướng giải quyết một hạn chế đã nêu |

Một nhận xét yếu như “mô hình có thể hoạt động không tốt trên dữ liệu mới” là chưa đủ. Cần nói **giả định nào bị vi phạm, theo cơ chế nào hiệu suất dự kiến giảm, và thí nghiệm nào có thể chứng minh hoặc bác bỏ lập luận đó**.

## 7. Lab 2 — Empirical Demonstration và poster

Nhóm **tự thiết kế và chạy thí nghiệm** nhằm kiểm chứng những điểm yếu đã lập luận ở Critical Analysis. Đây **khác với Reproduction**: Reproduction hỏi *“Có tái tạo được kết quả tác giả không?”*; Empirical Demonstration hỏi *“Trong điều kiện nào, và vì sao, phương pháp không còn đáp ứng tốt lời hứa của paper?”*

Các dạng thí nghiệm được đề gợi ý:

- **Assumption-Violating Data:** thay đổi dữ liệu để cố ý phá giả định.
- **Adversarial / Out-of-Distribution:** kiểm tra dữ liệu gây khó hoặc ngoài phân phối.
- **Component Ablation:** bỏ hoặc làm yếu thành phần của phương pháp.
- **Replacement Baseline:** thay thành phần quan trọng bằng phương án đơn giản hơn.
- **Scaling Experiment:** đo wall-clock time/memory khi tăng quy mô.
- **Sensitivity Sweep:** thay một tham số để đánh giá độ nhạy và tính ổn định.

Lab 2 **kế thừa toàn bộ quy tắc tái lập, code và tracking của Lab 1**: seeds, script chạy được, dataset, W&B và repository. **Dùng cùng public W&B project**, nhưng **tag riêng run của Lab 2** để tránh nhầm với Reproduction.

### Poster cuối kỳ — yêu cầu bắt buộc

- **Chính xác 1 slide**, kích thước **A2 dọc (portrait)**, nằm trong **một file `ID.pptx`**.
- Mỗi điểm yếu phải có **lập luận lý thuyết đặt ngay bên bằng chứng thực nghiệm liên quan**.
- Mọi số liệu do nhóm tự tạo phải dẫn đến **W&B run**.
- Nếu có **hơn 2 hạn chế** được chứng minh thuyết phục cả lý thuyết lẫn thực nghiệm, **chỉ chọn 2 điểm nổi bật nhất** lên poster; các điểm còn lại cùng bằng chứng được mô tả trong **README.md**.
- Nộp qua Moodle, sau đó **trình bày và bảo vệ trực tiếp tại poster workshop** — đây là hình thức **thi vấn đáp cuối kỳ**. Mỗi thành viên có thể bị hỏi về toàn bộ project.

## 8. Repository, W&B và khả năng tái lập

Toàn nhóm duy trì **một public GitHub repository xuyên suốt học kỳ**. Code, notebooks, tài liệu, README và lịch sử commit được tích lũy liên tục. Những phát biểu, hình vẽ và số liệu trong deck/poster phải có khả năng **truy ngược về source code và kết quả chạy cụ thể**.

Một cấu trúc quản lý có thể áp dụng (***đây là gợi ý, không phải cấu trúc bắt buộc của đề***):

```text
project/
├── README.md
├── requirements.txt
├── configs/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── experiments/
│   ├── reproduction/
│   └── empirical_demonstration/
├── notebooks/
└── docs/
```

README nên giải thích mục tiêu, cách chuẩn bị môi trường, cách tải dataset và tái hiện từng thí nghiệm, seeds/config, phụ thuộc và trích dẫn các phần được phép tái sử dụng. Một bảng mapping hữu ích:

| Experiment | Script/Notebook | W&B Run | Ý nghĩa |
| --- | --- | --- | --- |
| Reproduce main baseline | Đường dẫn file thực tế | URL run thực tế | Đối chiếu headline result |
| Sensitivity analysis | Đường dẫn file thực tế | URL run thực tế | Xác định độ nhạy |
| Test limitation #1 | Đường dẫn file thực tế | URL run thực tế | Kiểm chứng hạn chế thứ nhất |

**Không sử dụng các hàng ví dụ này như bằng chứng thật**; thay bằng run đã thực thi của nhóm.

## 9. Liêm chính học thuật và sử dụng AI

**Bị điểm 0 project nếu sao chép** nội dung/hình ảnh của paper, code từ implementation chính thức hoặc bài làm của nhóm khác. Mục tiêu là **tự hiểu và tự triển khai**, không thay nhãn lên sản phẩm sẵn có.

**AI được phép hỗ trợ** viết, giải thích, lập trình và tổ chức công việc. Tuy nhiên nhóm phải kiểm tra kết quả, nắm rõ code và lý thuyết, đồng thời mỗi thành viên phải giải thích được bất kỳ phần nào khi bảo vệ. Giảng viên có quyền đánh giá bổ sung khi thấy cần.

## 10. Checklist ngắn trước từng mốc

### Trước khi đăng ký paper

- [ ] Paper thuộc phạm vi **Never-Ending Learning**.
- [ ] **Không quá 60 tháng** tính theo cách giảng viên/form quy định.
- [ ] Đọc rõ motivation, contributions, datasets và thí nghiệm cốt lõi.
- [ ] Nhóm có điều kiện tự code và tái hiện **ít nhất 2 thí nghiệm quan trọng** nếu chọn Core Scope.
- [ ] Có triển vọng xác định **ít nhất 2 điểm yếu lý thuyết** có thể kiểm chứng.
- [ ] Kiểm tra quy định trong **Google Form / Topics Sheet** và xác nhận được duyệt.

### Trước hạn GitHub — 07/11/2026

- [ ] Public GitHub repository đã tạo và truy cập được.
- [ ] README ban đầu rõ ràng; có code/notebooks bắt đầu triển khai và lịch sử commit.
- [ ] Đã **paste đúng URL công khai** vào textbox Moodle.
- [ ] Có kế hoạch tiếp tục commit sau 07/11.

### Trước hạn slide — 21/11/2026

- [ ] **20–40 slides**, tính cả appendix; file **`ID.pptx` dưới 20 MB**.
- [ ] Mỗi slide có **speaker notes ≤ 500 từ**.
- [ ] Các ý quan trọng, công thức, thành phần, pipeline và figures được **giải thích / tự vẽ**.
- [ ] Có **ít nhất 2 predecessor approaches** với giới hạn và hậu quả.
- [ ] Đã chọn **Full/Core Scope** và nêu lý do.
- [ ] Kết quả paper và nhóm đặt cạnh nhau; có **relative deviations**, phân tích sai khác.
- [ ] Mỗi kết quả nhóm có **W&B URL**; code/dataset/seeds có thể tái lập.
- [ ] Citation **IEEE dưới mỗi slide cần nguồn**, không làm references slide riêng.
- [ ] Nộp PPTX và repo link vào **đúng hai mục Moodle tương ứng**.

### Trước hạn poster — 05/12/2026

- [ ] Có **≥ 3 strengths, ≥ 2 theoretical limitations, ≥ 2 future directions** được lập luận.
- [ ] Mỗi limitation quan trọng có **thí nghiệm thiết kế riêng** và kết quả đã chạy.
- [ ] W&B runs Lab 2 được **tag riêng**, cùng project với Lab 1.
- [ ] Poster **đúng một slide A2 dọc**, file **`ID.pptx`**.
- [ ] Nếu chứng minh được hơn 2 điểm yếu, poster chỉ chọn **2 đáng chú ý nhất**; phần còn lại có trong README.
- [ ] Lập luận lý thuyết nằm sát kết quả thực nghiệm; số liệu có đường dẫn W&B.
- [ ] Repo, README và lịch sử commit ở trạng thái có thể trình bày, tái chạy.
- [ ] Mọi thành viên đủ khả năng giải thích **toàn bộ poster, phương pháp và thí nghiệm**.

## 11. Những dữ kiện nhóm còn cần tự xác nhận

Hiện các tài liệu/ảnh **chưa cho biết**: **ID nhóm**, danh sách thành viên, paper đã được duyệt hay chưa, deadline chính xác của Google Form chọn paper, dataset/máy nhóm sử dụng, link repository và W&B project, thời điểm tổ chức poster workshop cũng như múi giờ tài khoản Moodle. Khi lập kế hoạch thực hiện, **không tự suy đoán các thông tin này**.

## 12. Nguồn tham khảo

### Đề bài chính thức — HCMUS Course Lab Book

1. [Term Project: Overview](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)
2. [Selecting the Paper](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/selection/)
3. [Reading the Paper](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading/)
4. [Experimental Reproduction](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/)
5. [Critical Analysis](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis/)
6. [Empirical Demonstration](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics/)

### Thông báo Moodle từ ảnh người dùng

- **Ảnh 1 — Đăng ký chủ đề nghiên cứu:** chủ đề Never-Ending Learning, ba hướng chính, giới hạn paper 60 tháng, Google Form và Topics Sheet.
- **Ảnh 2 — GitHub Repository:** hạn **07/11/2026 23:59**, nộp link repo công khai, không phải hạn commit cuối.
- **Ảnh 3 — Slide giữa kỳ:** hạn **21/11/2026 23:59**, file **`ID.pptx`**.
- **Ảnh 4 — Poster cuối kỳ:** hạn **05/12/2026 23:59**, file **`ID.pptx`**.

---

**Kết luận:** Đồ án có hai trục cần triển khai song song: **(1) hiểu và tái hiện paper trung thực** để đạt mốc slide giữa kỳ; **(2) tìm giới hạn có cơ sở lý thuyết và chứng minh bằng thí nghiệm mới** để đạt mốc poster. **Tính tái lập, tracking W&B, repository và kỷ luật trích dẫn** là yêu cầu xuyên suốt, không phải phần phụ xử lý ở cuối.
