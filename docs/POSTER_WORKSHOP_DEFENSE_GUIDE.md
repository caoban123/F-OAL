# KỊCH BẢN TRÌNH BÀY & BỘ CÂU HỎI PHẢN BIỆN TẠI POSTER WORKSHOP (FINAL ORAL EXAM)

> **Học phần:** Nhập môn Học máy (*Introduction to Machine Learning*) — Trường ĐHKHTN, ĐHQG-HCM (HCMUS)
> **Hình thức thi cuối kỳ:** **Poster Workshop** — Nhóm treo Poster khổ dọc `Portrait A2` ([`poster/F-OAL_Critical_Analysis_Poster_A2.pptx`](../poster/F-OAL_Critical_Analysis_Poster_A2.pptx)), trình bày trực tiếp cạnh Poster trong **3–4 phút** và trả lời vấn đáp mở từ hội đồng giám khảo.

---

## 1. Kịch Bản Pitch 3 Phút 30 Giây Đứng Cạnh Poster (Chia Đều Cho 3 Thành Viên)

### Thành viên 1 (0:00 – 1:10) — Mở đầu, Bản đồ Kiến trúc & 3 Điểm mạnh có Bằng chứng (Ô `01` & `02`)
* *(Chỉ tay vào Ô `01` — Sơ đồ luồng F-OAL)*:
  "Chào thầy/cô. Nhóm em nghiên cứu bài báo **F-OAL (NeurIPS 2024)** giải bài toán Học tăng dần trực tuyến một lượt không lưu mẫu cũ (*Exemplar-Free Online CIL*). Ý tưởng trung tâm của bài báo là đóng băng toàn bộ $85.8$ triệu tham số của ViT-B/16, lấy trung bình 12 khối qua *Feature Fusion*, mở rộng phi tuyến qua *Smooth Projection* $\sigma(\mathbf{h}\mathbf{P})$, rồi giải bài toán hồi quy bình phương tối thiểu bằng cập nhật đệ quy **Woodbury RLS** mà không dùng lan truyền ngược."
* *(Chỉ tay sang Ô `02` — 3 Điểm mạnh S1–S3)*:
  "Qua tái lập độc lập trên 3 bộ dữ liệu **CIFAR-100, DTD và FGVC-Aircraft**, nhóm xác nhận **3 điểm mạnh thực chất**:
  1. **Thứ nhất (`S1`)**: Nghiệm một lượt trùng khít nghiệm học đồng thời toàn tập đến sai số $10^{-12}$, giữ độ quên CIFAR-100 chỉ $5.16\%$ không cần bộ đệm mẫu cũ — vượt xa Experience Replay ($16.74\%$) và LwF.
  2. **Thứ hai (`S2`)**: Ở $D = 1{,}000$, ma trận $\mathbf{R}$ chỉ chiếm $7.63\text{ MB}$ và cập nhật 10 tasks trên DTD mất chưa đầy $1.2$ giây.
  3. **Thứ ba (`S3`)**: Theo Định lý Cover, mở rộng phi tuyến cực kỳ hiệu quả trên miền kết cấu/phân loại tinh, mang lại **$+10.61\text{ pp}$ trên DTD** và **$+6.46\text{ pp}$ trên FGVC-Aircraft**.
  Tuy nhiên, khi kiểm toán sâu nền tảng toán học của bài báo, nhóm phát hiện **hai giả định ẩn chí mạng** bị tác giả bỏ ngỏ."

---

### Thành viên 2 (1:10 – 2:20) — Trục Phản biện I: Bẫy Mất Cân Bằng Lớp Của Hàm Mục Tiêu MSE One-Hot (Ô `03A` & `03B`)
* *(Chỉ vào khung công thức đỏ Ô `03A`)*:
  "Điểm gãy thứ nhất nằm ở việc thay Cross-Entropy bằng **MSE không trọng số trên nhãn one-hot $\{0, 1\}$**. Ở Hình 3 trang 9 bài báo, tác giả tuyên bố F-OAL không bị thiên lệch vì chuẩn cột trọng số $\|\hat{\mathbf{w}}_c\|_2$ đồng đều giữa các lớp. Bằng biến đổi Sherman–Morrison chính xác, nhóm chứng minh $\|\hat{\mathbf{w}}_c\|_2$ thực chất **tỉ lệ thuận với cỡ mẫu $N_c$** — Hình 3 phẳng đẹp chỉ vì tác giả dùng tập dữ liệu cân bằng tuyệt đối ($N_1 = \dots = N_C$)! Khi có $V - N_c$ mẫu thuộc lớp khác ép mục tiêu bằng $0$ trên nón kích hoạt dương của Sigmoid, chuẩn cột lớp hiếm bị sụp đổ theo $\mathcal{O}(1/\rho)$."
* *(Chỉ sang Hình 2 Ô `03B`)*:
  "Để chứng minh không bị lẫn với thứ tự thời gian, nhóm thiết kế thực nghiệm **54 lượt chạy Long-Tail** phân bổ đều lớp hiếm khắp 10 tasks trên 60,000 ảnh CIFAR-100 qua 3 seeds. Đúng như phương trình dự báo: hệ số tương quan giữa $\|\hat{\mathbf{w}}_c\|_2$ và $N_c$ vọt lên **$r = +0.89$** (Hình 2C), kéo độ chính xác nhóm lớp hiếm ở $\rho = 100$ rơi tự do **$-60.84\text{ pp}$** — từ $87.48\%$ xuống chỉ còn $26.64\%$!
  Để khắc phục mà vẫn giữ nguyên tính chất **1 lượt, 0 lưu mẫu**, nhóm đề xuất **`BalancedOnlineRidgeRLS` (`FD1`)**: nhân trọng số tần suất nghịch đảo nhân quả ngay trong cập nhật Woodbury kết hợp chuẩn hóa biên độ cột lúc suy luận, giúp khôi phục **$+42.67\text{ pp}$** cho nhóm lớp hiếm và **$+13.32\text{ pp}$** độ chính xác toàn cục."

---

### Thành viên 3 (2:20 – 3:30) — Trục Phản biện II: Lỗi Độ Phức Tạp, Bẫy Suy Biến Phổ $D \approx V$ & Tổng kết (Ô `04A`, `04B` & `05`)
* *(Chỉ vào Ô `04A`)*:
  "Điểm gãy thứ hai nằm ở câu chuyện độ phức tạp và điều kiện phổ khi mở rộng chiều chiếu $D$. Trước hết, nhóm phát hiện **Phương trình (14) trang 6 của bài báo viết sai đại số** khi đưa số lớp $C$ vào độ phức tạp cập nhật $\mathbf{R}$ là $\mathcal{O}(D^2 C)$; thực tế từ Eq. (11), bước cập nhật $\mathbf{R}$ hoàn toàn độc lập với $C$ nhưng tốn **$\Theta(D^2 S)$ mỗi batch**, tức nổ bậc hai $\Theta(V D^2)$ toàn khóa học. Nguy hiểm hơn, qua khai triển bộ lọc phổ kỳ dị Tikhonov, khi cỡ mẫu $V$ nhỏ (như DTD $1{,}600$ ảnh hay Aircraft $3{,}334$ ảnh) mà chiều chiếu $D$ xấp xỉ $V$ và $\gamma$ nhỏ, các trị riêng sát $0$ khuếch đại nhiễu lên $\frac{1}{2\sqrt{\gamma}}$ — giải thích trọn vẹn vì sao Aircraft ở Bảng 6 bài báo sập từ $61.1\%$ xuống $22.4\%$."
* *(Chỉ sang Hình 3 Ô `04B` và Bảng 1 Ô `05`)*:
  "Trên **98 lượt chạy thực nghiệm E3**, độ dốc thời gian đo được là **$\alpha = 2.11$** bám sát quy luật bậc hai lý thuyết ($D=6{,}000$ chậm gấp $35.6$ lần mà chỉ tăng $0.77\text{ pp}$). Đặc biệt ở chế độ ít mẫu $V = 1{,}600$ (Hình 3B & 3D), ngay tại ngưỡng nội suy **$D = 2{,}000 \approx V$**, chuẩn trọng số $\|\hat{\mathbf{W}}\|_F$ nổ vọt gấp **$16\text{ lần}$** ($155.0$) làm $A_{10}$ sập xuống $72.64\%$. Giải pháp **điều chuẩn thích nghi theo vết phổ $\gamma_{\text{auto}}$ (`FD2`)** của nhóm tự động định cỡ $\gamma$ ngay từ Task 1 không cần tập validation, đạt độ chính xác cao nhất toàn lưới ($88.19\%$ ở $D = 4{,}000$). Ngoài ra, nhóm còn chứng minh thêm điểm yếu thứ 3 về bão hòa $81\%$ hàm Sigmoid trên cả 3 bộ dữ liệu và trình bày đầy đủ tại Mục 16 `README.md`. Nhóm xin phép lắng nghe câu hỏi từ thầy/cô ạ!"

---

## 2. Bộ 12 Câu Hỏi Vấn Đáp Sâu & Đáp Án Định Lượng Ngắn Gọn

1. **Hỏi: Tại sao F-OAL lại dùng hàm mất mát bình phương tối thiểu (MSE) mà không dùng luôn Cross-Entropy giải tích?**
   - **Đáp:** Vì đạo hàm của Cross-Entropy chứa hàm phi tuyến Softmax $\mathbf{p}_i(\mathbf{W}) = \text{softmax}(\boldsymbol{\phi}_i \mathbf{W})$, khiến hệ phương trình đạo hàm bằng $0$ không có nghiệm đóng tuyến tính và không thể tách thành tổng tích ngoài $\sum \boldsymbol{\phi}_i^\top \boldsymbol{\phi}_i$ để cập nhật một lượt bằng bổ đề Woodbury. MSE đánh đổi tính tối ưu biên phân lớp lấy tính giải tích đệ quy chính xác một lượt.

2. **Hỏi: Trong `BalancedOnlineRidgeRLS` (`FD1`), việc nhân trọng số mẫu $\omega_i$ có làm mất tính tương đương với nghiệm đóng hay phụ thuộc thứ tự chia lô không?**
   - **Đáp:** Nếu trọng số mẫu $\omega_i$ cố định theo lớp, nghiệm đệ quy trên $(\sqrt{\omega_i}\boldsymbol{\phi}_i, \sqrt{\omega_i}\mathbf{y}_i)$ **trùng khít tuyệt đối** đến sai số `< 1e-11` với nghiệm đóng Weighted Ridge $(\mathbf{\Phi}^\top \mathbf{\Omega}\mathbf{\Phi} + \gamma \mathbf{I})^{-1}\mathbf{\Phi}^\top \mathbf{\Omega}\mathbf{Y}$ bất kể kích thước mini-batch (đã kiểm chứng trong `tests/test_critical_analysis.py`). Khi dùng bộ đếm trực tuyến nhân quả $\hat{N}_c^{(t)}$ không biết trước tương lai, tổng trọng số hiệu dụng của lớp $c$ tăng theo chuỗi điều hòa $\sum_{m=1}^{N_c} m^{-\beta}$, nén tỷ lệ lệch $100\times$ xuống mức rất nhỏ mà không cần lưu bất kỳ mẫu cũ nào.

3. **Hỏi: Tại sao trong Thí nghiệm E1, nhóm lại dùng chế độ `permuted_global` thay vì để các lớp đầu nhiều mẫu, các lớp cuối ít mẫu?**
   - **Đáp:** Nếu xếp các lớp ít mẫu dồn vào các task cuối, giám khảo sẽ không thể phân biệt được sự sụt giảm độ chính xác là do **mất cân bằng tần suất lớp** hay do **hiệu ứng thứ tự task / bão hòa dung lượng ở cuối chuỗi**. Việc phân bổ ngẫu nhiên có kiểm soát hạn ngạch mẫu đều khắp cả 10 tasks chứng minh rằng ngay cả một lớp học ở Task 10 vừa mới học xong vẫn sập nếu ít mẫu, còn lớp học từ Task 1 vẫn đạt $>91\%$ nếu giàu mẫu!

4. **Hỏi: Tại sao chỉ riêng bước chuẩn hóa cột `FD1b` ($\tilde{\mathbf{w}}_c = \hat{\mathbf{w}}_c / \|\hat{\mathbf{w}}_c\|_2^\tau$, không sửa quá trình train) đã hồi phục được `+28.5 pp` ở $\rho = 100$, nhưng kết hợp `FD1a+b` lại tăng tới `+42.7 pp`?**
   - **Đáp:** Vì mất cân bằng trong MSE gây ra **hai tác hại**: (1) làm co độ dài vector $\|\hat{\mathbf{w}}_c\|_2$, và (2) làm **xoay hướng vector $\hat{\mathbf{w}}_c$** lệch khỏi tâm lớp hiếm để trực giao hóa với đám mây mẫu khổng lồ của lớp đa số trên nón dương $(0,1)^D$. Chuẩn hóa biên độ `FD1b` chỉ sửa được độ dài (nên hồi phục `+28.5 pp`), còn cập nhật Woodbury có trọng số `FD1a` cân bằng lại cả ma trận tự tương quan $\tilde{\mathbf{G}}$ lẫn hướng của $\hat{\mathbf{w}}_c$, giúp tăng thêm `+14.2 pp` nữa (tổng `+42.7 pp`).

5. **Hỏi: Hãy chỉ chính xác dòng nào trong bài báo chứng minh Phương trình (14) của tác giả bị sai?**
   - **Đáp:** Trang 5, Phương trình (11) và (13): $\mathbf{R}_{k,i} = \mathbf{R}_{k,i-1} - \mathbf{R}_{k,i-1}\mathbf{X}_{k,i}^{(a)\top}(\mathbf{I}_S + \mathbf{X}_{k,i}^{(a)}\mathbf{R}_{k,i-1}\mathbf{X}_{k,i}^{(a)\top})^{-1}\mathbf{X}_{k,i}^{(a)}\mathbf{R}_{k,i-1}$. Trong toàn bộ vế phải chỉ xuất hiện $\mathbf{R}_{k,i-1} \in \mathbb{R}^{D \times D}$ và $\mathbf{X}_{k,i}^{(a)} \in \mathbb{R}^{S \times D}$, hoàn toàn **không có mặt ma trận nhãn $\mathbf{Y}$ hay số lớp $C$**. Do đó không thể có thừa số $C$ như $\mathcal{O}(D^2 C)$ ở Eq. (14) trang 6. Ngược lại, bước cập nhật $\hat{\mathbf{W}}$ ở Eq. (12) có phép nhân phần dư kích thước $(S \times D) \times (D \times C_s)$ tốn $\Theta(S D C_s)$ thì Eq. (15) lại bỏ sót $C_s$.

6. **Hỏi: Tại sao ở Hình 3B ($V = 1{,}600, \gamma = 10^{-3}$), độ chính xác ở $D = 4{,}000$ lại tốt hơn ở $D = 2{,}000$ dù số tham số nhiều gấp đôi?**
   - **Đáp:** Đó chính là hiện tượng **Double-Descent của ma trận ngẫu nhiên (Marchenko–Pastur)**: sự suy biến tồi tệ nhất xảy ra ngay tại ngưỡng nội suy $D / V \approx 1$ ($D = 2{,}000 \approx V = 1{,}600$), nơi trị riêng dương nhỏ nhất của $\mathbf{\Phi}^\top \mathbf{\Phi}$ tiến sát về $0$ làm nổ nghiệm tối thiểu chuẩn $\|\hat{\mathbf{W}}\|_F = 155.0$. Khi $D = 4{,}000 \gg V$, năng lượng giả nghịch đảo Moore–Penrose được phân tán đều ra không gian $4{,}000$ chiều nên chuẩn $\|\hat{\mathbf{W}}\|_F$ giảm một nửa về $77.8$, làm dịu bớt nhiễu nội suy — dù vẫn kém xa mức ổn định của giải pháp điều chuẩn thích nghi `FD2` ($\|\hat{\mathbf{W}}\|_F = 3.65$, $\bar{A} = 88.19\%$).

7. **Hỏi: Công thức $\gamma_{\text{auto}}$ (`FD2`) có vi phạm ràng buộc trực tuyến (online streaming) vì phải nhìn trước tổng số mẫu $V_{\text{est}}$ không?**
   - **Đáp:** Không vi phạm, vì $\gamma_{\text{auto}}$ chỉ đo vết năng lượng thực tế trên Task 1 ($\mathbf{\Phi}_1$), còn $V_{\text{est}} = T \cdot V_1$ chỉ là tích của cỡ mẫu Task 1 với số tác vụ dự kiến (hoặc đặt hằng số tham chiếu $V_{\text{ref}} = 1{,}000$). Thậm chí nhóm còn cài đặt thêm hàm `select_gcv_gamma` chọn $\gamma^*$ bằng **Generalized Cross-Validation (GCV)** thuần túy từ SVD của riêng Task 1 mà không cần bất kỳ tham số ước lượng nào!

8. **Hỏi: Tại sao bài báo viết khởi tạo ma trận chiếu từ *"normal distribution"* (trang 4) lại làm mất tới `16–19 pp` trên DTD và FGVC-Aircraft?**
   - **Đáp:** Vì đầu ra `LayerNorm` của ViT có chuẩn $\|\mathbf{h}\|_2 \approx \sqrt{768} \approx 27.7$. Nếu $P_{ab} \sim \mathcal{N}(0, 1)$, tiền kích hoạt $u_b = \mathbf{h}\mathbf{P}_{:,b}$ có độ lệch chuẩn đo được `~19.3–20.7`, đẩy `>81%` neuron rơi vào vùng bão hòa cứng $\sigma(u) \in \{0, 1\}$ có đạo hàm bằng $0$. Trong code gốc, tác giả dùng mặc định `nn.Linear(768, 1000)` của PyTorch có phương sai $\frac{1}{3 \times 768}$ nhỏ hơn $2{,}304$ lần mà không ghi vào bài báo. Nhóm trình bày chi tiết bằng chứng C1 trên cả 3 tập dữ liệu tại Mục 16 `README.md`.

9. **Hỏi: Tại sao các run của Lab 2 trên W&B lại tách thành 6 run đường cong thay vì trộn lẫn với các run của giai đoạn trước?**
   - **Đáp:** Tuân thủ đúng quy định của mục *Empirical Demonstration* trong Lab Book: dùng chung một project công khai `hcmus-foal-reproduction` nhưng gắn bộ tag riêng `["lab2-empirical-demo", "critical-analysis"]` và đặt tiền tố group `lab2_*` để người chấm lọc một lần là tách bạch hoàn toàn thực nghiệm phản biện của Lab 2 với thực nghiệm tái lập của Lab 1.

10. **Hỏi: Vì sao trên CIFAR-100 ở $\rho = 1$, $\bar{A}$ của Thực nghiệm Lab 2 là `91.85%` trong khi C3-D của Lab 1 là `90.59%`?**
    - **Đáp:** Rất minh bạch trong `provenance` của W&B config: Lab 2 tái sử dụng bộ cache đặc trưng GPU đã kiểm chứng `c2_feature_cache` (`augreg2_in21k_ft_in1k` với tiền xử lý `bicubic + normalize`, vốn đạt `91.77%` ở `seed=42` trong thực nghiệm C2-B1), trong khi C3-D dùng `bilinear + raw [0,1]` khớp code gốc tác giả (`90.40%` ở `seed=42`). Mọi phương pháp so sánh trong Lab 2 đều chạy trên cùng một ma trận đặc trưng đầu vào nên độ chênh lệch tương đối hoàn toàn khách quan và công bằng tuyệt đối.

11. **Hỏi: Nếu kết hợp đồng thời cả Long-Tail ($\rho \gg 1$) VÀ cỡ mẫu nhỏ ($V < D$) trên thực tế thì sao?**
    - **Đáp:** Hai giải pháp `FD1` và `FD2` hoàn toàn **trực giao và ghép nối trực tiếp được với nhau**: `FD2` xác định hệ số điều chuẩn khởi tạo $\mathbf{R}_0 = \gamma_{\text{auto}}^{-1}\mathbf{I}_D$ ở Task 1 để chặn suy biến phổ, sau đó `BalancedOnlineRidgeRLS` (`FD1`) thực hiện cập nhật Woodbury có trọng số tần suất trên nền $\mathbf{R}_0$ đó và chuẩn hóa cột lúc dự đoán.

12. **Hỏi: Hạn chế nào của F-OAL mà cả `FD1` lẫn `FD2` vẫn chưa thể giải quyết triệt để nếu giữ nguyên `Frozen ViT`?**
    - **Đáp:** Khi miền dữ liệu đầu vào hoàn toàn xa lạ với ImageNet-21K (ví dụ ảnh X-quang y tế, ảnh radar SAR hoặc nhiễu mờ cực nặng làm sập không gian đặc trưng $\phi(\mathbf{x})$ từ gốc), mọi bộ phân loại giải tích phía sau đều bị giới hạn bởi dung lượng thông tin của encoder đóng băng (Bất đẳng thức xử lý dữ liệu — *Data Processing Inequality*). Hướng mở rộng tất yếu khi đó là tinh chỉnh nhẹ các tham số `LayerNorm` hoặc gắn một *First-Task LoRA/Adapter* ở Task 1 trước khi đóng băng cho chuỗi RLS trực tuyến từ Task 2 trở đi.
