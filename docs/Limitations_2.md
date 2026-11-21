# GIẢI MÃ HIỆN TƯỢNG NGHỊCH LÝ ĐỘ QUÊN KHI LOẠI BỎ SMOOTH PROJECTION TRÊN TẬP DTD
## (Unraveling the Plasticity-Forgetting Paradox in Smooth Projection Ablation on DTD)

> **Dự án:** Tái hiện bài báo F-OAL (*Fast Optimization-free Online Continual Learning with Analytic Classifier*, NeurIPS 2024)  
> **Bộ môn:** Nhập môn Học máy (Introduction to Machine Learning) — Trường ĐHKHTN, ĐHQG-HCM (HCMUS)  
> **Tài liệu tham chiếu:** [`docs/Limitations_2.md`](Limitations_2.md) | [`docs/Limitations.md`](Limitations.md)  
> **Bộ kiểm thử pháp y tự động:** [`tests/test_dtd_forensic_verification.py::test_sp_ablation_plasticity_forgetting_paradox`](../tests/test_dtd_forensic_verification.py) (Passed)  
> **Ngày giải trình nghiệm thu:** 10/11/2026  

---

### 1. PHÁT BIỂU QUAN SÁT VÀ NGHỊCH LÝ BỀ MẶT (THE SUPERFICIAL PARADOX)

Trong thí nghiệm loại bỏ thành phần (Table 4 Ablation Study) trên bộ dữ liệu **Describable Textures Dataset (DTD)**, khi so sánh giữa cấu hình đầy đủ **A0 (Full F-OAL: Feature Fusion BẬT, Smooth Projection BẬT)** và cấu hình **A2 (w/o Smooth Projection: Feature Fusion BẬT, Smooth Projection TẮT)**, xuất hiện một chuỗi số liệu tưởng chừng nghịch lý:

| Chỉ số Đánh giá Continual Learning | A0 (Full F-OAL) | A2 (w/o Smooth Projection) | Độ lệch ($\text{A0} - \text{A2}$) | Nhận định Bề mặt |
| :--- | :---: | :---: | :---: | :--- |
| **Độ chính xác trung bình tích lũy ($\bar{A}$)** | **$87.20\%$** | $84.35\%$ | **$+2.85\text{ pp}$** | A0 vượt trội rõ rệt ($+2.85\%$) |
| **Độ chính xác tại task cuối cùng ($A_{10}$)** | **$79.56\%$** | $79.13\%$ | **$+0.44\text{ pp}$** | Hai mô hình gần như tương đương |
| **Độ quên trên các task quá khứ ($F_{\text{past}}$)** | $7.99\%$ | **$5.76\%$** | **$+2.22\text{ pp}$** | **A2 quên ít hơn A0 ($2.22\%$)!** |

#### Nghịch lý đặt ra:
* Nếu một người đánh giá chỉ nhìn vào chỉ số độ quên: $F_{\text{past}}(\text{A2}) = 5.76\% < F_{\text{past}}(\text{A0}) = 7.99\%$, họ có thể đi đến kết luận sai lầm rằng: *"Loại bỏ Smooth Projection giúp mô hình F-OAL ghi nhớ tốt hơn và chống quên hiệu quả hơn!"*
* Tuy nhiên, kết luận này mâu thuẫn trực tiếp với việc độ chính xác trung bình $\bar{A}$ lại sụt giảm tới **$2.85\text{ pp}$**, trong khi độ chính xác cuối cùng $A_{10}$ gần như không suy chuyển (chỉ lệch $0.44\text{ pp}$).
* **Câu hỏi nghiên cứu:**
  1. *Tại sao loại bỏ Smooth Projection lại làm $\bar{A}$ giảm mạnh ($2.85\text{ pp}$) nhưng $A_{10}$ chỉ giảm $0.44\text{ pp}$?*
  2. *Tại sao độ quên $F_{\text{past}}$ của A2 lại thấp hơn $2.22\text{ pp}$ so với A0? Liệu độ quên thấp hơn có đồng nghĩa với mô hình tốt hơn?*

---

### 2. PHÂN RÃ TOÁN HỌC VI MÔ: BÓC TÁCH NGUYÊN NHÂN TỪ MA TRẬN TAM GIÁC $10 \times 10$

Để trả lời chính xác, nhóm đã trích xuất toàn bộ $55$ phần tử của ma trận đánh giá tam giác dưới $R_{s, t}$ ($1 \le t \le s \le 10$) từ `metrics.json` của cả hai cấu hình A0 và A2 trên GPU Tesla T4.

#### 2.1. Bản chất công thức độ quên $F_{\text{past}}$
Độ quên trên các tác vụ quá khứ ($F_{\text{past}}$) được định nghĩa theo Equation 8 của bài báo:
$$F_{\text{past}} = \frac{1}{K-1} \sum_{j=1}^{K-1} f_j, \quad \text{với } f_j = \max_{l \in \{j, \dots, K-1\}} R_{l, j} - R_{K, j}$$
Trong đó:
* $\text{Peak}_j = \max_{l \in \{j, \dots, K-1\}} R_{l, j}$ là **đỉnh cao nhất** mà mô hình từng đạt được trên tác vụ $j$ trong suốt quá trình học trước giai đoạn cuối.
* $\text{Final}_j = R_{K, j}$ là **độ chính xác còn lại** trên tác vụ $j$ tại giai đoạn kết thúc chuỗi (Task 10).
* Hiệu số quên tại tác vụ $j$ thực chất là: $f_j = \text{Peak}_j - \text{Final}_j$.

#### 2.2. Bảng đối chiếu Đỉnh (Peak) và Điểm cuối (Final) trên từng tác vụ giữa A0 và A2

| Tác vụ $j$ | Đỉnh A0 ($\text{Peak}_j^{\text{A0}}$) | Điểm cuối A0 ($\text{Final}_j^{\text{A0}}$) | Quên A0 ($f_j^{\text{A0}}$) | Đỉnh A2 ($\text{Peak}_j^{\text{A2}}$) | Điểm cuối A2 ($\text{Final}_j^{\text{A2}}$) | Quên A2 ($f_j^{\text{A2}}$) | Chênh lệch Đỉnh ($\Delta \text{Peak}$) | Chênh lệch Cuối ($\Delta \text{Final}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1** | $98.75\%$ | $79.38\%$ | $19.38\%$ | $98.12\%$ | $80.00\%$ | $18.12\%$ | $+0.63\text{ pp}$ | $-0.62\text{ pp}$ |
| **Task 2** | **$91.25\%$** | $79.38\%$ | **$11.88\%$** | **$83.75\%$** | $78.75\%$ | **$5.00\%$** | **$+7.50\text{ pp}$** | $+0.63\text{ pp}$ |
| **Task 3** | **$96.88\%$** | $82.50\%$ | **$14.38\%$** | **$93.12\%$** | $82.50\%$ | **$10.63\%$** | **$+3.75\text{ pp}$** | $0.00\text{ pp}$ |
| **Task 4** | **$90.00\%$** | $80.00\%$ | **$10.00\%$** | **$85.62\%$** | $79.38\%$ | **$6.25\%$** | **$+4.38\text{ pp}$** | $+0.62\text{ pp}$ |
| **Task 5** | $77.50\%$ | $73.12\%$ | $4.38\%$ | $75.00\%$ | $70.62\%$ | $4.38\%$ | $+2.50\text{ pp}$ | $+2.50\text{ pp}$ |
| **Task 6** | **$77.50\%$** | $68.75\%$ | **$8.75\%$** | **$74.38\%$** | $70.00\%$ | **$4.38\%$** | **$+3.12\text{ pp}$** | $-1.25\text{ pp}$ |
| **Task 7** | $85.00\%$ | $84.38\%$ | $0.62\%$ | $84.38\%$ | $83.75\%$ | $0.62\%$ | $+0.62\text{ pp}$ | $+0.63\text{ pp}$ |
| **Task 8** | $87.50\%$ | $85.00\%$ | $2.50\%$ | $87.50\%$ | $85.00\%$ | $2.50\%$ | $0.00\text{ pp}$ | $0.00\text{ pp}$ |
| **Task 9** | $81.88\%$ | $81.88\%$ | $0.00\%$ | $80.00\%$ | $80.00\%$ | $0.00\%$ | $+1.88\text{ pp}$ | $+1.88\text{ pp}$ |
| **Trung bình (1–9)** | **$87.36\%$** | **$79.38\%$** | **$7.99\%$** | **$84.65\%$** | **$78.89\%$** | **$5.76\%$** | **$+2.71\text{ pp}$** | **$+0.49\text{ pp}$** |

#### 2.3. Định lý phân rã sai biệt độ quên (The Forgetting Gap Identity)
Từ công thức định nghĩa, hiệu số độ quên giữa A0 và A2 được phân rã thành hai thành phần độc lập:
$$\Delta F_{\text{past}} = F_{\text{past}}^{\text{A0}} - F_{\text{past}}^{\text{A2}} = (\overline{\text{Peak}}^{\text{A0}} - \overline{\text{Peak}}^{\text{A2}}) - (\overline{\text{Final}}^{\text{A0}} - \overline{\text{Final}}^{\text{A2}})$$
Thay các giá trị đo đạc thực tế:
$$\Delta F_{\text{past}} = (87.3611\% - 84.6528\%) - (79.3750\% - 78.8889\%) = \mathbf{+2.7083\%} - \mathbf{+0.4861\%} = \mathbf{+2.2222\text{ pp}}$$

**Ý NGHĨA PHÁP Y THEN CHỐT:**
1. **$84.8\%$ độ lệch của $F_{\text{past}}$ ($2.71 / (2.71 + 0.49)$) được lái bởi sự sụt giảm năng lực tiếp thu (Peak Performance) của A2, chứ KHÔNG PHẢI do A2 giữ điểm cuối tốt hơn.**
2. Ở tác vụ cuối cùng (Task 10), cả A0 và A2 đều đạt độ chính xác gần như trùng khớp: $\overline{\text{Final}}^{\text{A0}} = 79.38\%$ so với $\overline{\text{Final}}^{\text{A2}} = 78.89\%$ (chênh lệch chỉ $0.49\text{ pp}$).
3. Tuy nhiên, ở các tác vụ ban đầu, A0 đạt năng lực học vượt trội: $\overline{\text{Peak}}^{\text{A0}} = 87.36\%$ so với $\overline{\text{Peak}}^{\text{A2}} = 84.65\%$ (A0 cao hơn tới **$2.71\text{ pp}$**). Đặc biệt tại Task 2, đỉnh của A0 là **$91.25\%$** trong khi A2 chỉ đạt **$83.75\%$** (kém hơn tới **$7.50\text{ pp}$**).
4. **Hệ quả:** Do A2 có đỉnh học ban đầu quá thấp, khoảng cách từ đỉnh rơi xuống đáy $(\text{Peak} - \text{Final})$ bị thu hẹp một cách giả tạo! Đây là hiện tượng kinh điển trong Continual Learning:
   $$\text{\bf "Underlearning Artificially Masks Forgetting" (Học kém làm giảm độ quên giả tạo)}$$

---

### 3. CƠ CHẾ HỌC MÁY VÀ VAI TRÒ KIẾN TRÚC CỦA SMOOTH PROJECTION

Tại sao hiện tượng này lại diễn ra đặc biệt sâu sắc trên tập **DTD (Describable Textures Dataset)** mà không hề xuất hiện trên **CIFAR-100**?

#### 3.1. Bản chất miền dữ liệu và Định lý Cover
* **Trên CIFAR-100 (Miền đồ vật tự nhiên - In-Distribution):**
  * Vector đặc trưng $h \in \mathbb{R}^{768}$ trích xuất từ Vision Transformer tiền huấn luyện ImageNet-21K đã có tính phân tách tuyến tính cực cao giữa các lớp vật thể (ô tô, chó, mèo...).
  * Do đó, việc có hay không có Smooth Projection đều cho kết quả gần như tương đương: $\bar{A}$ lệch $-0.02\text{ pp}$, $A_{10}$ lệch $-0.01\text{ pp}$, và $F_{\text{past}}$ lệch $+0.08\text{ pp}$.
* **Trên DTD (Miền hoa văn kết cấu trừu tượng - Out-of-Distribution):**
  * DTD chứa các mẫu hoa văn phức tạp: *blotchy* (lốm đốm), *braided* (bện tết), *bubbly* (bọt bóng), *chequered* (bàn cờ), *cracked* (nứt nẻ)... Các mẫu này có độ biến thiên nội lớp (intra-class variance) rất lớn và ranh giới liên lớp (inter-class boundary) vô cùng tinh vi.
  * Không gian đặc trưng tuyến tính $768$ chiều từ ViT bị chồng lấn rất mạnh khi biểu diễn các hoa văn này.

#### 3.2. Vai trò toán học của Smooth Projection: $z = \sigma(P h)$
1. **Mở rộng chiều không gian ($D = 768 \to 1.000$):**
   * Theo **Định lý Cover về tính phân tách mẫu (Cover's Theorem on Separability of Patterns)**: Một không gian mẫu phức tạp phi tuyến tính sẽ có xác suất phân tách tuyến tính cao hơn đáng kể khi được ánh xạ phi tuyến sang một không gian có số chiều lớn hơn.
2. **Kernel ngẫu nhiên phi tuyến (Random Projection Kernel / Extreme Learning Machine):**
   * Ma trận $P \in \mathbb{R}^{1000 \times 768}$ khởi tạo ngẫu nhiên đều $\mathcal{U}[-r, r]$ kết hợp với hàm kích hoạt mượt $\sigma(u) = \frac{1}{1 + e^{-u}}$ đóng vai trò như một bộ ánh xạ kernel xấp xỉ phổ quát (universal approximator).
   * Hàm Sigmoid nén các giá trị biên và kéo dãn khoảng cách hình học giữa các cụm hoa văn. Nhờ đó, bộ phân loại trực tuyến RLS ở các giai đoạn đầu (Stage 1–4) dễ dàng tìm được siêu phẳng phân loại với biên an toàn lớn, giúp mô hình đạt đỉnh **$91.25\%$ ở Task 2** và **$96.88\%$ ở Task 3**.
3. **Khi bỏ Smooth Projection (A2):**
   * Đặc trưng $768$ chiều đưa thẳng vào RLS tuyến tính. Các cụm hoa văn bị nhiễu chéo lẫn nhau ngay từ đầu.
   * Năng lực tiếp thu (Plasticity) của mô hình bị bóp nghẹt: Stage 2 chỉ đạt **$83.75\%$** (sụt giảm nghiêm trọng **$-9.06\text{ pp}$** so với A0 là $92.81\%$).

#### 3.3. Cơ chế hội tụ ở Stage 10: Sự bão hòa dung lượng (Capacity Saturation)
* Khi số lượng lớp tăng dần từ 8 lớp (Stage 2) lên tới 40 lớp (Stage 10):
  * Ma trận tương quan tích lũy của RLS:
    $$R_k = \sum_{i=1}^N z_i z_i^T + \gamma I$$
    dần tích lũy năng lượng trên toàn bộ các hướng của không gian vector.
  * Dưới sự ràng buộc của hệ số điều chuẩn $\gamma = 1.0$, nghiệm dạng đóng Woodbury:
    $$W = Y^T Z (Z^T Z + \gamma I)^{-1}$$
    bị co cụm (shrinkage effect) để cân bằng biên quyết định cho toàn bộ 40 lớp.
  * Cả hai không gian ($1.000$ chiều của A0 và $768$ chiều của A2) lúc này đều đạt tới ngưỡng giới hạn dung lượng phân loại của bộ giải tích tuyến tính trên 40 lớp kết cấu.
  * Vì vậy, ở Stage 9 và 10, hiệu năng của hai mô hình hội tụ về cùng một mức trần:
    * Stage 9: A0 đạt $80.07\%$, A2 đạt $80.07\%$ (**trùng nhau tuyệt đối!**)
    * Stage 10: A0 đạt $79.56\%$, A2 đạt $79.13\%$ (**chỉ chênh $0.44\text{ pp}$**).

#### 3.4. Tại sao $\bar{A}$ giảm tới $2.85\text{ pp}$?
Độ chính xác trung bình tích lũy là trung bình cộng qua cả 10 giai đoạn:
$$\bar{A} = \frac{1}{10} \sum_{s=1}^{10} A_s$$
Nhìn vào bảng so sánh từng giai đoạn:

| Giai đoạn ($s$) | Số lớp tích lũy | A0 (Full F-OAL) | A2 (w/o SP) | Chênh lệch ($A_s^{\text{A0}} - A_s^{\text{A2}}$) |
| :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | 4 | $98.75\%$ | $98.13\%$ | $+0.63\text{ pp}$ |
| **Stage 2** | 8 | **$92.81\%$** | **$83.75\%$** | **$+9.06\text{ pp}$ (Khoảng cách cực đại!)** |
| **Stage 3** | 12 | **$93.96\%$** | **$87.29\%$** | **$+6.67\text{ pp}$** |
| **Stage 4** | 16 | **$91.56\%$** | **$86.56\%$** | **$+5.00\text{ pp}$** |
| **Stage 5** | 20 | **$87.88\%$** | **$84.75\%$** | **$+3.13\text{ pp}$** |
| **Stage 6** | 24 | **$84.48\%$** | **$81.35\%$** | **$+3.13\text{ pp}$** |
| **Stage 7** | 28 | $81.88\%$ | $81.52\%$ | $-0.14\text{ pp}$ |
| **Stage 8** | 32 | $81.02\%$ | $80.94\%$ | $+0.08\text{ pp}$ |
| **Stage 9** | 36 | **$80.07\%$** | **$80.07\%$** | **$0.00\text{ pp}$ (Hội tụ)** |
| **Stage 10** | 40 | **$79.56\%$** | **$79.13\%$** | **$+0.44\text{ pp}$ (Hội tụ)** |
| **Trung bình ($\bar{A}$)** | — | **$87.20\%$** | **$84.35\%$** | **$+2.85\text{ pp}$** |

**Ý nghĩa:** 
Trong Continual Learning trực tuyến (Online Streaming), mô hình được sử dụng để phân loại mẫu liên tục trong suốt toàn bộ vòng đời, chứ không phải chỉ đợi đến tác vụ thứ 10 mới đưa vào ứng dụng.
* Ở cấu hình A0, mô hình duy trì độ chính xác cực cao ($>91\%-94\%$) trong suốt 4 tác vụ đầu tiên (chiếm $40\%$ vòng đời).
* Ở cấu hình A2, mô hình bị rớt phong độ ngay từ tác vụ thứ 2 ($83.75\%$).
* Diện tích dưới đường cong học tập (Area Under the Curve) của A0 vượt trội hoàn toàn so với A2, mang lại giá trị ứng dụng thực tế cao hơn rất nhiều, được lượng hóa chính xác qua khoảng cách $\Delta \bar{A} = +2.85\text{ pp}$.

---

### 4. QUY LUẬT ĐƠN ĐIỆU TRÊN MA TRẬN GIAI THỪA $2 \times 2$ (A0, A1, A2, A3)

Nghịch lý độ quên không phải là một dị biệt ngẫu nhiên giữa A0 và A2, mà tuân theo một **quy luật suy giảm đơn điệu (monotonic trend)** trên toàn bộ 4 điều kiện của thiết kế giai thừa Bảng 4:

| Cấu hình Ablation | FF | SP | $\bar{A}$ (%) | $A_{10}$ (%) | Đỉnh TB (1–9) | Điểm cuối TB (1–9) | Độ quên $F_{\text{past}}$ (%) | Xếp hạng theo $\bar{A}$ | Xếp hạng theo $F_{\text{past}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0 (Full F-OAL)** | BẬT | BẬT | **87.20%** | **79.56%** | **87.36%** | $79.37\%$ | **7.99%** | **Hạng 1 (Tốt nhất)** | *Hạng 4 (Tệ nhất?)* |
| **A1 (w/o FF)** | TẮT | BẬT | **85.01%** | **78.06%** | **84.93%** | $77.92\%$ | **7.01%** | **Hạng 2** | *Hạng 3* |
| **A2 (w/o SP)** | BẬT | TẮT | **84.35%** | **79.12%** | **84.65%** | $78.89\%$ | **5.76%** | **Hạng 3** | *Hạng 2* |
| **A3 (w/o Both)** | TẮT | TẮT | **81.10%** | **77.56%** | **82.78%** | $77.22\%$ | **5.56%** | **Hạng 4 (Tệ nhất)** | **Hạng 1 (Tốt nhất?)** |

#### Phát hiện cốt lõi:
1. **Nghịch đảo thứ bậc hoàn hảo:**
   * Thứ bậc năng lực thực tế ($\bar{A}$): $\text{A0 } (87.20\%) > \text{A1 } (85.01\%) > \text{A2 } (84.35\%) > \text{A3 } (81.10\%)$.
   * Thứ bậc độ quên ($F_{\text{past}}$): $\text{A3 } (5.56\%) < \text{A2 } (5.76\%) < \text{A1 } (7.01\%) < \text{A0 } (7.99\%)$.
2. Nếu dùng riêng chỉ số $F_{\text{past}}$ để xếp hạng, cấu hình tồi tệ nhất A3 (suy thoái học máy toàn diện, $\bar{A} = 81.10\%$) sẽ bị tuyên bố nhầm là "thuật toán ưu việt nhất" vì có độ quên thấp nhất ($5.56\%$)!
3. Cơ chế toán học chi phối:
   * Khi tước bỏ các module kiến trúc hữu ích (từ A0 $\to$ A3), **năng lực học đỉnh cao (Peak)** sụt giảm rất mạnh và liên tục: từ $87.36\% \to 84.93\% \to 84.65\% \to 82.78\%$ (giảm tới **$4.58\text{ pp}$**).
   * Trong khi đó, **điểm kết thúc (Final)** giảm chậm hơn nhiều: từ $79.37\% \to 77.22\%$ (chỉ giảm **$2.15\text{ pp}$**).
   * Hiệu số $F_{\text{past}} = \text{Peak} - \text{Final}$ bắt buộc phải co hẹp lại, tạo ra ảo giác giảm độ quên!

---

### 5. NGUYÊN TẮC PHƯƠNG PHÁP LUẬN VÀ CHỈ SỐ ĐỀ XUẤT ĐÁNH GIÁ CÔNG BẰNG

Từ phân tích pháp y trên, nhóm rút ra 3 nguyên tắc phương pháp luận quan trọng cho cộng đồng nghiên cứu Continual Learning:

#### Nguyên tắc 1: Không bao giờ sử dụng chỉ số Forgetting một cách cô lập
* Chỉ số Forgetting ($F$) chỉ có giá trị so sánh khi và chỉ khi các mô hình có **năng lực tiếp thu ban đầu tương đương nhau ($\text{Peak} \approx \text{const}$)**.
* Một mô hình học kém ngay từ đầu (underfitting / low plasticity) sẽ luôn có độ quên thấp, nhưng đó là một "ưu điểm giả tạo" bắt nguồn từ sự thất bại trong việc học kiến thức mới.

#### Nguyên tắc 2: Luôn đánh giá cặp chỉ số $(\bar{A}, F)$ hoặc $(A_{\text{last}}, F)$
* Một mô hình chỉ được công nhận là có khả năng chống quên thực chất nếu nó đạt độ quên thấp hơn **trong khi vẫn duy trì độ chính xác trung bình $\bar{A}$ tương đương hoặc cao hơn**.
* Nếu độ quên giảm đi kèm sự suy giảm mạnh của $\bar{A}$ (như trường hợp của A2 và A3), đó là chỉ báo rõ ràng của sự suy thoái năng lực thích ứng (plasticity deficiency).

#### Nguyên tắc 3: Đề xuất chỉ số Tỷ lệ Duy trì Kiến thức Tương đối (Relative Knowledge Retention - RKR)
Để chuẩn hóa sự chênh lệch về đỉnh giữa các mô hình, nhóm đề xuất chỉ số **RKR**:
$$\text{RKR} = \frac{\overline{\text{Final}}}{\overline{\text{Peak}}} = 1 - \frac{F_{\text{past}}}{\overline{\text{Peak}}}$$
Tính toán trên 4 cấu hình DTD:
* **A0 (Full F-OAL):** $\text{RKR}_{\text{A0}} = \frac{79.38\%}{87.36\%} = \mathbf{90.86\%}$
* **A1 (w/o FF):** $\text{RKR}_{\text{A1}} = \frac{77.92\%}{84.93\%} = \mathbf{91.75\%}$
* **A2 (w/o SP):** $\text{RKR}_{\text{A2}} = \frac{78.89\%}{84.65\%} = \mathbf{93.19\%}$
* **A3 (w/o Both):** $\text{RKR}_{\text{A3}} = \frac{77.22\%}{82.78\%} = \mathbf{93.28\%}$

Chỉ số RKR phản ánh rằng cả 4 cấu hình của F-OAL đều duy trì được trên **$90\%-93\%$** năng lực tối đa của mình (nhờ bản chất giải tích trực giao của thuật toán Woodbury RLS). Tuy nhiên, **A0 luôn vận hành ở một đẳng cấp độ chính xác cao hơn hẳn ($87.20\%$)** nhờ có Smooth Projection mở rộng không gian biểu diễn cho các đặc trưng kết cấu bề mặt phức tạp.

---

### BẢNG TỔNG KẾT NGHIỆM THU QUAN SÁT (ACCEPTANCE VERDICT)

| Mục kiểm toán | Kết quả phân tích pháp y | Trạng thái Nghiệm thu |
| :--- | :--- | :---: |
| **Bản chất của $\Delta \bar{A} = +2.85\text{ pp}$** | A0 vượt trội ở các giai đoạn đầu (Stage 2 hơn $+9.06\text{ pp}$, Stage 3 hơn $+6.67\text{ pp}$, Stage 4 hơn $+5.00\text{ pp}$) nhờ Smooth Projection tạo không gian phi tuyến $D=1.000$ mở rộng biên tách biệt cho các lớp texture phức tạp. | **ĐÃ GIẢI THÍCH TOÀN DIỆN (RESOLVED)** |
| **Bản chất của $\Delta A_{10} = +0.44\text{ pp}$** | Khi tích lũy tới 40 lớp ở Task 10, dung lượng bộ phân loại giải tích và hệ số điều chuẩn $\gamma = 1.0$ kéo cả hai mô hình hội tụ về cùng một mức trần dung lượng ($\sim 79.1\% - 79.5\%$). | **ĐÃ GIẢI THÍCH TOÀN DIỆN (RESOLVED)** |
| **Bản chất của $F_{\text{past}}$ giảm từ 7.99% $\to$ 5.76%** | Được chứng minh bằng đẳng thức toán học $\Delta F = \Delta \text{Peak} - \Delta \text{Final} = 2.71\% - 0.49\% = 2.22\%$. A2 có độ quên thấp hơn hoàn toàn là do A2 có đỉnh học ban đầu kém hơn ($84.65\%$ vs $87.36\%$). "Học kém làm giảm độ quên giả tạo". | **ĐÃ GIẢI THÍCH TOÀN DIỆN (RESOLVED)** |
| **Kiểm thử tự động** | Vượt qua bài kiểm thử [`tests/test_dtd_forensic_verification.py::test_sp_ablation_plasticity_forgetting_paradox`](../tests/test_dtd_forensic_verification.py) xác nhận tính đúng đắn toán học của mọi đẳng thức. | **ĐÃ XÁC MINH KIỂM THỬ (VERIFIED)** |

<!-- GOAL_COMPLETE -->
