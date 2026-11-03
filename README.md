# PHÂN TÍCH PAPER F-OAL: FORWARD-ONLY ONLINE ANALYTIC LEARNING

## 1. Thông tin tổng quan

**Tên paper:** F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning.

**Tác giả:** Huiping Zhuang, Yuchen Liu, Run He, Kai Tong, Ziqian Zeng, Cen Chen, Yi Wang và Lap-Pui Chau.

**Năm công bố:** 2024.

**Hội nghị:** Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Main Conference Track.

**Xếp hạng hội nghị:** A* theo ICORE2026.

**Lĩnh vực:** Machine Learning, Continual Learning, Online Class-Incremental Learning, Computer Vision.

**Paper chính thức:** [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/48ffa38c13078d6ce26b328e7f373243-Abstract-Conference.html).

**Mã nguồn:** [GitHub — F-OAL](https://github.com/liuyuchen-cz/F-OAL).

### 1.1. Tóm tắt nghiên cứu

F-OAL là một phương pháp **Online Class-Incremental Learning (OCIL)** được đề xuất nhằm giải quyết hiện tượng **Catastrophic Forgetting**, đồng thời giảm thời gian huấn luyện và chi phí bộ nhớ GPU.

Khác với những phương pháp Deep Learning truyền thống dựa trên Backpropagation, F-OAL sử dụng một **Vision Transformer (ViT) đã được pre-train và đóng băng**, kết hợp với **Feature Fusion**, **Smooth Projection** và **Analytic Classifier**. Bộ phân loại được cập nhật bằng **Recursive Least Squares (RLS)**, không cần tính gradient hay lưu lại các mẫu huấn luyện cũ để replay.

Ý tưởng cốt lõi của nghiên cứu là chuyển việc huấn luyện bộ phân loại từ một bài toán tối ưu lặp bằng Gradient Descent sang một bài toán **Ridge Regression có nghiệm giải tích**. Nhờ khả năng cập nhật nghiệm theo từng mini-batch, F-OAL có thể tiếp nhận dữ liệu mới liên tục mà không cần truy cập lại toàn bộ dữ liệu trước đó.

Kết quả thực nghiệm trên sáu bộ dữ liệu cho thấy F-OAL đạt độ chính xác cạnh tranh với nhiều phương pháp hiện có, đặc biệt nổi bật trên các bộ dữ liệu fine-grained, đồng thời có thời gian huấn luyện ngắn và mức sử dụng GPU tương đối thấp.

---

## 2. Bối cảnh và bài toán nghiên cứu

### 2.1. Class-Incremental Learning

Trong Machine Learning truyền thống, mô hình thường được huấn luyện trên một tập dữ liệu cố định, chứa đầy đủ các lớp cần dự đoán. Tuy nhiên, trong nhiều ứng dụng thực tế, các lớp dữ liệu mới có thể xuất hiện sau khi mô hình đã được triển khai.

**Class-Incremental Learning (CIL)** giải quyết tình huống mô hình phải liên tục học thêm các lớp mới mà vẫn duy trì khả năng nhận biết những lớp đã học.

Ví dụ, ở giai đoạn đầu, một mô hình phân loại hình ảnh được huấn luyện trên 10 lớp đầu tiên của CIFAR-100. Ở giai đoạn tiếp theo, mô hình được cung cấp dữ liệu của 10 lớp mới. Sau khi hoàn thành giai đoạn thứ hai, mô hình phải phân loại được cả 20 lớp thay vì chỉ 10 lớp gần nhất.

Quá trình này tiếp tục cho đến khi mô hình học đủ 100 lớp.

Khó khăn lớn nhất của CIL là **Catastrophic Forgetting**: khi học dữ liệu mới, những thay đổi trong tham số mô hình có thể làm suy giảm hiệu suất dự đoán trên các lớp cũ.

### 2.2. Online Class-Incremental Learning

**Online Class-Incremental Learning (OCIL)** bổ sung thêm ràng buộc về cách dữ liệu được cung cấp. Thay vì có toàn bộ dữ liệu của một task để huấn luyện qua nhiều epoch, mô hình tiếp nhận dữ liệu dưới dạng các mini-batch liên tiếp và thông thường chỉ được học trên mỗi mẫu một lần.

Ta có thể biểu diễn luồng huấn luyện như sau:

$$
\mathcal D_1\rightarrow\mathcal D_2
\rightarrow\cdots\rightarrow\mathcal D_t
$$

Với mỗi mini-batch:

$$
\mathcal D_t=(X_t,Y_t)
$$

Trong đó, $X_t$ là dữ liệu đầu vào và $Y_t$ là nhãn tương ứng.

Khi mini-batch mới xuất hiện, mô hình không được truy cập lại dữ liệu của những mini-batch đã xử lý trước đó.

Cần lưu ý rằng **Online Learning không đồng nghĩa với Unsupervised Learning**. Trong thiết lập OCIL của F-OAL, dữ liệu huấn luyện vẫn có nhãn. Từ *online* nhấn mạnh tính tuần tự và giới hạn số lần truy cập dữ liệu.

### 2.3. Những hạn chế của các phương pháp trước đây

Một hướng phổ biến để giảm Catastrophic Forgetting là **Experience Replay**, trong đó hệ thống lưu một tập nhỏ các mẫu dữ liệu cũ và sử dụng lại chúng khi học task mới. Phương pháp này có thể duy trì hiệu suất tốt nhưng cần bộ nhớ bổ sung, tăng chi phí huấn luyện và có thể gây ra vấn đề riêng tư khi dữ liệu cũ phải được lưu trữ.

Các phương pháp **Exemplar-Free** không yêu cầu lưu dữ liệu cũ để replay, nhưng thường gặp khó khăn trong việc cân bằng khả năng học lớp mới và duy trì độ chính xác trên lớp cũ.

Một hướng tiếp cận khác là **Analytic Continual Learning (ACL)**, sử dụng các nghiệm giải tích thay cho quá trình tối ưu bằng Backpropagation. Tuy nhiên, những phương pháp ACL trước đó thường cần giai đoạn huấn luyện nền hoặc xử lý dữ liệu theo từng task lớn, gây khó khăn khi áp dụng trực tiếp vào luồng dữ liệu online.

F-OAL được đề xuất nhằm khắc phục các hạn chế trên bằng cách xây dựng một phương pháp **Exemplar-Free, Forward-Only và cập nhật trực tiếp theo mini-batch**.

---

## 3. Mục tiêu và đóng góp của F-OAL

F-OAL hướng đến ba mục tiêu chính: duy trì độ chính xác khi học thêm lớp mới, giảm thời gian huấn luyện và giảm mức tiêu thụ bộ nhớ GPU.

Đóng góp đầu tiên là mở rộng **Analytic Continual Learning** sang bối cảnh **Online Class-Incremental Learning**, trong đó dữ liệu xuất hiện tuần tự và chỉ được xử lý một lần.

Đóng góp thứ hai là sử dụng một **Frozen Pre-trained Encoder** kết hợp **Feature Fusion** và **Smooth Projection** nhằm tạo ra biểu diễn đặc trưng có chất lượng mà không cần Fine-tuning toàn bộ mạng.

Đóng góp thứ ba là sử dụng **Recursive Least Squares** để cập nhật Analytic Classifier theo từng mini-batch. Cơ chế này giúp duy trì nghiệm tối ưu của Ridge Regression trên toàn bộ dữ liệu đã quan sát mà không cần lưu lại các mẫu gốc.

Cuối cùng, nghiên cứu cung cấp kết quả thực nghiệm nhằm đánh giá độ chính xác, mức độ quên, thời gian huấn luyện và bộ nhớ GPU so với những phương pháp CIL/OCIL trước đó.

---

## 4. Kiến trúc và phương pháp đề xuất

Kiến trúc F-OAL gồm hai thành phần chính: **Frozen Encoder** và **Analytic Classifier**.

Dữ liệu đầu vào được xử lý bởi Vision Transformer. Những đặc trưng trung gian từ các transformer block được kết hợp bằng Feature Fusion, sau đó biến đổi qua Smooth Projection. Vector đặc trưng cuối cùng được đưa vào Analytic Classifier để dự đoán lớp.

### 4.1. Frozen Pre-trained Encoder

F-OAL sử dụng **ViT-B được pre-train trên ImageNet-1K** làm backbone. Toàn bộ tham số của bộ trích xuất đặc trưng được giữ cố định trong quá trình học tăng dần.

Với ảnh đầu vào $x$, encoder tạo ra vector đặc trưng:

$$
h=\phi(x)
$$

Trong đó, $\phi$ là hàm trích xuất đặc trưng cố định.

Việc đóng băng encoder giúp giảm nhu cầu lưu các activation phục vụ Backpropagation và loại bỏ chi phí tính gradient cho backbone trong giai đoạn học tăng dần.

Ngoài ra, vì tham số encoder không thay đổi, biểu diễn của một ảnh cũ cũng không bị thay đổi do quá trình Fine-tuning trên dữ liệu mới.

Tuy nhiên, điều này không có nghĩa mô hình hoàn toàn không thể quên. Bộ phân loại vẫn thay đổi, và độ chính xác trên các lớp cũ vẫn có thể suy giảm khi bổ sung lớp mới.

### 4.2. Feature Fusion

Thay vì chỉ sử dụng đặc trưng từ transformer block cuối cùng, F-OAL khai thác các đặc trưng trung gian từ nhiều block của ViT.

Giả sử ViT có $L$ transformer block, với đặc trưng sau block thứ $i$ là $h_i$. Feature Fusion có thể được biểu diễn bằng:

$$
h_{\text{fusion}}
=
\frac{1}{L}\sum_{i=1}^{L}h_i
$$

Mục tiêu của cơ chế này là kết hợp thông tin từ các mức biểu diễn khác nhau. Những transformer block trung gian có thể chứa thông tin bổ sung cho đặc trưng cuối cùng, qua đó cải thiện khả năng phân biệt giữa các lớp.

Feature Fusion đặc biệt có ý nghĩa với các bộ dữ liệu **Fine-Grained Classification**, nơi nhiều lớp có đặc điểm thị giác tương đối giống nhau.

### 4.3. Smooth Projection

Sau Feature Fusion, đặc trưng được đưa qua một phép chiếu tuyến tính ngẫu nhiên, kết hợp với hàm kích hoạt Sigmoid:

$$
z=\sigma(h_{\text{fusion}}P)
$$

Trong đó:

- $P$ là ma trận chiếu ngẫu nhiên.
- $\sigma$ là hàm Sigmoid.
- $z$ là vector đặc trưng cuối cùng để huấn luyện bộ phân loại.

**Điểm quan trọng là $P$ chỉ được khởi tạo một lần, sau đó giữ cố định trong toàn bộ quá trình học.** Các phần tử của $P$ được lấy mẫu từ một phân phối chuẩn.

Vì vậy, Smooth Projection không được tối ưu bằng Backpropagation. Nó đóng vai trò biến đổi đặc trưng sang một không gian mới, có thêm tính phi tuyến nhờ hàm Sigmoid.

Trong thực nghiệm chính, F-OAL sử dụng **1.000 chiều** cho đặc trưng đầu ra của phép chiếu.

Việc lựa chọn phép chiếu cố định giúp bảo đảm không gian đặc trưng không thay đổi giữa các task, một điều kiện quan trọng để duy trì nghiệm Recursive Least Squares trên dữ liệu tích lũy.

### 4.4. Analytic Classifier

**Analytic Classifier (AC)** là bộ phân loại tuyến tính học ánh xạ từ đặc trưng sang nhãn One-Hot.

Giả sử có $N$ mẫu, đặc trưng có $D$ chiều và tổng cộng $C$ lớp. Ta ký hiệu:

$$
H\in\mathbb R^{N\times D}
$$

$$
Y\in\mathbb R^{N\times C}
$$

$$
W\in\mathbb R^{D\times C}
$$

Trong đó, $H$ là ma trận đặc trưng, $Y$ là ma trận nhãn One-Hot và $W$ là trọng số của bộ phân loại.

Đầu ra dự đoán được tính bằng:

$$
\hat Y=HW
$$

Không giống bộ phân loại Neural Network được huấn luyện bằng Cross-Entropy và Backpropagation, Analytic Classifier trong F-OAL giải bài toán **Ridge Regression**:

$$
\boxed{
W^*=
\arg\min_W
\left(
\|HW-Y\|_F^2+\gamma\|W\|_F^2
\right)
}
$$

Trong đó, $\|\cdot\|_F$ là Frobenius Norm và $\gamma$ là hệ số L2 Regularization.

Nghiệm tối ưu có dạng:

$$
\boxed{
W^*=(H^TH+\gamma I)^{-1}H^TY
}
$$

Công thức này có thể được suy ra bằng cách lấy đạo hàm của hàm mục tiêu theo $W$ rồi cho đạo hàm bằng 0.

Đây là **Closed-Form Solution**, cho phép xác định nghiệm tối ưu của hàm mục tiêu Least Squares mà không cần thực hiện nhiều bước Gradient Descent.

Tuy nhiên, nếu tính trực tiếp công thức này, hệ thống cần sử dụng toàn bộ dữ liệu đã quan sát. Vì vậy, F-OAL áp dụng Recursive Least Squares để cập nhật nghiệm theo mini-batch.

---

## 5. Recursive Least Squares (RLS)

### 5.1. Ý tưởng cốt lõi

**Recursive Least Squares** cho phép cập nhật nghiệm Ridge Regression khi có dữ liệu mới mà không cần giải lại toàn bộ bài toán từ đầu.

Giả sử tại thời điểm $t$, mô hình đã tiếp nhận các mini-batch:

$$
(H_1,Y_1),\ldots,(H_t,Y_t)
$$

Nghiệm Ridge Regression trên toàn bộ dữ liệu đã quan sát là:

$$
W_t^*=
\left(
\gamma I+\sum_{i=1}^{t}H_i^TH_i
\right)^{-1}
\left(
\sum_{i=1}^{t}H_i^TY_i
\right)
$$

Ta định nghĩa ma trận:

$$
\boxed{
R_t=
\left(
\gamma I+\sum_{i=1}^{t}H_i^TH_i
\right)^{-1}
}
$$

Ma trận $R_t$ lưu thông tin thống kê tích lũy từ các đặc trưng đã quan sát dưới dạng nghịch đảo của ma trận Gram có regularization.

Điều cần phân biệt là **$R_t$ không lưu toàn bộ thông tin nhãn**. Quan hệ giữa đặc trưng và nhãn được duy trì kết hợp thông qua trọng số bộ phân loại $W_t$.

### 5.2. Công thức cập nhật RLS

Khi mini-batch mới $(H_t,Y_t)$ xuất hiện, F-OAL sử dụng hai công thức cập nhật.

Đầu tiên là cập nhật $R_t$:

$$
\boxed{
R_t=R_{t-1}
-R_{t-1}H_t^T
(I+H_tR_{t-1}H_t^T)^{-1}
H_tR_{t-1}
}
$$

Công thức này được suy ra từ **Woodbury Matrix Identity**, cho phép cập nhật nghịch đảo ma trận khi có thêm dữ liệu.

Sau đó, trọng số bộ phân loại được cập nhật:

$$
\boxed{
W_t=W_{t-1}
+R_tH_t^T(Y_t-H_tW_{t-1})
}
$$

Trong công thức thứ hai, $W_{t-1}$ là trọng số cũ, $H_tW_{t-1}$ là dự đoán trên mini-batch mới và $Y_t-H_tW_{t-1}$ là sai số dự đoán.

Khác với Gradient Descent, phép điều chỉnh này không sử dụng learning rate nhân với gradient của hàm Loss. Thay vào đó, hệ số cập nhật được xác định từ ma trận $R_t$, nhằm duy trì nghiệm Least Squares trên dữ liệu tích lũy.

Với hệ số regularization $\gamma>0$, trạng thái có thể được khởi tạo:

$$
R_0=\gamma^{-1}I,\qquad W_0=0
$$

Trong cấu hình chính của F-OAL, tác giả sử dụng $\gamma=1$, nên $R_0=I$.

### 5.3. Ví dụ tính toán

Xét một bài toán hồi quy đơn giản với một đặc trưng, một đầu ra và hệ số regularization $\gamma=1$.

Khởi tạo:

$$
R_0=1,\qquad W_0=0
$$

Mini-batch đầu tiên có:

$$
H_1=1,\qquad Y_1=1
$$

Khi đó:

$$
R_1=1-\frac{1}{1+1}=\frac12
$$

Và:

$$
W_1=0+\frac12(1-0)=\frac12
$$

Tiếp theo, mini-batch thứ hai có:

$$
H_2=1,\qquad Y_2=0
$$

Ta cập nhật:

$$
R_2=\frac12-\frac{(1/2)^2}{1+1/2}
=\frac13
$$

$$
W_2=\frac12+\frac13\left(0-\frac12\right)
=\frac13
$$

Nếu gom cả hai mẫu lại và giải Ridge Regression trực tiếp, ta nhận được:

$$
W^*=
\frac{1\cdot1+1\cdot0}{1^2+1^2+1}
=\frac13
$$

Như vậy, nghiệm được cập nhật tuần tự bằng RLS bằng với nghiệm tính từ toàn bộ dữ liệu.

**Ý nghĩa:** RLS duy trì nghiệm tối ưu của bài toán Ridge Regression mà không cần lưu lại các mẫu đã xử lý, miễn là không gian đặc trưng và hàm mục tiêu được giữ cố định. Ví dụ này minh họa cơ chế cập nhật ở dạng đơn giản, không phải toàn bộ bài toán phân loại nhiều lớp của F-OAL.

### 5.4. Học thêm lớp mới

Khi một task mới xuất hiện, số lượng lớp tăng lên. F-OAL mở rộng ma trận trọng số $W$ bằng cách thêm các cột mới được khởi tạo bằng 0.

Ví dụ, nếu ban đầu có ba lớp:

$$
W\in\mathbb R^{1000\times3}
$$

Và task tiếp theo bổ sung hai lớp, trọng số được mở rộng thành:

$$
W'\in\mathbb R^{1000\times5}
$$

Các cột cũ được giữ lại, còn hai cột mới được khởi tạo bằng 0 trước khi cập nhật RLS.

Ma trận $R$ không cần mở rộng vì số chiều đặc trưng vẫn là 1.000:

$$
R\in\mathbb R^{1000\times1000}
$$

Cơ chế này cho phép F-OAL học thêm lớp mới mà không phải huấn luyện lại toàn bộ mô hình.

### 5.5. Vì sao RLS giúp hạn chế Catastrophic Forgetting?

Thay vì chỉ tối ưu trên mini-batch gần nhất, RLS duy trì nghiệm của hàm mục tiêu:

$$
\boxed{
W_t^*=
\arg\min_W
\left[
\sum_{i=1}^{t}\|H_iW-Y_i\|_F^2
+\gamma\|W\|_F^2
\right]
}
$$

Hàm mục tiêu này tính đến tất cả mini-batch đã xuất hiện.

Vì thế, dưới giả định đặc trưng được giữ cố định và các phép toán được thực hiện chính xác, cập nhật RLS cho nghiệm tương đương với Ridge Regression huấn luyện trên toàn bộ dữ liệu tích lũy.

Tuy nhiên, **tính tương đương về nghiệm tối ưu không bảo đảm độ chính xác trên lớp cũ sẽ không suy giảm**. Các lớp mới vẫn có thể gây nhầm lẫn với lớp cũ, đặc biệt khi đặc trưng trích xuất chưa đủ khả năng phân biệt chúng.

---

## 6. Phân tích thời gian và bộ nhớ

Ưu điểm quan trọng của F-OAL là loại bỏ Backpropagation trong quá trình học tăng dần.

Đối với mạng Neural Network thông thường, quá trình huấn luyện cần lưu những giá trị trung gian để tính gradient và cập nhật tham số. Với F-OAL, encoder được đóng băng và quá trình huấn luyện chỉ cần thực hiện Forward Pass, sau đó cập nhật các ma trận của Analytic Classifier.

Nếu $D$ là số chiều đặc trưng và $C$ là số lớp, hai ma trận trạng thái chính có kích thước:

$$
R\in\mathbb R^{D\times D},
\qquad
W\in\mathbb R^{D\times C}
$$

Vì vậy, lượng bộ nhớ bổ sung dành cho các ma trận này có độ phức tạp:

$$
O(D^2+DC)
$$

Trong cấu hình $D=1000$, ma trận $R$ chứa một triệu phần tử, tương đương khoảng 4 MB nếu lưu bằng float32. Tuy nhiên, đây chỉ là bộ nhớ của riêng $R$, chưa bao gồm ViT, các activation phục vụ Forward Pass và những tensor khác.

Với mini-batch có kích thước $B$, phép cập nhật RLS dựa trên Woodbury cần các phép nhân ma trận với chi phí chính thường ở mức $O(BD^2+B^3)$, cùng những phép toán liên quan đến bộ phân loại. Khi $B$ nhỏ, việc cập nhật có thể hiệu quả hơn so với giải lại nghiệm trên toàn bộ dữ liệu.

Một giới hạn quan trọng là **bộ nhớ của $R$ tăng theo bình phương số chiều đặc trưng**. Nếu tăng $D$ quá lớn, lợi thế bộ nhớ có thể giảm đáng kể.

---

## 7. Thực nghiệm và kết quả

### 7.1. Thiết lập thực nghiệm

F-OAL được đánh giá trên sáu bộ dữ liệu gồm CIFAR-100, CORe50, FGVCAircraft, DTD, Tiny-ImageNet và Country211.

Các phương pháp so sánh sử dụng backbone ViT-B đã pre-train trên ImageNet-1K. Đối với F-OAL, cấu hình chính sử dụng batch size 10, số chiều đặc trưng sau phép chiếu là 1.000 và hệ số regularization $\gamma=1$.

Tác giả thực hiện các thí nghiệm trên một GPU **NVIDIA RTX 4070 Ti 12 GB**, và báo cáo kết quả trung bình của ba lần chạy.

Các phương pháp được đánh giá theo ba chỉ số chính:

**Average Accuracy ($A_{\text{avg}}$)** phản ánh độ chính xác trung bình trong quá trình học tăng dần.

**Last Accuracy ($A_{\text{last}}$)** phản ánh độ chính xác sau khi hoàn thành task cuối cùng.

**Forgetting Rate ($F$)** đo mức suy giảm độ chính xác trên các task cũ so với kết quả tốt nhất đã từng đạt được trước đó.

Hai chỉ số Accuracy càng cao càng tốt, trong khi Forgetting Rate càng thấp càng tốt.

### 7.2. Kết quả độ chính xác

Bảng sau trình bày kết quả Average Accuracy từ Table 2 của paper:

| Dataset | F-OAL | Baseline tốt nhất ngoài F-OAL |
|---|---:|---:|
| CIFAR-100 | 91,1% | 92,4% — DVC |
| CORe50 | 96,3% | 97,1% — DVC |
| FGVCAircraft | 62,2% | 55,6% — SCR |
| DTD | 82,8% | 76,0% — EASE |
| Tiny-ImageNet | 91,2% | 92,0% — EASE |
| Country211 | 24,4% | 17,8% — SLCA |

Kết quả cho thấy F-OAL đặc biệt nổi bật trên FGVCAircraft, DTD và Country211. Trên FGVCAircraft, phương pháp đạt 62,2%, cao hơn SCR 6,6 điểm phần trăm. Trên DTD, F-OAL vượt EASE 6,8 điểm phần trăm.

Tuy nhiên, F-OAL không phải phương pháp có độ chính xác cao nhất trên mọi dataset. DVC đạt kết quả cao hơn trên CIFAR-100 và CORe50, còn EASE nhỉnh hơn trên Tiny-ImageNet.

Điều này cho thấy thế mạnh của F-OAL không chỉ nằm ở Accuracy tuyệt đối mà ở **sự cân bằng giữa độ chính xác, tốc độ huấn luyện và chi phí bộ nhớ**.

*Lưu ý khi trích dẫn:* Section 4.4 của paper ghi độ chính xác trên FGVCAircraft là 66,2%, nhưng Table 2 và Table 4 ghi 62,2%. Trong báo cáo này, số liệu 62,2% được lấy từ bảng kết quả chính.

### 7.3. Forgetting Rate

Trên CIFAR-100, F-OAL có:

| Chỉ số | Kết quả |
|---|---:|
| Average Accuracy | 91,1% |
| Last Accuracy | 86,5% |
| Forgetting Rate | 5,5% |

Trong khi đó, DVC có Forgetting Rate 8,2% trên cùng dataset.

Kết quả cho thấy F-OAL có khả năng hạn chế việc quên kiến thức cũ trong thiết lập thực nghiệm của paper.

Tuy vậy, Forgetting Rate không nên được đánh giá riêng lẻ. Một mô hình có độ chính xác thấp vẫn có thể có mức giảm độ chính xác nhỏ. Vì thế, cần kết hợp Forgetting Rate với Average Accuracy và Last Accuracy khi so sánh các phương pháp.

### 7.4. Thời gian huấn luyện

Theo Table 3, thời gian huấn luyện, bao gồm trích xuất đặc trưng, được báo cáo như sau:

| Dataset | F-OAL | ER | EASE |
|---|---:|---:|---:|
| CIFAR-100 | 261 giây | 652 giây | 383 giây |
| FGVCAircraft | 16 giây | 40 giây | 147 giây |
| DTD | 8 giây | 21 giây | 139 giây |
| Tiny-ImageNet | 507 giây | 1.315 giây | 638 giây |

Trên CIFAR-100, F-OAL nhanh hơn ER khoảng 2,5 lần. Trên DTD, F-OAL nhanh hơn EASE khoảng 17,4 lần trong cấu hình thực nghiệm được báo cáo.

Lợi thế này đến từ việc không cần Backpropagation qua ViT và không cần cập nhật toàn bộ mạng trong mỗi mini-batch.

### 7.5. Bộ nhớ GPU

Theo Figure 2, F-OAL có **Peak GPU Memory khoảng 1,9 GB** trên CIFAR-100 với batch size 10.

Một số phương pháp dựa trên Experience Replay sử dụng nhiều bộ nhớ hơn, chẳng hạn ER khoảng 6,3 GB và iCaRL khoảng 7,5 GB trong cùng thiết lập đo.

Tuy nhiên, F-OAL không phải phương pháp dùng ít VRAM nhất trong tất cả baseline. Một số phương pháp exemplar-free khác có kết quả bộ nhớ thấp hơn. Vì vậy, kết luận chính xác là **F-OAL đạt độ chính xác cao với mức sử dụng GPU tương đối thấp**, không phải luôn tối thiểu hóa VRAM.

---

## 8. Ablation Study

Ablation Study được thực hiện để kiểm tra mức đóng góp của từng thành phần trong mô hình.

### 8.1. Feature Fusion và Smooth Projection

Theo Table 4, kết quả trên DTD như sau:

| Feature Fusion | Smooth Projection | Average Accuracy |
|---|---|---:|
| Có | Có | 82,8% |
| Không | Có | 80,5% |
| Có | Không | 79,3% |
| Không | Không | 71,2% |

Khi sử dụng cả Feature Fusion và Smooth Projection, độ chính xác đạt 82,8%. Khi loại bỏ cả hai thành phần, độ chính xác giảm xuống 71,2%, tương ứng mức giảm 11,6 điểm phần trăm.

Kết quả này cho thấy việc kết hợp đặc trưng từ nhiều transformer block và biến đổi sang không gian đặc trưng mới có thể cải thiện đáng kể hiệu suất trên bộ dữ liệu DTD.

Tuy nhiên, ảnh hưởng của các thành phần không giống nhau trên mọi dataset. Ví dụ, trên Tiny-ImageNet, một số cấu hình loại bỏ thành phần vẫn cho độ chính xác tương đương hoặc nhỉnh hơn cấu hình đầy đủ.

### 8.2. Vai trò của Analytic Classifier

Tác giả so sánh Analytic Classifier với một Fully Connected Classifier có cấu trúc tương tự nhưng được cập nhật bằng Backpropagation.

Trong thí nghiệm trên CIFAR-100, mô hình sử dụng Analytic Classifier đạt 91,1% Average Accuracy, trong khi cấu hình thay thế bằng Fully Connected Classifier được báo cáo đạt 32,4%.

Kết quả cho thấy cơ chế cập nhật RLS đóng vai trò rất quan trọng trong hiệu suất của F-OAL ở thiết lập OCIL được nghiên cứu.

Tuy nhiên, kết quả này **không chứng minh Backpropagation nói chung kém hơn RLS trong mọi bài toán**. Đây là so sánh giữa hai cấu hình cụ thể trong cùng điều kiện thực nghiệm.

---

## 9. Đánh giá ưu điểm và hạn chế

### 9.1. Ưu điểm

**Thứ nhất, giảm Catastrophic Forgetting.** RLS duy trì nghiệm của bài toán Ridge Regression trên dữ liệu tích lũy, từ đó hạn chế tình trạng bộ phân loại chỉ tối ưu quá mức cho các lớp vừa xuất hiện.

**Thứ hai, không cần Experience Replay.** F-OAL không yêu cầu lưu ảnh cũ để huấn luyện lại. Thay vào đó, thông tin thống kê từ dữ liệu đã quan sát được duy trì thông qua các ma trận trạng thái.

**Thứ ba, huấn luyện nhanh.** Vì backbone và phép chiếu đều cố định, quá trình học tăng dần chủ yếu gồm Forward Pass và các phép toán ma trận.

**Thứ tư, tiết kiệm bộ nhớ.** Không cần lưu gradient và trạng thái tối ưu cho toàn bộ ViT trong giai đoạn học tăng dần, qua đó giảm nhu cầu sử dụng GPU.

**Thứ năm, có cơ sở toán học rõ ràng.** Phương pháp dựa trên Ridge Regression và Woodbury Matrix Identity, cho phép chứng minh tính tương đương giữa nghiệm cập nhật tuần tự và nghiệm giải trên dữ liệu tích lũy.

### 9.2. Hạn chế

**Phụ thuộc vào pre-trained encoder.** Đây là giới hạn quan trọng nhất. F-OAL giả định encoder đã học được biểu diễn đủ tốt cho những lớp sẽ xuất hiện. Nếu dữ liệu mới khác biệt nhiều so với dữ liệu pre-training, việc chỉ cập nhật bộ phân loại tuyến tính có thể không đủ.

**Không thích nghi biểu diễn đặc trưng.** Vì ViT và ma trận chiếu đều cố định, mô hình không thể học thêm những đặc trưng thị giác mới thông qua Fine-tuning trong quá trình học tăng dần.

**Chi phí ma trận tăng theo số chiều đặc trưng.** Ma trận $R$ có kích thước $D\times D$, nên bộ nhớ dành cho ma trận này tăng theo $O(D^2)$.

**Yêu cầu dữ liệu có nhãn.** F-OAL là phương pháp Online Supervised Learning. Nếu chỉ nhận được $X$ mà không có nhãn $Y$, mô hình không thể trực tiếp thực hiện bước cập nhật bộ phân loại theo công thức RLS của paper. Việc kết hợp Pseudo-Labeling hoặc Self-Supervised Learning sẽ là một hướng nghiên cứu mở rộng.

**Chưa giải quyết đầy đủ dữ liệu mất cân bằng.** Khi dữ liệu có phân phối long-tailed hoặc các lớp nhận số lượng mẫu rất khác nhau, chỉ cập nhật classifier có thể chưa đủ để đạt hiệu suất tốt. Tác giả cũng đề cập đây là hướng nghiên cứu tương lai.

**Lợi thế tài nguyên phụ thuộc vào mô hình pre-trained có sẵn.** F-OAL tiết kiệm tài nguyên trong giai đoạn học tăng dần, nhưng không loại bỏ chi phí tính toán đã được sử dụng để pre-train encoder ban đầu.

**Nghiệm giải tích không bảo đảm không quên tuyệt đối về mặt dự đoán.** Tính tương đương với joint Ridge Regression chỉ áp dụng cho hàm mục tiêu Least Squares dưới những giả định đã xác định. Nó không đồng nghĩa với việc độ chính xác của từng lớp cũ luôn được duy trì.

---

## 10. Khả năng tái hiện thực nghiệm

Đối với phạm vi một dự án môn Nhập môn Machine Learning, có thể ưu tiên tái hiện F-OAL trên **CIFAR-100** trước khi mở rộng sang những dataset khác.

CIFAR-100 được chia thành 10 task, mỗi task gồm 10 lớp. Mô hình phải học tuần tự qua 10 task và sau mỗi task phải phân loại được tất cả lớp đã xuất hiện.

Cấu hình cần cố gắng giữ nhất quán với paper gồm ViT-B pre-trained trên ImageNet-1K, batch size 10, kích thước Smooth Projection 1.000 chiều và hệ số regularization $\gamma=1$.

Nhóm có thể sử dụng mã nguồn chính thức do tác giả công bố để giảm rủi ro triển khai sai thuật toán.

Trong giai đoạn đầu, cần xác nhận mô hình có thể tải checkpoint ViT, chạy trích xuất đặc trưng và cập nhật Analytic Classifier bằng RLS. Sau đó thực hiện thí nghiệm CIFAR-100 đầy đủ, lưu lại các chỉ số Accuracy, Forgetting Rate, thời gian huấn luyện và Peak GPU Memory.

Kết quả tham chiếu trên CIFAR-100 của paper là Average Accuracy 91,1%, Last Accuracy 86,5% và Forgetting Rate 5,5%.

Nếu có thêm thời gian, nhóm có thể thực hiện Ablation Study bằng cách lần lượt loại bỏ Feature Fusion hoặc Smooth Projection để đánh giá vai trò của các thành phần.

**Về tài nguyên**, paper sử dụng GPU RTX 4070 Ti 12 GB và báo cáo mức sử dụng VRAM của F-OAL khoảng 1,9 GB trên CIFAR-100. Vì vậy, việc sử dụng Kaggle hoặc Google Colab để tái hiện phương pháp chính là hướng khả thi, mặc dù thời gian chạy và khả năng cấp GPU không được bảo đảm.

Các kết quả tái hiện cần được báo cáo trung thực, bao gồm phiên bản thư viện, checkpoint sử dụng, random seed, cấu hình GPU và những khác biệt giữa môi trường thực nghiệm của nhóm với môi trường của tác giả.

---

## 11. Hướng nghiên cứu mở rộng

Một hướng mở rộng đáng chú ý là kết hợp F-OAL với **Semi-Supervised Learning hoặc Self-Supervised Learning** để giảm sự phụ thuộc vào nhãn của dữ liệu online.

Trong các hệ thống thực tế, dữ liệu có thể xuất hiện liên tục nhưng nhãn không phải lúc nào cũng được cung cấp ngay lập tức. Do đó, việc kết hợp cơ chế phát hiện lớp mới, tạo pseudo-label và cập nhật bộ phân loại giải tích có thể mở rộng phạm vi ứng dụng của phương pháp.

Một hướng khác là cho phép encoder thích nghi có kiểm soát với phân phối dữ liệu mới. Tuy nhiên, điều này tạo ra thách thức vì khi không gian đặc trưng thay đổi, những thống kê đã lưu trong $R$ không còn trực tiếp tương ứng với biểu diễn mới. Một phương pháp mở rộng cần giải quyết vấn đề đồng bộ hoặc hiệu chỉnh những thống kê cũ.

Ngoài ra, có thể nghiên cứu các phương pháp giảm kích thước hoặc xấp xỉ ma trận $R$ để cải thiện khả năng mở rộng khi số chiều đặc trưng lớn.

Đây là những hướng đề xuất dựa trên phân tích hạn chế của F-OAL, **không phải các cải tiến đã được chứng minh trong paper gốc**.

---

## 12. Kết luận

F-OAL đề xuất một hướng tiếp cận hiệu quả cho **Online Class-Incremental Learning**, kết hợp khả năng biểu diễn của mô hình pre-trained với phương pháp học giải tích dựa trên Recursive Least Squares.

Điểm khác biệt chính so với nhiều phương pháp Continual Learning truyền thống là F-OAL không sử dụng Backpropagation để cập nhật mô hình trong giai đoạn học tăng dần. Thay vào đó, phương pháp giữ cố định encoder và cập nhật một bộ phân loại tuyến tính thông qua các phép toán đại số.

Nhờ cơ chế RLS, bộ phân loại có thể duy trì nghiệm Ridge Regression trên dữ liệu tích lũy mà không cần lưu các mẫu huấn luyện cũ để replay. Điều này giúp giảm chi phí bộ nhớ, rút ngắn thời gian học và hạn chế một phần hiện tượng Catastrophic Forgetting.

Kết quả thực nghiệm cho thấy F-OAL có độ chính xác cạnh tranh, đặc biệt tốt trên một số bộ dữ liệu fine-grained, đồng thời sử dụng tài nguyên tính toán tương đối thấp.

Tuy nhiên, phương pháp vẫn phụ thuộc mạnh vào chất lượng của pre-trained encoder, yêu cầu dữ liệu có nhãn và chưa giải quyết toàn diện bài toán thích nghi biểu diễn hoặc dữ liệu mất cân bằng.

**Đóng góp quan trọng nhất của F-OAL là chứng minh rằng, khi có một không gian đặc trưng cố định đủ tốt, việc học lớp mới có thể được thực hiện hiệu quả bằng các phương pháp giải tích thay vì nhất thiết phải cập nhật toàn bộ mạng bằng Backpropagation.**

Đối với môn Nhập môn Machine Learning, paper có giá trị nghiên cứu vì kết nối nhiều kiến thức nền tảng như **Linear Regression, Ridge Regression, Regularization, Matrix Algebra, Recursive Least Squares, Transfer Learning và Continual Learning** trong một hệ thống hoàn chỉnh.

---

## 13. Tài liệu tham khảo

[1] Zhuang, H., Liu, Y., He, R., Tong, K., Zeng, Z., Chen, C., Wang, Y., & Chau, L.-P. (2024). *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning*. Advances in Neural Information Processing Systems, 37, 41517–41538. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/48ffa38c13078d6ce26b328e7f373243-Abstract-Conference.html).

[2] Zhuang, H., et al. (2024). *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning*. [Full Paper PDF](https://papers.neurips.cc/paper_files/paper/2024/file/48ffa38c13078d6ce26b328e7f373243-Paper-Conference.pdf).

[3] Liu, Y., et al. *F-OAL: Official Implementation*. [GitHub Repository](https://github.com/liuyuchen-cz/F-OAL).

[4] ICORE Conference Portal. *Advances in Neural Information Processing Systems (NeurIPS)*. ICORE2026 Rank A*. [Official Conference Ranking](https://portal.core.edu.au/conf-ranks/98/).


---

## 14. Bảng Ánh xạ Bằng chứng Thực nghiệm (README Evidence Mapping — Checkpoint 3)

| Experiment | Implementation | Dataset | Seed | $\bar{A}$ | $A_{10}$ | $F$ ($F_{\text{official}}$ / $F_{\text{ours}}$) | Git SHA | Local Artifact | W&B Run | Status |
|---|---|---|---:|---:|---:|---:|---|---|---|---|
| **F-OAL Paper** | Paper (Table 2 & 3) | CIFAR-100 | 3 runs | 91.10% | 86.50% | 5.50% / — | — | — | — | `PAPER_REPORTED` |
| **F-OAL Official** | Original (`liuyuchen-cz/F-OAL`) | CIFAR-100 | `0` (`n=1, std=N/A`) | **90.98%** | **85.84%** | **4.69%** / **5.17%** | `73e4ac1` | [`results/official_foal_cifar100_seed0/`](results/official_foal_cifar100_seed0/) | [Run `3eca8232`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232) | `VERIFIED` (`A19 PASS`) |
| **F-OAL Full** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **79.33%** | **70.92%** | **8.29%** / **9.21%** | `aad7500` | [`results/foal_full_cifar100_seed42/`](results/foal_full_cifar100_seed42/) | [Run `2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv) | `VERIFIED` (`P1.2 PASS`) |
| **F-OAL Controlled (`C0` Metric)** | Ours (`src/foal`) | CIFAR-100 | `42` / `0` | Verified | Verified | **5.17%** / **4.69%** (Orig) & **9.21%** / **8.29%** (Ours) | `HEAD` | [`docs/experiments.md`](docs/experiments.md) | Verified | `VERIFIED` (`C0 PASS`) |
| **C1: P0 Gaussian std=1** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **79.33%** | **70.95%** | **8.25%** / **9.17%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k) | `VERIFIED FULL GPU` (`C1 PASS`) |
| **C1: P1 Official Uniform** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.28%** | **74.84%** | **8.07%** / **8.97%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z) | `VERIFIED FULL GPU` (`C1 PASS`) |
| **C1: P2 Matched Gaussian** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.37%** | **74.99%** | **7.93%** / **8.81%** | `269f05c` | [`results/c1_analysis/`](results/c1_analysis/) | [Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p) | `VERIFIED FULL GPU` (`C1 PASS`) |
| **C2: B0 Control (`augreg_in1k`)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **83.28%** | **74.84%** | **8.07%** / **8.97%** | `7fd0f4a` | [`results/c2_analysis/`](results/c2_analysis/) | [Run `3oiftwg6`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3oiftwg6) | `VERIFIED FULL GPU` (`C2 PASS`) |
| **C2: B1 Treatment (`augreg2_in21k_ft_in1k`)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **91.77%** | **87.43%** | **4.52%** / **5.02%** | `7fd0f4a` | [`results/c2_analysis/`](results/c2_analysis/) | [Run `ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | `VERIFIED FULL GPU` (`C2 SIGNED OFF`) |
| **C3-A Baseline (Bicubic, Norm)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **91.77%** | **87.43%** | **4.52%** / **5.02%** | `7fd0f4a` | [`results/c2_analysis/`](results/c2_analysis/) | [Run `ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | `REUSED C2-B1` (`C3_A_REUSE_APPROVED`) |
| **C3-B (Bilinear, Norm)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **91.39%** | **86.95%** | **4.40%** / **4.89%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3) | `VERIFIED FULL GPU` (`C3 PASS`) |
| **C3-C (Bicubic, Raw)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **90.88%** | **86.35%** | **4.71%** / **5.23%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h) | `VERIFIED FULL GPU` (`C3 PASS`) |
| **C3-D (Bilinear, Raw — Official Aligned)** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **90.40%** | **85.73%** | **4.64%** / **5.16%** | `15b8afc` | [`results/c3_analysis/`](results/c3_analysis/) | [Run `3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | `VERIFIED FULL GPU` (`C3 PASS`) |
| **F-OAL Multi-Seed** | Ours (`src/foal`) | CIFAR-100 | `42, 123, 2026` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `NOT YET SYNCED` | `PENDING` (`A12`) |
| **Table 4: A0 Full F-OAL** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1`) | **90.40%** | **85.73%** | **4.64%** / **5.16%** | `15b8afc` | [`results/table4_analysis/`](results/table4_analysis/) | [Run `3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | `REUSED C3-D` (`TABLE4 SIGNED OFF`) |
| **Table 4: A1 w/o FF** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1`) | **90.10%** | **85.48%** | **4.90%** / **5.44%** | `d40310f` | [`results/table4_analysis/`](results/table4_analysis/) | [Run `shl4p10l`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/shl4p10l) | `VERIFIED FULL GPU` (`TABLE4 SIGNED OFF`) |
| **Table 4: A2 w/o SP** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1`) | **90.38%** | **85.72%** | **4.72%** / **5.24%** | `d40310f` | [`results/table4_analysis/`](results/table4_analysis/) | [Run `rkedlkkk`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/rkedlkkk) | `VERIFIED FULL GPU` (`TABLE4 SIGNED OFF`) |
| **Table 4: A3 w/o Both** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1`) | **89.92%** | **85.48%** | **4.79%** / **5.32%** | `d40310f` | [`results/table4_analysis/`](results/table4_analysis/) | [Run `k01pp8st`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/k01pp8st) | `VERIFIED FULL GPU` (`TABLE4 SIGNED OFF`) |
| **Headline: LwF Distill** | Ours (`src/foal/baselines/lwf.py`) | CIFAR-100 | `42` (`n=1`) | Paper Ref: 69.3% | Paper Ref: 64.8% | Paper Ref: 1.3% | `HEAD` | [`configs/headline/lwf_cifar100.yaml`](configs/headline/lwf_cifar100.yaml) | `PENDING COLAB RUN` | `READY FOR GPU` (`G12`) |
| **Headline: ER Replay (5000)** | Ours (`src/foal/baselines/er.py`) | CIFAR-100 | `42` (`n=1`) | Paper Ref: 90.1% | Paper Ref: 84.6% | Paper Ref: 20.7% | `HEAD` | [`configs/headline/er_cifar100.yaml`](configs/headline/er_cifar100.yaml) | `PENDING COLAB RUN` | `READY FOR GPU` (`G13`) |
| **Cross-Dataset: DTD Track** | Ours (`src/foal/dataset_adapters/dtd.py`) | DTD (40 classes) | `42` | Table 4 Ref: 82.8% | Table 4 w/o Both: 71.2% | -11.6 pp drop | `HEAD` | [`docs/team_parallel_dataset_plan.md`](docs/team_parallel_dataset_plan.md) | Member A Branch | `ADAPTER IMPLEMENTED` |
| **Cross-Dataset: Aircraft Track** | Ours (`src/foal/dataset_adapters/fgvc_aircraft.py`) | FGVC-Aircraft (100 classes) | `42` | Table 4 Ref: 62.2% | Text: 66.2% Discrepancy | Table 4 w/o Both: 56.0% | `HEAD` | [`docs/team_parallel_dataset_plan.md`](docs/team_parallel_dataset_plan.md) | Member B Branch | `ADAPTER IMPLEMENTED` |

* **Lộ trình Tổng thể & Cột mốc Tái hiện:** [`docs/FOAL_FINAL_REPRODUCTION_ROADMAP.md`](docs/FOAL_FINAL_REPRODUCTION_ROADMAP.md)
* **Tiêu chuẩn Hoàn tất Pha CIFAR-100 (Checklist 15 tiêu chí):** [`docs/cifar100_completion_checklist.md`](docs/cifar100_completion_checklist.md)
* **Kế hoạch Phân công Nhóm Song song (DTD & Aircraft):** [`docs/team_parallel_dataset_plan.md`](docs/team_parallel_dataset_plan.md)
* **Kế hoạch So sánh Baseline (Table 2 & 3):** [`docs/headline_comparison_plan.md`](docs/headline_comparison_plan.md)

* **Hướng dẫn triển khai & thực nghiệm chi tiết:** [`README_IMPLEMENTATION.md`](README_IMPLEMENTATION.md)
* **Kế hoạch thực nghiệm & kết quả tái hiện:** [`docs/experiments.md`](docs/experiments.md)
* **Báo cáo kiểm toán kỹ thuật & tuân thủ Checkpoint 3:** [`docs/FOAL_AUDIT_AND_FIX_PLAN.md`](docs/FOAL_AUDIT_AND_FIX_PLAN.md)
* **Đối chiếu giao thức Code gốc tác giả vs. Code nhóm (`A19`/`A20`):** [`docs/original_vs_ours_protocol.md`](docs/original_vs_ours_protocol.md)
