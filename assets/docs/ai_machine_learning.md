# Tài Liệu Chuẩn Xác: Trí Tuệ Nhân Tạo & Học Máy (AI & Machine Learning)

Tài liệu chuẩn xác phục vụ sinh kịch bản và hoạt hoạ thuật toán AI (phong cách 3Blue1Brown, Jay Alammar).

---

## 1. Cơ Chế Tự Chú Ý (Self-Attention Mechanism) trong Transformer

### 1.1 Bộ Ba Vector Q-K-V (Query, Key, Value)
- Mỗi token đầu vào sau khi qua Embedding được nhân với 3 ma trận trọng số học được ($W_Q, W_K, W_V$) để tạo ra:
  - **Query ($Q$)**: Đại diện cho câu hỏi: *"Từ này đang tìm kiếm thông tin gì xung quanh?"*
  - **Key ($K$)**: Đại diện cho nhãn dán: *"Từ này chứa thông tin gì để trả lời câu hỏi của các từ khác?"*
  - **Value ($V$)**: Đại diện cho nội dung ý nghĩa thực tế của từ được truyền đi.

### 1.2 Công Thức Scaled Dot-Product Attention Chuẩn
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **Ý nghĩa từng bước toán học**:
  - $Q K^T$: Tích vô hướng đo độ tương đồng góc giữa câu hỏi $Q$ và nhãn $K$. Nếu 2 vector cùng hướng, tích vô hướng lớn (độ liên quan cao).
  - Chia cho $\sqrt{d_k}$: Hệ số co giãn (Scaling factor) để ngăn tích vô hướng quá lớn khi số chiều $d_k$ cao (ví dụ $d_k = 64$ hoặc $128$). Nếu không chia, giá trị lớn sẽ đẩy hàm Softmax vào vùng bão hòa cực đoan khiến gradient bị triệt tiêu ($\approx 0$) gây đứng hình khi huấn luyện.
  - $\text{softmax}(\dots)$: Chuẩn hóa ma trận điểm số thành phân phối xác suất (tổng mỗi hàng bằng 1.0 hay 100%).
  - Nhân với $V$: Trọng số hóa ý nghĩa của các từ xung quanh để cập nhật vector ngữ cảnh mới cho từ hiện tại.

### 1.3 Cơ Chế KV Cache Trong Sinh Text (Autoregressive Generation)
- **Vấn đề**: Khi sinh từng token một (ví dụ sinh từ thứ 100), nếu tính lại toàn bộ attention từ đầu sẽ mất $O(N^2)$ phép tính.
- **Giải pháp**: Lưu trữ sẵn các ma trận Key và Value của 99 token trước đó trong bộ nhớ VRAM (KV Cache). Token mới chỉ cần tính vector Query của chính nó và nhân với KV Cache sẵn có $\implies$ Giảm độ phức tạp sinh 1 token từ $O(N^2)$ xuống $O(N)$!

---

## 2. Lan Truyền Ngược (Backpropagation) & Quy Tắc Chuỗi (Chain Rule)

### 2.1 Bản Chất Hình Học Của Gradient Descent
- Mạng nơ-ron là một hàm toán học nhiều tầng: $y = f_L(\dots f_2(f_1(x; w_1); w_2) \dots)$.
- Hàm mất mát (Loss function) $L(\hat{y}, y)$ đo sai số giữa dự đoán và thực tế.
- Để cập nhật trọng số $w$ theo hướng giảm lỗi nhanh nhất:
  $$w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}$$
  (Trong đó $\eta$ là tốc độ học - Learning Rate).

### 2.2 Quy Tắc Chuỗi Giải Tích (Chain Rule)
- Gradient được lan truyền ngược từ đầu ra về đầu vào qua đạo hàm riêng từng chặng:
  $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$
- Bằng cách lưu trữ giá trị đầu ra của lượt truyền thẳng (Forward pass), lượt truyền ngược (Backward pass) tính toán toàn bộ gradient cho hàng triệu trọng số chỉ trong một lần duyệt ngược $O(N)$, nhanh hơn tính đạo hàm giải tích thô hàng tỷ lần.

---

## 3. RLHF (Học Tăng Cường Từ Phản Hồi Con Người) & Thuật Toán PPO

### 3.1 Quy Trình 3 Pha Chuẩn Của Mô Hình Ngôn Ngữ Hiện Đại
1. **Pre-training**: Huấn luyện tự giám sát trên hàng nghìn tỷ token để học ngữ pháp và tri thức thế giới (Base model).
2. **SFT (Supervised Fine-Tuning)**: Dạy mô hình đóng vai trợ lý AI thông qua các cặp câu hỏi - câu trả lời chất lượng cao.
3. **RLHF**: Uốn nắn mô hình theo tiêu chuẩn 3H (Helpful, Honest, Harmless).
   - Huấn luyện **Reward Model (RM)**: Con người xếp hạng các câu trả lời tốt/xấu $\to$ RM học cách chấm điểm số cho văn bản.
   - Dùng thuật toán **PPO (Proximal Policy Optimization)**: Tối đa hóa điểm thưởng từ Reward Model, đồng thời áp dụng hình phạt **KL Divergence** so với mô hình gốc để ngăn mô hình bị biến chất (Reward Hacking / Mode Collapse).
