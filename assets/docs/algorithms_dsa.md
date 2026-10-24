# Tài Liệu Chuẩn Xác: Thuật Toán & Cấu Trúc Dữ Liệu (Algorithms & Data Structures)

Tài liệu chuẩn xác phục vụ sinh kịch bản và hoạt hoạ thuật toán trực quan (phong cách 3Blue1Brown, NeetCode).

---

## 1. Kỹ Thuật Hai Con Trỏ (Two Pointers & Two Sum)

### 1.1 Bài Toán Chuẩn (Two Sum II - Mảng Đã Sắp Xếp)
- **Đề bài**: Cho mảng số nguyên đã sắp xếp tăng dần `A = [2, 7, 11, 15]`, tìm vị trí hai số có tổng bằng `Target = 9`.
- **Cách tiếp cận ngây thơ (Brute-Force)**:
  - 2 vòng lặp lồng nhau kiểm tra mọi cặp $(i, j)$: $O(N^2)$ thời gian, $O(1)$ bộ nhớ. Với $N = 10^5$, mất $10^{10}$ phép tính $\to$ TLE (Time Limit Exceeded).
- **Cách tối ưu với Two Pointers**:
  - Đặt `left = 0` (trỏ vào `2`), `right = 3` (trỏ vào `15`).
  - **Bước 1**: $A[\text{left}] + A[\text{right}] = 2 + 15 = 17 > 9$. Vì mảng tăng dần, cần giảm tổng $\implies \text{right} = \text{right} - 1$ (sang vị trí 2, giá trị `11`).
  - **Bước 2**: $A[\text{left}] + A[\text{right}] = 2 + 11 = 13 > 9 \implies \text{right} = \text{right} - 1$ (sang vị trí 1, giá trị `7`).
  - **Bước 3**: $A[\text{left}] + A[\text{right}] = 2 + 7 = 9 == 9$. Tìm thấy cặp nghiệm tại chỉ số `[0, 1]`!
- **Độ phức tạp**: Thời gian $O(N)$, Bộ nhớ $O(1)$.

---

## 2. Thuật Toán Sắp Xếp Trộn (Merge Sort) & Cây Đệ Quy

### 2.1 Mảng Dữ Liệu Mẫu Cho Hoạt Hoạ
- **Mảng đầu vào**: `[5, 2, 8, 1, 9, 3, 7, 4]` (8 phần tử).
- **Pha 1: Chia để trị (Divide)**:
  - Tầng 0: `[5, 2, 8, 1, 9, 3, 7, 4]`
  - Tầng 1: `[5, 2, 8, 1]` và `[9, 3, 7, 4]`
  - Tầng 2: `[5, 2]`, `[8, 1]` và `[9, 3]`, `[7, 4]`
  - Tầng 3: `[5]`, `[2]`, `[8]`, `[1]`, `[9]`, `[3]`, `[7]`, `[4]` (cơ sở đệ quy)
- **Pha 2: Trộn có thứ tự (Merge)**:
  - Trộn tầng 2: `[2, 5]`, `[1, 8]`, `[3, 9]`, `[4, 7]`
  - Trộn tầng 1: `[1, 2, 5, 8]` và `[3, 4, 7, 9]`
  - Trộn tầng 0: `[1, 2, 3, 4, 5, 7, 8, 9]`
- **Chứng minh Big-O trực quan bằng Cây Đệ Quy**:
  - Độ sâu của cây: $\log_2(N)$ tầng (với $N=8$, chiều cao là 3).
  - Công việc tại mỗi tầng: Trộn $N$ phần tử mất chính xác $O(N)$ phép so sánh.
  - Tổng thời gian: $\text{Chiều cao} \times \text{Công việc mỗi tầng} = O(N \log N)$ trong mọi trường hợp (Best, Average, Worst).
  - Bộ nhớ phụ: $O(N)$ để lưu mảng tạm khi trộn.

---

## 3. Thuật Toán Tìm Đường A* (A-Star Search)

### 3.1 Hàm Chi Phí Cốt Lõi
- **Công thức**: $f(n) = g(n) + h(n)$
  - $g(n)$: Chi phí thực tế đã đi từ điểm xuất phát đến nút $n$.
  - $h(n)$: Chi phí ước lượng heuristic từ nút $n$ đến đích (khoảng cách Manhattan $|x_1 - x_2| + |y_1 - y_2|$ hoặc Euclidean $\sqrt{\Delta x^2 + \Delta y^2}$).
  - $f(n)$: Tổng chi phí ước tính đường đi ngắn nhất qua nút $n$.
- **Ưu thế vượt trội so với Dijkstra**:
  - Dijkstra nở rộng sóng tròn đều theo mọi hướng (lãng phí duyệt các nút đi xa khỏi đích).
  - A* sử dụng $h(n)$ kéo hướng duyệt ưu tiên lao thẳng về đích như nam châm, giảm 70–80% số nút cần duyệt.

---

## 4. Cấu Trúc B+Tree Trong Database & Storage Engine

### 4.1 Vì Sao Database Dùng B+Tree Thay Vì B-Tree Hay Binary Search Tree?
- **Đặc điểm kiến trúc**:
  - Tất cả dữ liệu thực tế (Record/Payload) chỉ nằm ở các nút lá (Leaf Nodes).
  - Các nút trung gian (Internal Nodes) chỉ chứa khóa định tuyến (Routing Keys), không chứa payload $\to$ Mỗi nút nạp được hàng nghìn khóa vào 1 khối đĩa (Page 4KB hoặc 16KB).
  - Các nút lá được nối với nhau bằng danh sách liên kết đôi (Doubly Linked List).
- **Hiệu năng quét dải (Range Scan)**:
  - Truy vấn `WHERE age BETWEEN 20 AND 30`: B+Tree chỉ cần duyệt cây $O(\log N)$ để tìm lá `age = 20`, sau đó duyệt tuần tự theo con trỏ `next` sang các lá tiếp theo $\to$ Tốc độ đọc đĩa tuần tự (Sequential I/O) cực nhanh.

---

## 5. Bloom Filter (Bộ Lọc Xác Suất)

### 5.1 Cơ Chế Bit Array & K Hàm Băm
- Mảng bit độ dài $m$, ban đầu toàn bit 0.
- Có $k$ hàm băm độc lập $(h_1, h_2, \dots, h_k)$.
- **Thêm phần tử $x$**: Đặt các bit tại vị trí $h_1(x), h_2(x), \dots, h_k(x)$ thành 1.
- **Kiểm tra phần tử $y$**:
  - Nếu bất kỳ bit nào tại $h_i(y) == 0 \implies$ **CHẮC CHẮN 100%** phần tử chưa từng xuất hiện (No False Negatives).
  - Nếu tất cả bit đều là 1 $\implies$ Phần tử **CÓ THỂ** đã xuất hiện (Có xác suất dương tính giả - False Positive do va chạm băm).
- **Tỉ lệ lỗi dương tính giả**: $P \approx (1 - e^{-kn/m})^k$. Tối ưu khi chọn $k = \frac{m}{n} \ln 2$.
