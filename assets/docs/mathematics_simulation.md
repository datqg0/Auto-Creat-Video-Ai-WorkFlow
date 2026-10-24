# Tài Liệu Chuẩn Xác: Toán Học Trực Quan & Mô Phỏng Tự Nhiên (Math & Natural Simulations)

Tài liệu chuẩn xác phục vụ kịch bản và hoạt hoạ toán học (phong cách 3Blue1Brown, Numberphile).

---

## 1. Biến Đổi Fourier (Fourier Transform) & Vòng Tròn Quay Epicycles

### 1.1 Bản Chất Vật Lý & Toán Học
- Mọi tín hiệu tuần hoàn (âm thanh, sóng điện từ, hay một nét vẽ đường cong khép kín 2D) đều có thể phân tích thành tổng của các sóng sin hoặc vòng tròn quay với tần số và biên độ khác nhau.
- **Công thức biến đổi liên tục**:
  $$\hat{f}(\xi) = \int_{-\infty}^{\infty} f(t) e^{-2\pi i \xi t} dt$$
- **Ẩn dụ trực quan 3Blue1Brown**: Quấn tín hiệu $f(t)$ quanh gốc tọa độ trên mặt phẳng phức với tần số quay $\xi$. Khi tần số quay trùng với tần số nội tại của tín hiệu, "khối tâm" của vòng quấn bị kéo lệch xa khỏi gốc tọa độ $\implies$ Phát hiện tần số đó có mặt trong tín hiệu!
- **Vẽ tranh bằng vòng tròn (Epicycles)**: Chuỗi Fourier rời rạc $c_n = \frac{1}{N} \sum_{k=0}^{N-1} z_k e^{-i 2\pi n k / N}$. Mỗi hệ số $c_n$ là một vector quay (bán kính $|c_n|$, góc pha ban đầu $\arg(c_n)$). Ghép đầu vector này vào đuôi vector kia $\implies$ Đầu bút của vector cuối cùng vẽ ra hình ảnh hoàn chỉnh bất kỳ.

---

## 2. Tập Hợp Fractal Mandelbrot (The Mandelbrot Set)

### 2.1 Công Thức & Quy Tắc Sinh Vô Hạn Kỳ Vĩ
- Xét dãy số phức truy hồi trong mặt phẳng phức:
  $$z_0 = 0, \quad z_{n+1} = z_n^2 + c$$
  (với $c$ là một số phức $c = x + y i$).
- **Điều kiện thuộc tập Mandelbrot**:
  - Nếu khi $n \to \infty$, độ lớn $|z_n|$ không bao giờ vượt quá 2 (tức là không tiến tới vô cực) $\implies$ Điểm $c$ thuộc về tập Mandelbrot (tô màu đen).
  - Nếu $|z_n| > 2$ tại một bước lặp $k$ $\implies$ Điểm $c$ sẽ thoát tới vô cực $\implies$ Tô màu điểm đó theo tốc độ thoát $k$ (tạo nên dải viền màu huyền ảo).
- **Tính tự đồng dạng (Self-Similarity)**: Khi phóng to hàng tỷ lần vào đường biên của tập Mandelbrot, ta liên tục bắt gặp các "tiểu Mandelbrot" thu nhỏ giống hệt hình thể ban đầu, đi kèm với các xoáy ốc (spirals) và đuôi cá ngựa vô tận.

---

## 3. Trò Chơi Cuộc Sống (Conway's Game of Life)

### 3.1 Luật Chơi 4 Nguyên Tắc Trên Lưới 2D
Mỗi ô trên lưới 2D có 2 trạng thái: Sống (1) hoặc Chết (0), có 8 ô lân cận (Moore neighborhood). Thế hệ tiếp theo được quyết định đồng thời bởi 4 luật:
1. **Chết vì cô độc (Underpopulation)**: Ô sống có ít hơn 2 ô lân cận sống $\to$ Chết.
2. **Sống tiếp (Survival)**: Ô sống có 2 hoặc 3 ô lân cận sống $\to$ Tiếp tục sống sang thế hệ sau.
3. **Chết vì quá đông (Overpopulation)**: Ô sống có nhiều hơn 3 ô lân cận sống $\to$ Chết vì ngột ngạt.
4. **Sinh sản kỳ diệu (Reproduction)**: Ô chết có **chính xác 3 ô lân cận sống** $\to$ Trở thành ô sống mới.

### 3.2 Các Hình Mẫu Kinh Điển
- **Glider (Tàu lượn)**: Cấu trúc 5 ô di chuyển chéo qua lưới vĩnh cửu.
- **Gosper Glider Gun**: Khẩu súng tế bào liên tục bắn ra các tàu lượn vô tận.
- **Tính toàn năng Turing (Turing Completeness)**: Conway's Game of Life có thể mô phỏng lại cổng logic, bộ nhớ, và một chiếc máy tính hoàn chỉnh bên trong chính nó!
