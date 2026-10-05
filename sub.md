# Kế Hoạch & Bộ Khung Sư Phạm: Giải Thích Thuật Toán & Công Thức Toán Học (Algorithmic & Math Visual Plan)

Tài liệu này định hình chiến lược sư phạm trực quan, cấu trúc kịch bản và quy chuẩn thiết kế hình ảnh cho kênh khi giải thích **Thuật toán (Algorithms)**, **Bài tập lập trình (LeetCode / Codeforces)** và **Công thức toán học (Math Concepts & Big-O)** theo chuẩn chất lượng cao (*3Blue1Brown, Veritasium, NeetCode*).

---

## 1. Triết Lý Cốt Lõi (Core Pedagogy Philosophy)

1. **Trực giác đi trước, Công thức theo sau (Intuition First, Formalism Later)**:
   - Không mở đầu bằng định nghĩa toán học khô khan hay code phức tạp.
   - Bắt đầu bằng một câu hỏi trực quan, một câu đố tư duy hoặc một bài toán đời thường. Người xem phải "cảm nhận" được bản chất vấn đề trước khi gán nhãn thuật ngữ.
2. **Nút thắt & Khoảnh khắc "Aha!" (The Bottleneck & The "Aha!" Moment)**:
   - Mọi thuật toán sinh ra đều để giải quyết một nút thắt của cách làm "ngây thơ" (Brute-Force).
   - Luôn cho người xem thấy vì sao cách nghĩ thông thường lại bế tắc (Time Limit Exceeded - TLE), từ đó làm nổi bật sự tài tình của giải pháp tối ưu.
3. **Độ chính xác tuyệt đối (Zero Hallucination)**:
   - Dữ liệu minh họa phải là số liệu thật, mảng thật, đồ thị thật.
   - Mọi phân tích độ phức tạp thời gian/bộ nhớ ($O(N)$, $O(\log N)$, $O(N^2)$) và bất biến thuật toán (Loop Invariants) phải chính xác 100% theo tiêu chuẩn khoa học máy tính.
4. **Nhịp điệu diễn hoạt chậm rãi (Slow-Paced Motion)**:
   - Tuyệt đối không nhảy cóc trạng thái. Mỗi bước di chuyển con trỏ, so sánh, hoán đổi đều phải dừng tối thiểu 0.8s – 1.2s để não bộ người xem kịp xử lý.

---

## 2. Khung Sư Phạm Chuẩn 5 Bước Cho Bài Toán / Thuật Toán

Mỗi video/short giải thích thuật toán sẽ tuân theo cấu trúc 5 pha logic:

```
[Pha 1: Đặt Đề Bài & Ràng Buộc]
       │
       ▼
[Pha 2: Bẫy Ngây Thơ (Brute-Force) & Nút Thắt TLE]
       │
       ▼
[Pha 3: Ý Tưởng Đột Phá ("Aha!" Insight)]
       │
       ▼
[Pha 4: Mô Phỏng Từng Bước Bằng Chuyển Động (Visual Simulation)]
       │
       ▼
[Pha 5: Đóng Gói: Code Tối Giản, Big-O & Bài Tập Tự Luyện]
```

### Chi tiết từng pha:

### Pha 1: Đặt Đề Bài & Ràng Buộc (The Visual Dilemma)
- **Mục tiêu**: Khiến người xem hiểu ngay bài toán trong 10-15 giây đầu.
- **Cách làm**:
  - Dùng **Challenge Card** (màu hổ phách, icon dấu hỏi `?`).
  - Nêu rõ: Input cụ thể $\to$ Output mong muốn.
  - *Ví dụ*: "Cho mảng `[2, 7, 11, 15]`, tìm vị trí 2 số có tổng bằng `9`. Làm sao tìm nhanh nhất nếu mảng có 1 triệu phần tử?"

### Pha 2: Bẫy Ngây Thơ (Brute-Force) & Nút Thắt Hiệu Năng
- **Mục tiêu**: Giải thích cách làm tự nhiên nhất và chứng minh vì sao nó thất bại ở quy mô lớn.
- **Cách làm**:
  - Minh họa 2 vòng lặp lồng nhau duyệt qua mọi cặp phần tử.
  - Hiển thị con số cảnh báo: Với $N = 10^5$, cách làm vét cạn mất $10^{10}$ phép tính $\to$ hệ thống mất hơn 10 giây $\to$ nhận vé TLE ngay lập tức.

### Pha 3: Ý Tưởng Đột Phá (The "Aha!" Insight)
- **Mục tiêu**: Khai mở góc nhìn mới giải quyết nút thắt.
- **Các mẫu tư duy đột phá thường dùng**:
  - *Đổi chiều câu hỏi*: Thay vì tìm $A + B = Target$, ta lật ngược thành: Với mỗi $A$, liệu $(Target - A)$ đã từng xuất hiện chưa? $\to$ Dẫn tới **Hash Table**.
  - *Tận dụng tính chất đã sắp xếp*: Nếu mảng đã tăng dần, một cặp số có tổng quá lớn thì chắc chắn phải giảm con trỏ bên phải $\to$ Dẫn tới **Two Pointers**.
  - *Chia để trị / Tách bài toán con*: Bài toán lớn có cấu trúc bài toán con gối nhau $\to$ Dẫn tới **Quy hoạch động (DP)**.

### Pha 4: Mô Phỏng Từng Bước (Visual Step-by-Step Simulation)
- **Mục tiêu**: Giúp người xem nhìn thấy trực quan dữ liệu biến đổi qua từng dòng lệnh.
- **Tiêu chuẩn màu sắc trạng thái**:
  - **Màu nền**: Cyberpunk tối `#0d1117` hoặc xanh thẫm `#05060f`.
  - **Cyan (`#00f0ff` / `#58a6ff`)**: Phần tử ở trạng thái chờ / chưa xử lý.
  - **Vàng / Hồng neon (`#ffe600` / `#ff2bd6`)**: Con trỏ `left`, `right`, hoặc phần tử đang được xét.
  - **Xanh lá neon (`#39ff14` / `#3fb950`)**: Phần tử đã khớp, tìm thấy đích, hoặc vùng đã sắp xếp xong.
  - **Đỏ / Muted (`#ff7b72` / `#8b949e`)**: Phần tử bị loại bỏ khỏi không gian tìm kiếm.
- **Quy tắc nhịp điệu**: Giữ mỗi pha biến đổi từ 0.8s – 1.5s; không dồn dập quá 4 bước trong 1 cảnh.

### Pha 5: Đóng Gói: Code Tối Giản & Phân Tích Big-O
- **Mục tiêu**: Cung cấp đoạn mã sạch nhất và tổng kết giá trị khoa học.
- **Cách làm**:
  - Chỉ chiếu đoạn code lõi (Core logic 5–10 dòng), không chiếu code boilerplate.
  - So sánh trực quan:
    - Cách ngây thơ: $O(N^2)$ Time, $O(1)$ Space.
    - Cách tối ưu: $O(N)$ Time, $O(N)$ Space.
  - Đưa ra 1 bài toán thực tế tương tự để người xem tự giải và comment đáp án.

---

## 3. Chiến Lược Giải Thích Công Thức Toán Học (3Blue1Brown Style)

Các công thức toán học thường gây sợ hãi nếu chỉ viết dưới dạng ký hiệu. Cần trực quan hóa công thức theo 3 nguyên tắc:

### 1. Trực quan hóa hình học (Geometric Meaning)
- **Đại số tuyến tính (Vector & Ma trận)**:
  - Không viết bảng số $2 \times 2$.
  - Hãy vẽ không gian lưới 2D bị kéo dãn, xoay, hoặc bóp méo (Linear Transformation).
- **Hàm tích phân / Đạo hàm**:
  - Đạo hàm là độ dốc tức thời của tiếp tuyến (Tangent line).
  - Tích phân là diện tích quét dần dưới đường cong theo thời gian (Accumulation).

### 2. Trực quan hóa cây đệ quy cho Big-O (Recursion Trees)
- Để giải thích vì sao Merge Sort đạt $O(N \log N)$:
  - Vẽ cây phân rã mảng thành từng tầng:
    - Có $\log_2(N)$ tầng phân rã.
    - Mỗi tầng cần tổng cộng $N$ phép so sánh khi gộp lại.
    - Tổng phép tính: $N \times \log_2(N)$. Người xem nhìn hình là hiểu ngay lập tức mà không cần chứng minh toán học phức tạp.

### 3. Quy hoạch động: Trực quan hóa bảng DAG / 2D Grid
- Biểu diễn mảng ghi nhớ (Memoization table) bằng lưới ô vuông:
  - Mũi tên chỉ rõ ô $(i, j)$ được tính từ giá trị của ô $(i-1, j)$ và $(i, j-w_i)$.
  - Tránh nói công thức truy hồi thuần túy như $dp[i] = \max(dp[i-1], ...)$ khi chưa vẽ bảng ô vuông.

---

## 4. Danh Mục Thuật Toán & Bài Toán Ưu Tiên Triển Khai (Curriculum Matrix)

### Giai đoạn 1: Các bài toán kinh điển (High Search & High Retention)
1. **Two Pointers & Sliding Window**:
   - Two Sum (Bẫy $O(N^2) \to$ Đột phá Hash Map $O(N)$).
   - Container With Most Water (2 con trỏ co hẹp).
   - Longest Substring Without Repeating Characters (Cửa sổ trượt).
2. **Binary Search & Biến thể**:
   - Tìm kiếm nhị phân cơ bản (Chia đôi không gian tìm kiếm).
   - Tìm kiếm trong mảng xoay vòng (Search in Rotated Sorted Array).
   - Tìm căn bậc hai / Tìm kiếm trên tập kết quả.
3. **Đồ thị cơ bản (Graph Traversals)**:
   - BFS vs DFS: Sự khác biệt giữa hàng đợi (Queue - sóng lan tỏa) và ngăn xếp (Stack - đào sâu).
   - Dijkstra: Thuật toán tìm đường đi ngắn nhất hoạt động trên bản đồ GPS thực tế.

### Giai đoạn 2: Nâng cao & Tư duy chiều sâu
1. **Dynamic Programming (Quy hoạch động)**:
   - Bài toán cái túi (0/1 Knapsack Problem).
   - Dãy con tăng dài nhất (Longest Increasing Subsequence - LIS).
   - Đồng xu đổi tiền (Coin Change).
2. **Cấu trúc dữ liệu nâng cao**:
   - Trie: Cách Google gợi ý từ khóa khi người dùng vừa gõ (Autocomplete).
   - B-Tree: Vì sao cơ sở dữ liệu MySQL/PostgreSQL không dùng cây nhị phân mà dùng B-Tree.
3. **Toán học ứng dụng**:
   - Biến đổi Fourier nhanh (FFT): Cách máy tính nén nhạc MP3 và xử lý sóng âm.
   - Mật mã RSA: Phép nhân modulo và nghịch lý số nguyên tố bảo vệ thẻ tín dụng.

---

## 5. Quy Chuẩn Prompt Cho Script Writer & AI Code Runner

Để tích hợp liền mạch vào codebase hiện tại, khi giải thích bài toán thuật toán, cần tuân thủ các quy tắc prompt sau:

### 1. Chuẩn prompt trong `script_writer.py`:
- Bắt buộc trả về object `exercises` có đủ:
  - `question`: Bài toán có số liệu cụ thể.
  - `hint`: Gợi ý tư duy trực giác (không giải sẵn code).
  - `answer`: Kết quả ngắn gọn kèm độ phức tạp tối ưu.
- Trong `scenes`: Bắt buộc có ít nhất 1 scene mô phỏng từng bước (visual_type: `animation`, preset: `pycode` hoặc `manim`).

### 2. Chuẩn visual prompt cho Animation Code:
- Phải mô tả tối thiểu 3 phase:
  - `Phase 1 (Setup)`: Khởi tạo mảng / đồ thị với màu cyan ban đầu.
  - `Phase 2 (Execution)`: Con trỏ di chuyển qua từng phần tử, đổi màu hồng khi đang xét, xanh khi khớp.
  - `Phase 3 (Conclusion)`: Đóng khung kết quả hoặc hiện chữ `FOUND`/`OPTIMAL` neon.
- Cấm nhồi nhét quá 4 chuyển động trong 1 cảnh.

---

## 6. Lộ Trình Triển Khai (Action Plan)

| Bước | Nhiệm vụ | File tác động | Mục tiêu đầu ra |
|---|---|---|---|
| **B1** | Tích hợp preset chủ đề thuật toán vào `topic_selector.py` | `src/topic_selector.py` | Tự động chọn đề tài từ danh sách 50 bài LeetCode / Computer Science kinh điển |
| **B2** | Tinh chỉnh prompt sư phạm cho bài tập thuật toán | `src/script_writer.py` | LLM luôn chia đủ 5 pha: Đề bài $\to$ Brute-force $\to$ Insight $\to$ Mô phỏng $\to$ Big-O |
| **B3** | Bổ sung helper vẽ mảng & cây nhị phân vào Manim Prelude | `src/ai_code_runner.py` | Cung cấp sẵn các hàm `neon_array()`, `neon_tree()`, `neon_pointer()` giúp AI vẽ cấu trúc dữ liệu cực chuẩn |
| **B4** | Thử nghiệm render 1 video mẫu (Two Sum hoặc Binary Search) | CLI / Pipeline | Đánh giá độ mượt, nhịp điệu diễn hoạt và khả năng tiếp thu của người xem |
