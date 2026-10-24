# Tài Liệu Chuẩn Xác: Đồ Họa Máy Tính & Shaders (Graphic Programming & Shaders)

Tài liệu chuẩn xác phục vụ sinh kịch bản và hoạt hoạ đồ họa trực quan (phong cách Inigo Quilez, 3Blue1Brown, Scratchapixel).

---

## 1. Dò Tia (Ray Tracing) vs Chiếu Đa Giác (Rasterization)

### 1.1 Chiếu Đa Giác (Rasterization)
- **Cơ chế**: Lấy từng tam giác 3D (Vertices $\to$ Triangles), dùng ma trận chiếu (MVP: Model-View-Projection) chiếu phẳng lên màn hình 2D, sau đó thuật toán Scanline kiểm tra xem tâm pixel nào rơi vào trong tam giác và gán màu (Fragment Shader).
- **Ưu điểm**: Cực kỳ nhanh, có phần cứng chuyên dụng trên GPU. Đạt 144 - 240 FPS cho game.
- **Nhược điểm**: Khó tính toán chính xác bóng đổ mềm (Soft Shadows), phản xạ gương (Mirror Reflections), và khúc xạ ánh sáng (Refraction) vì chỉ xử lý độc lập từng tam giác.

### 1.2 Dò Tia Vật Lý (Ray Tracing)
- **Cơ chế**: Bắn các tia ảo từ mắt người xem (Camera) xuyên qua từng pixel trên màn hình vào thế giới 3D.
- Khi tia va chạm một bề mặt vật thể (tính toán giao điểm tia và tam giác bằng thuật toán Möller–Trumbore hoặc BVH Bounding Volume Hierarchy):
  - Bắn tiếp tia bóng (Shadow Ray) về phía nguồn sáng.
  - Bắn tia phản xạ (Reflection Ray) theo định luật phản xạ: $\vec{r} = \vec{d} - 2(\vec{d} \cdot \vec{n})\vec{n}$.
  - Bắn tia khúc xạ (Refraction Ray) theo định luật Snell: $n_1 \sin \theta_1 = n_2 \sin \theta_2$.
- **Ưu điểm**: Hình ảnh chân thực 100% theo quang học vật lý.
- **Nhược điểm**: Khối lượng tính toán khổng lồ ($O(\text{pixel} \times \text{tia} \times \text{số lần nảy})$).

---

## 2. Hàm Khoảng Cách Có Dấu (Signed Distance Functions - SDF) & Raymarching

### 2.1 Bản Chất Của SDF
- Một hàm toán học $d = \text{SDF}(p)$ nhận toạ độ 3D $p(x, y, z)$ và trả về khoảng cách ngắn nhất từ $p$ tới bề mặt vật thể:
  - $d > 0$: Điểm nằm bên ngoài vật thể.
  - $d = 0$: Điểm nằm ngay trên bề mặt vật thể.
  - $d < 0$: Điểm nằm sâu bên trong vật thể.
- **Công thức hình học cơ bản**:
  - Hình cầu tâm $O$ bán kính $r$: $d = \|p\| - r$.
  - Mặt phẳng ngang: $d = p_y$.
  - Hình hộp kích thước $b$: $d = \|\max(|p| - b, 0)\| + \min(\max(p_x - b_x, p_y - b_y, p_z - b_z), 0)$.

### 2.2 Thuật Toán Dò Mặt Bằng Bước Cầu (Sphere Tracing / Raymarching)
- Xuất phát từ Camera $p_0$, hướng tia $\vec{v}$.
- Tại mỗi bước $i$: Tính $d = \text{SDF}(p_i)$.
- Bước tiếp một đoạn an toàn đúng bằng $d$: $p_{i+1} = p_i + d \cdot \vec{v}$.
- Vì $d$ là khoảng cách tới bề mặt gần nhất, tia **chắc chắn không đâm xuyên qua bất kỳ vật thể nào** trong bán kính $d$.
- Lặp lại cho đến khi $d < \epsilon$ (chạm bề mặt) hoặc vượt quá khoảng cách tối đa (bay vào vô cực).

---

## 3. Đường Cong Bézier (Bézier Curves)

### 3.1 Công Thức & Ẩn Dụ De Casteljau
- **Bézier bậc 1 (Đoạn thẳng)**: $P(t) = (1 - t)P_0 + t P_1$ với $t \in [0, 1]$.
- **Bézier bậc 2 (Quadratic)**: Có 3 điểm kiểm soát $(P_0, P_1, P_2)$:
  $$P(t) = (1 - t)^2 P_0 + 2(1 - t)t P_1 + t^2 P_2$$
- **Bézier bậc 3 (Cubic)**: Dùng trong toàn bộ font chữ vector (TrueType / OpenType) và đường cong CSS Transition:
  $$P(t) = (1 - t)^3 P_0 + 3(1 - t)^2 t P_1 + 3(1 - t) t^2 P_2 + t^3 P_3$$
- **Thuật toán De Casteljau**: Vẽ đường cong bằng phép nội suy tuyến tính lồng nhau: Đoạn $P_0 P_1$ lấy điểm $Q_0(t)$, đoạn $P_1 P_2$ lấy điểm $Q_1(t)$. Đoạn $Q_0 Q_1$ lại tiếp tục nội suy lấy điểm $R(t)$ $\to$ Quỹ đạo của $R(t)$ tạo thành đường cong hoàn hảo.
