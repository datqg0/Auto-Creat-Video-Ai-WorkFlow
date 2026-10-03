# Kế hoạch nâng cấp Animation

Dựa trên phân tích repo tham chiếu `pdoom-video` (TypeScript + three.js + GPU shader) và
đối chiếu với stack hiện tại (Python: Pillow + Matplotlib + Manim + MoviePy + ffmpeg).

> **Sự thật phũ phàng:** Không thể đạt đúng mức pdoom vì họ render GPU shader, ray-march 3D,
> mỗi frame là hàm toán thuần và chỉnh thủ công từng scene theo lyric. Stack của bạn phải tự động
> hoàn toàn cho mọi chủ đề. Nhưng Manim có thể đạt ~70% cảm giác đó nếu dùng đúng — vì Manim sinh
> ra chính để làm animation kiểu 3Blue1Brown.

---

## Tầng 1 — Dễ, tác động ngay (không đụng kiến trúc)

Rủi ro ~0, thấy khác biệt ngay. Nên làm trước.

### 1.1. Dùng easing mạnh trong mathviz
- **Vấn đề:** `mathviz/easing.py` đã có `smooth`, `ease_out_back`, `ease_out_elastic`, `ease_out_bounce`
  nhưng các preset dùng chủ yếu `smooth` (lướt đều, mềm oặt).
- **Việc cần làm:**
  - Thêm `ease_out_expo`, `ease_in_expo`, `ease_in_out_expo` vào `easing.py` (hiện đang thiếu expo).
  - Thêm `spring_step` (dao động giảm chấn) mô phỏng `springStep` của pdoom.
  - Sửa các preset trong `mathviz/scenes.py` dùng easing mạnh cho entrance (`ease_out_expo` / `back`).
- **Triết lý pdoom:** *"strong eases, holds, then snaps"* — giữ yên → bùng nổ nhanh → giữ yên.
  KHÔNG lướt đều kiểu screensaver.

### 1.2. Nâng Ken Burns (scene tĩnh bớt nhàm)
- **Vấn đề:** `compositor._ken_burns` chỉ zoom thẳng 1.0 → 1.06, rất nhạt.
- **Việc cần làm:**
  - Thêm pan (dịch ngang/dọc) kèm zoom, hướng pan chọn ngẫu nhiên-ổn-định theo seed scene.
  - Áp easing (`ease_in_out`) cho chuyển động thay vì tuyến tính.

### 1.3. Micro-motion cho ảnh tĩnh
- Overlay film grain / bụi động nhẹ lên ảnh tĩnh (phối với post film ở Tầng 2).
- Làm ảnh tĩnh "thở" thay vì chết cứng.

---

## Tầng 2 — Trung bình, tạo khác biệt lớn

Cần thêm thư viện / sửa logic, nhưng đây là phần tạo "linh hồn".

### 2.1. Sync nhịp nhạc (QUAN TRỌNG NHẤT)
- **Đây là core của pdoom:** `hook.ts` tính `timeOfBeat()`, `beatAt()` → mọi chuyển động chốt vào beat.
- **Việc cần làm:**
  - Dùng `librosa` phân tích nhạc nền → lấy BPM, mảng beats, downbeats.
  - Snap điểm cắt scene (transition) vào downbeat gần nhất.
  - Punch-in / hit đúng vào beat ở câu hook.
- **Rủi ro:** Trung bình (thêm dependency `librosa` + sửa logic transition trong `compositor.compose`).

### 2.2. Punch-in on beat (hook)
- Ở câu hook (đã có ~5s), thêm zoom-punch + shake nhẹ đúng beat.
- Dùng easing `ease_out_expo` cho cú punch (vọt nhanh rồi dừng).

### 2.3. Palette kỷ luật + glow
- **pdoom:** bảng màu nghiêm ngặt (ink/bone/signal/ember), chỉ màu accent được glow (>0.85 linear).
- **Việc cần làm:**
  - Gom `mathviz/theme.py` + `visual_engine.py` về 1 bộ màu nhất quán (bớt 7 màu xoay vòng).
  - Chỉ accent được bloom/glow.
- Phối với post-processing film ở ffmpeg `_finalize` (bloom + vignette + grain + tone curve).

---

## Tầng 3 — Khó, cần nhiều công

Đẩy chất lượng lên mức "chuyên nghiệp" nhưng tốn thời gian.

### 3.1. Kinetic typography bằng Manim
- **pdoom:** chữ là nhân vật chính — bay vào theo từng từ, co giãn width/weight theo nốt nhạc,
  gõ như token, dập như con dấu.
- **Việc cần làm:**
  - Ưu tiên Manim cho scene heading/hook thay vì ảnh Pillow tĩnh.
  - Chữ bay vào theo từ (word-by-word reveal), scale punch, slide-in có ease.

### 3.2. Camera move trong Manim
- Dùng `self.camera.frame.animate` cho punch-in / orbit / dolly.
- Tạo cảm giác không gian 3D-ish mà vẫn 2D.

### 3.3. Motif xuyên suốt
- **pdoom:** "spark" (tia lửa) chạy nối các scene, tạo mạch liền.
- **Việc cần làm (tùy chọn):** một motif đồ họa nhỏ lặp lại giữa các scene để video có "mạch".

---

## Lưu ý giới hạn (không bê được)

- **GPU raymarched / shader scenes** (shoggoth, paperclips, loss landscape 3D...) — KHÔNG tái tạo
  được rẻ trong MoviePy/Manim. Bỏ qua, tập trung vào easing + beat-sync + post + palette + đẩy Manim.
- **Adaptive sub-frame motion blur** — GPU-specific. Manim fps 60 đã mô phỏng phần nào độ mượt.

---

## Thứ tự thực thi đề xuất

1. **Tầng 1 toàn bộ** (easing mạnh + Ken Burns + micro-motion) — nền chắc, rủi ro ~0.
2. **Tầng 2.1** (sync nhịp nhạc) — tác động "sống động" lớn nhất.
3. **Tầng 2.2 + 2.3** (punch-in + palette/glow).
4. **Tầng 3** khi có thời gian.

> Giữ nguyên bất biến bảo mật: AI code chạy subprocess env rỗng, KHÔNG eval/exec in-process.
> Giữ chuỗi fallback pycode/custom/manim. Verify bằng py_compile + render thử sau mỗi thay đổi.
