# Tài Liệu Chuẩn Xác: Công Nghệ Đời Sống & Mạng Máy Tính (Networking & Everyday Tech)

Tài liệu chuẩn xác phục vụ kịch bản và hoạt hoạ công nghệ thường thức (phong cách Kurzgesagt, Veritasium).

---

## 1. Giao Thức TCP/IP & Bắt Tay 3 Bước (Three-Way Handshake)

### 1.1 Quá Trình Thiết Lập Kết Nối
- Để hai máy tính (Client và Server) bắt đầu truyền dữ liệu tin cậy qua Internet:
  1. **Bước 1 (SYN)**: Client gửi gói tin có cờ `SYN` (Synchronize) với số thứ tự ngẫu nhiên ban đầu $ISN_C = x$.
  2. **Bước 2 (SYN-ACK)**: Server nhận được, gửi lại gói tin có cờ `SYN` và `ACK` (Acknowledgment):
     - Xác nhận đã nhận của Client: $\text{ACK} = x + 1$.
     - Khởi tạo số thứ tự của Server: $ISN_S = y$.
  3. **Bước 3 (ACK)**: Client gửi gói tin có cờ `ACK`: $\text{ACK} = y + 1$.
- Sau 3 bước, kênh kết nối tin cậy hai chiều chính thức mở. Nếu một gói tin bị rơi trên đường cáp quang, Client sẽ tự động phát hiện và gửi lại (Retransmission).

---

## 2. Định Vị GPS & Thuyết Tương Đối Của Einstein

### 2.1 Con Số 38 Microseconds Bắt Buộc Phải Bù Trừ Mỗi Ngày
- Mỗi vệ tinh GPS bay trên quỹ đạo độ cao **20,200 km** với vận tốc khoảng **14,000 km/h**.
- Để định vị chính xác vị trí ô tô/điện thoại trong vòng vài mét, đồng hồ nguyên tử trên vệ tinh phải có độ chính xác đến nanosecond.
- **Thuyết Tương Đối Hẹp (Special Relativity - Chuyển động nhanh)**:
  - Vệ tinh bay rất nhanh ($v \approx 3.9 \text{ km/s}$) so với mặt đất $\implies$ Theo hiệu ứng giãn nở thời gian do vận tốc, đồng hồ trên vệ tinh **chạy chậm hơn** đồng hồ dưới đất khoảng **7 microseconds mỗi ngày**.
- **Thuyết Tương Đối Rộng (General Relativity - Trọng lực yếu)**:
  - Vệ tinh ở trên cao 20,200 km, nơi trường trọng lực Trái Đất yếu hơn rất nhiều so với mặt đất $\implies$ Thời gian trôi nhanh hơn $\implies$ Đồng hồ trên vệ tinh **chạy nhanh hơn** đồng hồ dưới đất khoảng **45 microseconds mỗi ngày**.
- **Hiệu ứng ròng (Net Effect)**:
  $$+45 \mu\text{s} - 7 \mu\text{s} = +\mathbf{38 \text{ microseconds/ngày}}$$
- **Hậu quả nếu không bù trừ**:
  - Ánh sáng truyền đi với vận tốc $c \approx 300,000 \text{ km/s}$.
  - Sai số khoảng cách tích lũy mỗi ngày: $38 \times 10^{-6} \text{ s} \times 300,000 \text{ km/s} \approx \mathbf{11.4 \text{ kilômét/ngày}}$!
  - Chỉ sau 1 ngày, bản đồ GPS sẽ chỉ vị trí bạn trôi lệch ra giữa biển hoặc thành phố khác! Do đó các kỹ sư bắt buộc phải lập trình đồng hồ vệ tinh chạy chậm đi tần số từ 10.23 MHz xuống 10.22999999543 MHz trước khi phóng lên quỹ đạo.

---

## 3. Nén Ảnh JPEG & Biến Đổi Cosin Rời Rạc (DCT)

### 3.1 Vì Sao File Ảnh Nén Thu Nhỏ 10 Lần Mà Mắt Vẫn Thấy Đẹp?
- **Chuyển đổi không gian màu**: Chuyển từ RGB sang $Y C_b C_r$ (trong đó $Y$ là độ sáng Luma, $C_b, C_r$ là màu sắc Chroma). Mắt người nhạy với độ sáng hơn màu sắc $\implies$ Nén giảm mẫu màu sắc một nửa (Chroma Subsampling 4:2:0) mà mắt không nhận ra.
- **Biến đổi DCT trên từng khối 8x8 pixels**: Chuyển đổi 64 giá trị pixel từ miền không gian sang miền tần số không gian. Tần số thấp (nền phẳng, mảng lớn) tập trung ở góc trên bên trái; tần số cao (nét sắc, chi tiết li ti) nằm ở góc dưới bên phải.
- **Lượng tử hóa (Quantization)**: Chia các hệ số tần số cao cho bảng lượng tử lớn để làm tròn về 0. Sau đó quét zigzag và nén Huffman để lưu trữ siêu gọn.
