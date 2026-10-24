# Tài Liệu Chuẩn Xác: Kiến Trúc Máy Tính & Phần Cứng (Computer Architecture & Hardware)

Tài liệu chuẩn xác phục vụ sinh kịch bản và hoạt hoạ phần cứng máy tính (phong cách Kurzgesagt, Branch Education).

---

## 1. Hệ Thống Phân Cấp Bộ Nhớ CPU (CPU Cache Hierarchy)

### 1.1 Bảng Trễ Chuẩn Xác (Latency & Bandwidth Numbers Every Engineer Should Know)
Mọi kịch bản và bảng số liệu so sánh bắt buộc phải dùng bảng số liệu chuẩn này:

| Tầng bộ nhớ | Vị trí | Dung lượng điển hình | Độ trễ (Cycles) | Độ trễ thực tế (Nanoseconds) | Ẩn dụ khoảng cách (Nếu 1 cycle = 1 giây) |
|---|---|---|---|---|---|
| **CPU Registers** | Ngay trong lõi CPU | ~1 - 2 KB | 1 cycle | **0.5 - 1.0 ns** | Cầm sẵn sách trên tay |
| **L1 Cache** | Trong từng nhân riêng | ~32 - 64 KB / core | 4 - 5 cycles | **1.0 - 1.5 ns** | V với tay lấy trên bàn |
| **L2 Cache** | Trong từng nhân | ~512 KB - 1 MB / core | 12 - 14 cycles | **3 - 4 ns** | Đứng dậy ra giá sách góc phòng |
| **L3 Cache** | Dùng chung các nhân | ~16 - 96 MB shared | 40 - 50 cycles | **10 - 15 ns** | Đi bộ xuống sảnh tòa nhà |
| **RAM (DDR5)** | Bo mạch chủ | 16 - 64 GB | ~150 - 200 cycles | **60 - 100 ns** | Đi xe máy 10 cây số sang thành phố khác |
| **NVMe SSD** | Khe cắm PCIe | 512 GB - 2 TB | ~100,000 cycles | **10 - 50 µs** | Đi máy bay từ Hà Nội sang New York |
| **HDD cơ** | Cáp SATA | 2 - 8 TB | ~10,000,000 cycles | **5 - 10 ms** | Đi bộ 3 năm vòng quanh Trái Đất |

### 1.2 Nguyên Lý Cục Bộ (Locality) & Dòng Cache (Cache Line)
- Bộ nhớ cache nạp dữ liệu theo từng khối cố định **64 bytes (Cache Line)**, không nạp từng byte đơn lẻ.
- **Tính cục bộ không gian (Spatial Locality)**: Nếu truy cập mảng tại phần tử `A[0]`, toàn bộ các phần tử lân cận `A[1]...A[15]` sẽ được kéo sẵn vào cùng 1 dòng cache 64-byte $\to$ Duyệt mảng tuần tự nhanh hơn duyệt con trỏ ngẫu nhiên (Linked List) tới 10–50 lần.
- **Tính cục bộ thời gian (Temporal Locality)**: Một biến vừa được sử dụng trong vòng lặp nhiều khả năng sẽ tiếp tục được dùng lại trong tương lai gần $\to$ Giữ lại trong L1 cache.

---

## 2. Kiến Trúc Lõi GPU vs CPU & Hiện Tượng Warp Divergence

### 2.1 Bản Chất Khác Nhau Giữa CPU và GPU
- **CPU (Độ trễ thấp - Latency-Oriented)**:
  - Có 8–24 nhân cực mạnh.
  - Phần lớn diện tích silicon dành cho: Bộ nhớ đệm L1/L2/L3 khổng lồ, Mạch dự đoán rẽ nhánh (Branch Predictor), và Điều khiển thực thi lệnh không theo thứ tự (Out-of-Order Execution).
  - Tối ưu cho việc thực thi nhanh một luồng lệnh phức tạp tuần tự.
- **GPU (Băng thông cao - Throughput-Oriented)**:
  - Có hàng nghìn đến hàng chục nghìn nhân tính toán nhỏ (ALU).
  - Ghép thành các cụm gọi là **Streaming Multiprocessors (SM)**.
  - Phần lớn diện tích silicon dành trực tiếp cho ALU tính toán dấu phẩy động (FP32/FP16/INT8).
  - Tối ưu cho xử lý dữ liệu song song cực lớn: Ma trận đồ họa pixel, Tensor AI.

### 2.2 Hiện Tượng Lệch Nhịp Phân Nhánh (Warp Divergence)
- Trong GPU NVIDIA, các luồng (threads) được nhóm thành từng đơn vị **32 threads gọi là 1 Warp**.
- Một Warp vận hành theo mô hình **SIMT (Single Instruction, Multiple Threads)**: Cả 32 luồng bắt buộc phải cùng thực thi một câu lệnh phần cứng tại cùng một chu kỳ xung nhịp.
- **Khi xuất hiện lệnh rẽ nhánh `if-else`**:
  - Nếu trong 32 luồng, có 16 luồng thỏa điều kiện `if` và 16 luồng nhảy vào `else`:
  - GPU **KHÔNG THỂ** cho 2 nhánh chạy song song.
  - Bước 1: 16 luồng của nhánh `if` thực thi, 16 luồng còn lại bị vô hiệu hóa (Masked OFF - ngồi chơi).
  - Bước 2: 16 luồng của nhánh `else` thực thi, 16 luồng ban đầu bị vô hiệu hóa (Masked OFF).
  - **Hậu quả**: Thời gian thực thi tăng gấp đôi, hiệu năng tính toán bị tụt giảm 50% chỉ vì một câu lệnh `if-else` phân nhánh!
