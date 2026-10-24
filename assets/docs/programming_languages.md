# Tài Liệu Chuẩn Xác: Ngôn Ngữ Lập Trình & Runtimes (Programming Languages & Runtimes)

Tài liệu chuẩn xác phục vụ kịch bản và hoạt hoạ so sánh ngôn ngữ lập trình (phong cách Fireship, Low Level JavaScript).

---

## 1. Vì Sao Python Chậm Hơn C Gấp 50 Đến 100 Lần?

### 1.1 Cơ Chế Đằng Sau Vỏ Bọc `a + b`
- **Trong C**:
  - `int a = 5; int b = 10; int c = a + b;`
  - Trình biên dịch dịch thẳng ra đúng 1 lệnh máy Assembly: `ADD EAX, EBX`. CPU thực thi trong 1 chu kỳ xung nhịp (~0.3 nanosecond).
- **Trong Python**:
  - `a = 5; b = 10; c = a + b`
  - Mọi thứ trong Python là một `PyObject` nằm trên Heap:
    - Chứa trường tham chiếu đếm `ob_refcnt`.
    - Chứa con trỏ kiểu dữ liệu `ob_type` (`&PyLong_Type`).
    - Chứa mảng các chữ số `ob_digit`.
  - Phép cộng `+` phải trải qua:
    1. Kiểm tra kiểu của `a` và `b` xem có phải số nguyên hay chuỗi / danh sách (Dynamic Type Checking).
    2. Tìm kiếm hàm nạp chồng toán tử `__add__` trong bảng hàm.
    3. Trích xuất giá trị số nguyên từ cấu trúc `PyObject`.
    4. Cấp phát một `PyObject` hoàn toàn mới trên Heap để chứa kết quả.
    5. Cập nhật con trỏ và đếm tham chiếu (Reference Counting).
  - Tốn hàng chục đến hàng trăm câu lệnh máy thay vì chỉ 1 lệnh như C!

---

## 2. JavaScript V8 Engine & JIT Compiler (Ignition + TurboFan)

### 2.1 Hidden Classes (Shapes) & Inline Caching
- JavaScript là ngôn ngữ động, đối tượng có thể thêm bớt thuộc tính tùy ý.
- V8 Engine tối ưu bằng cách tạo ngầm các **Hidden Class (Shape)**:
  - Khi tạo `p = {x: 1}`, V8 gán Shape `C0`.
  - Khi thêm `p.y = 2`, V8 chuyển tiếp sang Shape `C1` với offset bộ nhớ cố định cho `x` (offset 0) và `y` (offset 1).
- **Inline Caching**: Nếu một hàm `getCoords(p) { return p.x + p.y; }` liên tục nhận các object có cùng Shape `C1`, trình tối ưu **TurboFan** sẽ dịch hàm này thành mã máy trực tiếp đọc thẳng offset bộ nhớ mà không cần tra cứu thuộc tính $\to$ Tốc độ tiệm cận C++!

---

## 3. Go Goroutines vs Hệ Thống Luồng Hệ Điều Hành (OS Threads)

### 3.1 So Sánh Chi Phí Bộ Nhớ & Chuyển Ngữ Cảnh (Context Switch)
- **OS Thread (Java, C++, Rust)**:
  - Được quản lý bởi Kernel hệ điều hành.
  - Mỗi Thread chiếm **1 MB đến 2 MB bộ nhớ Stack** cố định ngay khi khởi tạo.
  - Chuyển ngữ cảnh (Context Switch) đòi hỏi chuyển chế độ người dùng sang nhân (User to Kernel space), lưu toàn bộ 16 thanh ghi CPU $\to$ Mất 1 - 2 microsecond. Một máy chủ chỉ chịu được vài nghìn luồng trước khi cạn RAM.
- **Go Goroutine**:
  - Được quản lý hoàn toàn ở không gian người dùng (User space) bởi Go Runtime Scheduler (M:N scheduler).
  - Mỗi Goroutine khởi đầu với **chỉ 2 KB bộ nhớ Stack** (tự động co giãn khi cần).
  - Chuyển ngữ cảnh chỉ lưu 3 thanh ghi (PC, SP, DX) $\to$ Mất vài chục nanosecond.
  - Một máy chủ có thể chạy đồng thời **hàng triệu Goroutine** cùng lúc mà không sập nguồn!

---

## 4. Rust: Quyền Sở Hữu (Ownership) & Bộ Kiểm Tra Vay Mượn (Borrow Checker)

### 4.1 Quản Lý Bộ Nhớ Không Cần Garbage Collector
- Trong Rust, không có Garbage Collector (như Go/Java) gây dừng chương trình (Stop-the-world pause), cũng không cần lập trình viên tự `malloc`/`free` thủ công (như C/C++) dễ gây lỗi rò rỉ hoặc Use-After-Free.
- **3 Nguyên tắc vàng của Ownership**:
  1. Mỗi giá trị trong Rust có một biến sở hữu duy nhất (Owner).
  2. Tại một thời điểm, chỉ có DUY NHẤT một Owner.
  3. Khi Owner đi ra khỏi phạm vi (out of scope), giá trị tự động được giải phóng ngay lập tức (RAII).
- **Quy tắc Borrow Checker (Chống Race Condition tại compile-time)**:
  - Bạn có thể có **vô số tham chiếu đọc bất biến (`&T`)**, HOẶC **DUY NHẤT 1 tham chiếu ghi khả biến (`&mut T`)**, nhưng tuyệt đối không bao giờ được phép có cả hai cùng lúc!
