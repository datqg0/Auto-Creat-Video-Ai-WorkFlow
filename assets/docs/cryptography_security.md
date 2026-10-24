# Tài Liệu Chuẩn Xác: Mật Mã Học & An Toàn Thông Tin (Cryptography & Security)

Tài liệu tham chiếu chuẩn xác 100% về mặt toán học và khoa học máy tính (Zero Hallucination). Mọi kịch bản (Script Writer) và Hoạt hoạ (Animation Runner) bắt buộc phải tuân theo các giá trị số và bước giải dưới đây.

---

## 1. Mật Mã Bất Đối Xứng RSA (RSA Asymmetric Cryptosystem)

### 1.1 Khái niệm & Nguyên lý
RSA dựa trên tính bất đối xứng của bài toán phân tích thừa số nguyên tố: Nhân hai số nguyên tố rất dễ, nhưng tìm thừa số nguyên tố của một số khổng lồ là bài toán bất khả thi về mặt tính toán (NP-hard).

### 1.2 Walkthrough Số Liệu Mẫu Chuẩn Xác (BẮT BUỘC DÙNG BỘ SỐ NÀY)
Để video minh họa trực quan, dễ hiểu, tránh số quá lớn nhưng vẫn chuẩn xác 100% modulo:
- **Bước 1: Chọn 2 số nguyên tố bí mật**:
  - $p = 11$, $q = 17$.
- **Bước 2: Tính Modulus công khai $n$**:
  - $n = p \times q = 11 \times 17 = 187$.
- **Bước 3: Tính hàm Euler Totient $\phi(n)$**:
  - $\phi(n) = (p - 1)(q - 1) = (11 - 1)(17 - 1) = 10 \times 16 = 160$.
- **Bước 4: Chọn Khóa Công Khai (Public Exponent $e$)**:
  - Chọn $e = 7$ sao cho $1 < e < \phi(n)$ và $\gcd(e, \phi(n)) = \gcd(7, 160) = 1$.
  - Cặp Khóa Công Khai: $(e, n) = (7, 187)$.
- **Bước 5: Tính Khóa Bí Mật (Private Exponent $d$)**:
  - Tìm nghịch đảo modulo: $e \times d \equiv 1 \pmod{\phi(n)} \implies 7 \times d \equiv 1 \pmod{160}$.
  - Giải phương trình Diophantine: $7 \times 23 = 161 = 1 \times 160 + 1$.
  - Do đó: $d = 23$.
  - Khóa Bí Mật: $d = 23$.
- **Bước 6: Mã hóa thông điệp (Encryption)**:
  - Thông điệp số gốc $M = 5$ (với $M < n$).
  - Bản mã: $C = M^e \pmod{n} = 5^7 \pmod{187}$.
  - Tính toán: $5^7 = 78,125$.
  - $78,125 = 417 \times 187 + 146 \implies C = 146$.
  - *(Lưu ý: Không bao giờ được tính nhầm ra 56; kết quả chính xác 100% là 146!)*.
- **Bước 7: Giải mã bản mã (Decryption)**:
  - Bản rõ phục hồi: $M = C^d \pmod{n} = 146^{23} \pmod{187}$.
  - Theo Định lý Euler: $146^{23} \pmod{187} = 5$.
  - Khôi phục chính xác thông điệp gốc $M = 5$.

### 1.3 Phân Tích Độ Phức Tạp & Ứng Dụng
- **Độ phức tạp**: Mã hóa/Giải mã dùng Modular Exponentiation (Bình phương và nhân) mất $O(\log e)$ hoặc $O(\log d)$. Tấn công phá mã bằng phân tích thừa số nguyên tố hiện đại (General Number Field Sieve) mất thời gian dưới hàm mũ: $O(\exp((c + o(1))(\ln n)^{1/3}(\ln \ln n)^{2/3}))$.
- **Ứng dụng**: HTTPS/TLS Handshake, Chữ ký số GPG, SSH Keys, Blockchain identities.

---

## 2. Giao Thức Trao Đổi Khóa Diffie-Hellman (Diffie-Hellman Key Exchange)

### 2.1 Ẩn Dụ Trực Quan & Toán Học
- **Ẩn dụ màu sắc**: Hai người (Alice & Bob) bắt đầu bằng một màu sơn công khai (vàng). Mỗi người chọn một giọt màu bí mật (Alice chọn đỏ, Bob chọn lam). Hai bên pha màu và trao đổi màu đã pha qua kênh công cộng. Kẻ nghe lén (Eve) nhìn thấy màu pha nhưng không thể tách ngược lại (bài toán pha màu 1 chiều). Alice và Bob thêm giọt bí mật của mình vào màu nhận được và cùng thu được màu bí mật giống hệt nhau (nâu sẫm).
- **Bộ số toán học chuẩn**:
  - Số nguyên tố công khai $p = 23$, căn nguyên thủy $g = 5$.
  - Khóa bí mật Alice: $a = 6 \implies A = g^a \pmod{p} = 5^6 \pmod{23} = 15625 \pmod{23} = 8$.
  - Khóa bí mật Bob: $b = 15 \implies B = g^b \pmod{p} = 5^{15} \pmod{23} = 19$.
  - Alice và Bob trao đổi công khai $A = 8$ và $B = 19$.
  - Alice tính bí mật chung: $s = B^a \pmod{p} = 19^6 \pmod{23} = 47045881 \pmod{23} = 2$.
  - Bob tính bí mật chung: $s = A^b \pmod{p} = 8^{15} \pmod{23} = 35184372088832 \pmod{23} = 2$.
  - Bí mật chung $s = 2$ trùng khớp hoàn toàn!

---

## 3. Hàm Băm Mật Mã Một Chiều SHA-256 (Secure Hash Algorithm 256-bit)

### 3.1 Cấu Trúc Khối & Nguyên Lý
- **Cấu trúc Merkle–Damgård**: Thông điệp được đệm bit (Padding: bit '1' theo sau là các số '0' và 64-bit độ dài) để chia thành các khối 512-bit ($M^{(1)}, M^{(2)}, \dots$).
- **Trạng thái nội tại**: 8 thanh ghi 32-bit ($H_0 \dots H_7$) khởi tạo bằng căn bậc hai của 8 số nguyên tố đầu tiên (2, 3, 5, 7, 11, 13, 17, 19).
- **Vòng lặp nén**: Mỗi khối trải qua 64 vòng lặp (Rounds) với các hàm logic thao tác trên bit:
  - $\text{Ch}(x, y, z) = (x \land y) \oplus (\neg x \land z)$ (Choose)
  - $\text{Maj}(x, y, z) = (x \land y) \oplus (x \land z) \oplus (y \land z)$ (Majority)
  - $\Sigma_0(x) = \text{ROTR}^2(x) \oplus \text{ROTR}^{13}(x) \oplus \text{ROTR}^{22}(x)$
  - $\Sigma_1(x) = \text{ROTR}^6(x) \oplus \text{ROTR}^{11}(x) \oplus \text{ROTR}^{25}(x)$
- **Hiệu ứng Thác Đổ (Avalanche Effect)**: Chỉ cần thay đổi 1 bit đơn lẻ ở đầu vào, ít nhất 50% số bit ở chuỗi băm đầu ra 256-bit sẽ đảo trạng thái ngẫu nhiên.

---

## 4. Mã Sửa Lỗi Reed-Solomon trong Mã QR (QR Code Error Correction)

### 4.1 Cơ chế Tự Sửa Lỗi Khi Rách Nát
- **Nguyên lý Galois Field**: Mã QR sử dụng trường hữu hạn $GF(2^8) = GF(256)$ với đa thức nguyên thủy $p(x) = x^8 + x^4 + x^3 + x^2 + 1$.
- **Mức độ chịu lỗi (Error Correction Levels)**:
  - Level L (Low): Phục hồi tới 7% diện tích bị mất.
  - Level M (Medium): Phục hồi tới 15% diện tích bị mất.
  - Level Q (Quartile): Phục hồi tới 25% diện tích bị mất.
  - Level H (High): Phục hồi tới **30% diện tích** bị rách nát, che khuất hoặc vẽ đè logo.
- **Đa thức hội chứng (Syndromes)**: Khi quét mã QR bị khuyết, thuật toán Berlekamp–Massey hoặc Euclid mở rộng tìm đa thức định vị lỗi (Error Locator Polynomial) $\Lambda(x)$, sau đó dùng thuật toán Forney để tính giá trị lỗi tại từng byte và tái tạo nguyên vẹn 100% dữ liệu gốc.
