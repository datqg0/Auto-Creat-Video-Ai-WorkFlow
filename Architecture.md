# Architecture

## 1. Project Overview

### Goal

> Auto makeing video and upload to youtube

### Users

> Me

### Main Features

* Feature 1: cread insign and write it completely
* Feature 2: write code 
* Feature 3:

---

# 2. System Architecture

## High-level Architecture

```text
                    ┌──────────────┐
                    │    Client    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  API Server  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │   Auth   │ │ AI Model │ │ Database │
        └──────────┘ └──────────┘ └──────────┘
```

### Components

| Component  | Responsibility | Technology |
| ---------- | -------------- | ---------- |
| Client     |                |            |
| API Server |                |            |
| AI Model   |                |            |
| Database   |                |            |
| Cache      |                |            |

---

# 3. Data Flow

## Main Request

```text
User
 ↓
Client
 ↓
API
 ↓
Validation
 ↓
Preprocessing
 ↓
AI Model
 ↓
Postprocessing
 ↓
Database
 ↓
Response
 ↓
Client
```

### Step-by-step

1. User sends ...
2. Client sends request to ...
3. API validates ...
4. Data is transformed by ...
5. Model performs ...
6. Result is ...
7. Database stores ...
8. API returns ...

---

# 4. Control Flow

> Thành phần nào gọi thành phần nào?

```text
Client
  │
  ▼
API
  │
  ├──→ Auth
  │
  ├──→ Database
  │
  └──→ AI Service
          │
          └──→ Model
```

---

# 5. Data

## Input

```text
Input:
- type:
- format:
- size:
- source:
```

## Output

```text
Output:
- type:
- format:
- destination:
```

## Database

### Tables / Collections

```text
User
 ├── id
 ├── name
 └── created_at

Prediction
 ├── id
 ├── user_id
 ├── input
 ├── result
 └── created_at
```

---

# 6. State

> Hệ thống cần nhớ những trạng thái nào?

```text
User
 ↓
logged_out
 ↓
logged_in
 ↓
processing
 ↓
completed
```

### Important State

* User authentication state:
* Processing state:
* Model state:
* Job state:

---

# 7. Failure Cases

> Điều gì có thể hỏng?

| Failure           | Effect | Handling |
| ----------------- | ------ | -------- |
| API down          |        |          |
| Database down     |        |          |
| AI model error    |        |          |
| Invalid input     |        |          |
| Network timeout   |        |          |
| GPU unavailable   |        |          |
| Too many requests |        |          |

### Example

```text
AI Model
   │
   ├── Success → Return result
   │
   └── Failure
          ↓
       Retry?
          ↓
       Fallback?
          ↓
       Error response
```

---

# 8. Performance

## Expected Load

```text
Users:
Requests / second:
Data / day:
Concurrent requests:
```

## Bottlenecks

> Thành phần nào có khả năng trở thành bottleneck?

```text
Client
 ↓
API
 ↓
Database       ← ?
 ↓
AI Model       ← ?
 ↓
Storage        ← ?
```

### Questions

* Model inference mất bao lâu?
* Database query mất bao lâu?
* Có cần caching không?
* Có cần batching không?
* Có cần asynchronous processing không?

---

# 9. Scaling

## Current

```text
1 API Server
1 Database
1 AI Worker
```

## If traffic increases 10x

```text
             ┌── API Server
Client ──────┼── API Server
             └── API Server
                    │
                  Queue
                    │
             ┌──────┼──────┐
             ▼      ▼      ▼
           Worker Worker Worker
             │      │      │
             └──────┼──────┘
                    ▼
                  Model
```

### What changes?

* API:
* Database:
* AI inference:
* Queue:
* Cache:
* Storage:

---

# 10. Security

### Authentication

> Làm thế nào xác thực user?

### Authorization

> User A có thể truy cập dữ liệu của User B không?

### Input Validation

> User có thể gửi dữ liệu độc hại không?

### Secrets

> API keys được lưu ở đâu?

```text
❌ Hardcode API key

API_KEY = "abc123"
```

```text
✅ Environment variable

API_KEY = os.getenv("API_KEY")
```

---

# 11. Observability

> Làm sao biết hệ thống đang hoạt động tốt hay không?

### Logs

* API request
* Error
* Model inference
* Database error

### Metrics

* Requests / second
* Latency
* Error rate
* CPU usage
* GPU usage
* Memory usage

### Alerts

```text
Error rate > X%
       ↓
     Alert
```

---

# 12. Deployment

## Environment

```text
Development
     ↓
Staging
     ↓
Production
```

### Infrastructure

```text
Frontend:
Backend:
Database:
AI Model:
Storage:
Docker:
Cloud:
```

---

# 13. Trade-offs

> Không có architecture hoàn hảo. Ghi lại những quyết định quan trọng.

## Decision 1

### Problem

...

### Options

```text
A:
B:
C:
```

### Decision

...

### Why?

...

### Trade-off

```text
Pros:
-

Cons:
-
```

---

# 14. Why This Architecture?

> Đây là phần quan trọng nhất.

Viết bằng lời của mình:

```text
Tôi chọn architecture này vì ...

Nếu hệ thống nhỏ thì ...

Nếu traffic tăng thì ...

Nếu AI model trở thành bottleneck thì ...

Nếu database chết thì ...

Nếu cần scale lên 100x thì ...
```

---

# 15. Future Improvements

```text
[ ] Add caching
[ ] Add queue
[ ] Add monitoring
[ ] Improve model inference
[ ] Add load balancing
[ ] Improve database
[ ] Add automated deployment
```

---

# 16. Things I Don't Understand Yet

> Đây là phần cực kỳ quan trọng khi học.

* [ ] Tại sao cần Redis?
* [ ] Tại sao cần Queue?
* [ ] Tại sao model server tách khỏi API?
* [ ] Khi nào dùng async?
* [ ] Khi nào cần Load Balancer?
* [ ] Database bottleneck nằm ở đâu?
* [ ] Làm thế nào scale GPU?

---

# 17. Lessons Learned

Sau khi hoàn thành project, ghi lại:

### Before

> Ban đầu tôi nghĩ ...

### After

> Sau khi làm tôi hiểu rằng ...

### Biggest Problem

> Vấn đề lớn nhất là ...

### Important Insight

> Điều quan trọng nhất tôi học được là ...
