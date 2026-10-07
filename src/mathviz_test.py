"""Test nhanh thư viện mathviz: render PNG (mọi preset) + 1 mp4.

Chạy: python -m src.mathviz_test  (từ thư mục gốc dự án)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# cho phép chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mathviz import scenes  # type: ignore  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output" / "mathviz_preview"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    tests = {
        "function": scenes.function_scene(
            "Hàm sin(x)", lambda x: math.sin(x), subtitle="Sóng tuần hoàn cơ bản"
        ),
        "neural_net": scenes.neural_net_scene(
            "Mạng Neural", layers=(3, 5, 4, 2), subtitle="Lan truyền tiến"
        ),
        "bar_chart": scenes.bar_chart_scene(
            "So sánh tốc độ",
            values=[3, 7, 5, 9, 4],
            labels=["A", "B", "C", "D", "E"],
            subtitle="Đơn vị: điểm",
        ),
        "counter": scenes.counter_scene(
            "Người dùng", to_value=1000000, unit="tài khoản", subtitle="Tăng trưởng"
        ),
        "steps": scenes.steps_scene(
            "Quy trình học máy",
            steps=["Thu thập dữ liệu", "Huấn luyện mô hình", "Đánh giá", "Triển khai"],
        ),
        "terminal": scenes.terminal_scene(
            "Thực thi mô hình AI",
            code_lines=[
                "import torch",
                "model = AutoModel.from_pretrained('gpt-oss')",
                "inputs = tokenizer('Explain Feynman technique', return_tensors='pt')",
                "outputs = model.generate(**inputs, max_new_tokens=100)",
                "print('Inference completed in 12ms')",
            ],
            language="python",
            subtitle="Benchmark tốc độ suy luận",
        ),
        "architecture": scenes.architecture_flow_scene(
            "Kiến trúc Distributed Microservices",
            nodes=["Client App", "API Gateway", "Auth & Cache", "Worker Engine", "Database Cluster"],
            subtitle="Quy trình xử lý yêu cầu",
        ),
        "comparison": scenes.comparison_scene(
            "Độ phức tạp thuật toán",
            left_title="Bubble Sort (Naive)",
            left_items=["Độ phức tạp O(N^2)", "Chậm khi mảng lớn", "Nhiều thao tác swap thừa"],
            right_title="Merge Sort (Tối ưu)",
            right_items=["Độ phức tạp O(N log N)", "Chia để trị mượt mà", "Tối ưu cho dữ liệu lớn"],
            subtitle="Phân tích hiệu năng",
        ),
        "sorting": scenes.sorting_scene(
            "Sắp xếp nổi bọt", data=[5, 2, 8, 1, 9, 3, 7, 4], duration=6.0
        ),
    }

    for name, sc in tests.items():
        png = OUT / f"{name}.png"
        sc.render_png(png)
        print(f"PNG  {name:12s} -> {png}  (duration={sc.duration:.2f}s)")

    # render 1 mp4 để chắc chắn video pipeline chạy
    mp4 = OUT / "function.mp4"
    try:
        scenes.function_scene(
            "Hàm sin(x)", lambda x: math.sin(x), subtitle="Sóng tuần hoàn", duration=5.0
        ).render_mp4(mp4)
        print(f"MP4  function     -> {mp4}")
    except Exception as e:  # noqa: BLE001
        print(f"MP4 lỗi (bỏ qua nếu chưa có moviepy): {e}")


if __name__ == "__main__":
    main()
