"""Test preview for upgraded visual engine."""
from pathlib import Path
from src.models import Scene
from src.visual_engine import render_scene

OUT = Path("output/preview_upgrade")
OUT.mkdir(parents=True, exist_ok=True)

# Test 1: Title
sc_title = Scene(
    narration="Chào mừng bạn đến với chuyên đề giải mã kiến trúc Microservices và cơ chế đồng bộ dữ liệu.",
    visual_type="title",
    heading="Kiến trúc Microservices & Distributed Systems",
)
render_scene(sc_title, OUT / "0_title.png")

# Test 2: Bullets with NO image (triggers Companion Tech Card)
sc_bullets = Scene(
    narration="Khi xây dựng hệ thống phân tán, có ba thách thức lớn: tính sẵn sàng, độ trễ mạng và phân vùng dữ liệu.",
    visual_type="bullets",
    heading="Ba Thách Thức Trong Hệ Thống Phân Tán",
    bullets=[
        "Độ trễ mạng giữa các microservices (RPC / gRPC overhead)",
        "Tính nhất quán dữ liệu theo định lý CAP (Eventual Consistency)",
        "Quản lý distributed transaction và saga pattern",
        "Khả năng mở rộng theo chiều ngang khi chịu tải cao",
    ],
)
render_scene(sc_bullets, OUT / "1_bullets.png")

# Test 3: Code scene with syntax highlighting
sc_code = Scene(
    narration="Dưới đây là cài đặt hàm kiểm tra chu kỳ trong đồ thị bằng thuật toán DFS.",
    visual_type="code",
    heading="Thuật Toán DFS Kiểm Tra Chu Kỳ",
    code_language="python",
    bullets=[
        "def has_cycle(graph, node, visited, rec_stack):",
        "    visited.add(node)",
        "    rec_stack.add(node)",
        "    for neighbor in graph[node]:",
        "        if neighbor not in visited:",
        "            if has_cycle(graph, neighbor, visited, rec_stack):",
        "                return True",
        "        elif neighbor in rec_stack:",
        "            return True  # Phát hiện chu kỳ cạnh ngược",
        "    rec_stack.remove(node)",
        "    return False",
    ],
)
render_scene(sc_code, OUT / "2_code.png")

print("Rendered preview upgrade scenes successfully!")
