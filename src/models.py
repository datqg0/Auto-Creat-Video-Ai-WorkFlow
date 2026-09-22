"""Data model cho kịch bản video, dùng chung cho toàn pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Các loại visual mà visual_engine biết render
VisualType = Literal["title", "bullets", "chart", "code", "algorithm", "quote", "diagram"]


class Scene(BaseModel):
    narration: str = Field(..., description="Lời đọc cho scene này")
    visual_type: VisualType = "bullets"
    heading: str = ""
    # Nội dung tùy theo visual_type:
    # - bullets: danh sách gạch đầu dòng
    # - code: các dòng code
    # - quote: 1 câu trích dẫn (đặt ở bullets[0])
    bullets: list[str] = Field(default_factory=list)
    # chart: dữ liệu dạng {"labels": [...], "values": [...], "kind": "bar|line|pie"}
    chart: dict | None = None
    # code: ngôn ngữ để tô màu (chỉ hiển thị, không thực thi)
    code_language: str = "python"
    # algorithm: tên thuật toán để chọn animation Manim có sẵn
    algorithm: str = ""


class Script(BaseModel):
    topic: str
    title: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
