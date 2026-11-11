"""Data model cho kịch bản video, dùng chung cho toàn pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Các loại visual mà visual_engine biết render
VisualType = Literal[
    "title", "bullets", "chart", "code", "algorithm", "quote", "diagram", "animation", "challenge", "mascot", "math_equation", "math"
]

RendererName = Literal["manim", "motion_graphics", "broll", "static"]


class EvidenceClaim(BaseModel):
    """Một khẳng định có thể truy ngược về nguồn dùng trong kịch bản."""

    claim_id: str
    claim: str
    source_url: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    scene_indices: list[int] = Field(default_factory=list)


class TimelineCue(BaseModel):
    """Mốc tương đối để renderer đồng bộ hình ảnh với lời đọc."""

    at: float = Field(ge=0.0, le=1.0)
    action: str
    target: str = ""


class SceneContract(BaseModel):
    """Hợp đồng giữa biên tập nội dung và renderer của một scene."""

    teaching_goal: str
    preferred_renderer: RendererName = "static"
    fallback_renderers: list[RendererName] = Field(default_factory=lambda: ["motion_graphics", "static"])
    objects: list[str] = Field(default_factory=list)
    focus: str = ""
    timeline: list[TimelineCue] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    max_on_screen_chars: int = Field(default=70, ge=12, le=180)
    safe_margin_percent: float = Field(default=8.0, ge=0.0, le=30.0)
    subtitle_safe_area_percent: float = Field(default=18.0, ge=0.0, le=35.0)


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
    # algorithm: tên thuật toán để gợi ý animation phù hợp
    algorithm: str = ""
    # từ khóa tiếng Anh để tự tìm ảnh minh họa nền cho scene
    image_query: str = ""
    # từ khóa tiếng Anh để tự tìm VIDEO b-roll minh họa (footage động trên mạng).
    # Nếu có, compositor dùng video làm nền + chữ overlay -> trực quan, sinh động hơn ảnh tĩnh.
    video_query: str = ""
    # animation: cấu hình clip động render bằng thư viện mathviz.
    # {"preset": "function|neural_net|bar_chart|sorting|counter|steps", ...tham số}
    animation: dict | None = None
    # visual_prompt: mô tả chi tiết thuật toán/hình ảnh trực quan để giao cho LLM chuyên code (DeepSeek R1 / Qwen Coder) sinh animation
    visual_prompt: str = ""
    # Được Visual Director bổ sung sau khi script vượt content gate.
    contract: SceneContract | None = None
    claim_ids: list[str] = Field(default_factory=list)

    @field_validator("animation", "chart", mode="before")
    @classmethod
    def _coerce_dict_or_none(cls, v):
        # LLM đôi khi trả chuỗi mô tả thay vì object -> bỏ qua thay vì crash
        return v if isinstance(v, dict) else None

    @field_validator("bullets", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [str(v)]


class Exercise(BaseModel):
    """Một bài toán thực tế đặt ở cuối video để người xem tự luyện."""

    question: str
    hint: str = ""
    answer: str = ""


class Script(BaseModel):
    topic: str
    title: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    # Public URLs extracted from the compact fact pack and added to the description.
    references: list[str] = Field(default_factory=list)
    # Bài toán thực tế đặt ở cuối video (LLM sinh nhiều bài cho người xem luyện)
    exercises: list[Exercise] = Field(default_factory=list)
    evidence: list[EvidenceClaim] = Field(default_factory=list)
    content_score: float | None = Field(default=None, ge=0.0, le=100.0)

    @field_validator("exercises", mode="before")
    @classmethod
    def _coerce_exercises(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for e in v:
            if isinstance(e, str):
                out.append({"question": e})
            elif isinstance(e, dict) or isinstance(e, Exercise):
                out.append(e)
        return out

