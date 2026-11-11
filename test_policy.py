"""Unit test nhẹ: không render, không gọi mạng và không cần media dependencies."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.manim_validator import validate_manim_code
from src.models import Scene, Script
from src.publish_policy import evaluate_publish_policy
from src.visual_director import plan_visuals


class TestProductionPolicies(unittest.TestCase):
    def test_manim_validator_accepts_scene(self):
        code = "from manim import *\nclass NeonScene(Scene):\n    def construct(self):\n        self.add(Circle())\n"
        self.assertTrue(validate_manim_code(code).allowed)

    def test_manim_validator_blocks_filesystem_and_processes(self):
        code = "import os\nfrom manim import *\nclass NeonScene(Scene):\n    def construct(self):\n        os.system('echo unsafe')\n"
        result = validate_manim_code(code)
        self.assertFalse(result.allowed)
        self.assertTrue(result.errors)

    def test_visual_director_routes_algorithm_to_manim(self):
        script = Script(
            topic="Binary Search",
            title="Binary Search hoạt động thế nào?",
            references=["https://example.com/reference"],
            scenes=[Scene(narration="Ví dụ thuật toán binary search từng bước.", visual_type="algorithm")],
        )
        with tempfile.TemporaryDirectory() as temp:
            planned = plan_visuals(script, Path(temp) / "storyboard.json")
        self.assertEqual(planned.scenes[0].contract.preferred_renderer, "manim")

    def test_publish_policy_requires_all_gates(self):
        manifest = {"quality_passed": True, "title": "T", "topic": "X"}
        approved = evaluate_publish_policy(
            manifest,
            {"passed": True, "blocking_errors": []},
            {"passed": True, "errors": []},
        )
        self.assertTrue(approved.approved)
        rejected = evaluate_publish_policy(
            manifest,
            {"passed": False, "blocking_errors": [{"name": "video"}]},
            {"passed": True, "errors": []},
        )
        self.assertFalse(rejected.approved)


if __name__ == "__main__":
    unittest.main()
