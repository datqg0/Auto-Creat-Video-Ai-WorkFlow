"""Smoke test kiểm tra toàn bộ các module và hàm cốt lõi của pipeline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class TestPipelineModules(unittest.TestCase):
    def test_config_load(self):
        from src.config import CONFIG, apply_mode
        self.assertIn("llm", CONFIG)
        self.assertIn("visual", CONFIG)
        self.assertIn("series", CONFIG)
        self.assertEqual(CONFIG["series"]["default_episodes"], 5)
        self.assertIn("providers", CONFIG["llm"])
        # Kiểm tra timeout Gemini đã được nâng lên >= 60s
        for p in CONFIG["llm"]["providers"]:
            if p["name"].startswith("gemini"):
                self.assertGreaterEqual(p["timeout"], 60)
        self.assertEqual(apply_mode("documentary"), "documentary")
        self.assertEqual(CONFIG["target_duration_seconds"], 720)
        self.assertEqual((CONFIG["visual"]["width"], CONFIG["visual"]["height"]), (1920, 1080))
        self.assertEqual(CONFIG["modes"]["documentary"]["min_duration_seconds"], 600)
        apply_mode("long")

    def test_visual_engine_log_and_text_w(self):
        import logging
        from PIL import Image, ImageDraw, ImageFont
        from src.visual_engine import log, _text_w, _clean_heading
        self.assertIsInstance(log, logging.Logger)
        img = Image.new("RGB", (100, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        w = _text_w(draw, "Xin chào thế giới 123", font)
        self.assertGreater(w, 0)
        self.assertEqual(_clean_heading("Hook: Bí mật của AI"), "Bí mật của AI")

    def test_script_writer_prompts(self):
        from src.script_writer import (
            _load_settings,
            _build_long_prompt,
            _build_short_prompt,
            _build_long_part2_prompt,
        )
        s = _load_settings()
        p1 = _build_long_prompt(s, "Test Topic", "Research Notes", None)
        self.assertIn("Test Topic", p1)

        ps = _build_short_prompt(s, "Test Topic", "Research Notes", None)
        self.assertIn("Test Topic", ps)
        self.assertIn("binary_search", ps)
        self.assertIn("dp_grid", ps)
        self.assertIn("Không dùng sorting bars nếu chủ đề không phải sắp xếp", ps)

        # Kiểm tra lỗi x_{t-1} đã được escape thành công không còn NameError
        p2 = _build_long_part2_prompt(
            s, "Test Topic", "Research Notes", None,
            {"title": "Test Title", "scenes": [{"heading": "H1", "narration": "N1"}]},
            4, 6, 400, 2
        )
        self.assertIn("x_{t-1}", p2)
        self.assertIn("Test Title", p2)

    def test_short_algorithm_variety_and_routing(self):
        from src.onjsdev_visualizer import AlgorithmSimulator, normalize_algorithm_name
        from src.pipeline import _short_visualizer_algorithm

        values = [9, 2, 7, 1, 5]
        for simulator in (
            AlgorithmSimulator.bubble_sort,
            AlgorithmSimulator.quick_sort,
            AlgorithmSimulator.merge_sort,
            AlgorithmSimulator.insertion_sort,
            AlgorithmSimulator.selection_sort,
            AlgorithmSimulator.heap_sort,
        ):
            steps = simulator(values)
            self.assertEqual(steps[-1].array, sorted(values), simulator.__name__)

        search = AlgorithmSimulator.binary_search(values, target=7)
        self.assertIsNotNone(search[-1].found_index)
        path = AlgorithmSimulator.pathfinding("a_star")
        self.assertEqual(path[-1].path[0], path[-1].start)
        self.assertEqual(path[-1].path[-1], path[-1].goal)

        self.assertEqual(normalize_algorithm_name("A* pathfinding"), "a_star")
        self.assertEqual(
            _short_visualizer_algorithm({"preset": "onjsdev_visualizer", "algorithm": "binary_search"}, ""),
            "binary_search",
        )
        self.assertIsNone(_short_visualizer_algorithm({}, "Công thức Pi Chudnovsky"))
        self.assertIsNone(_short_visualizer_algorithm({}, "Git merge conflict và cách giải quyết"))

    def test_documentary_mode_is_manual_slow_and_animation_limited(self):
        from src.config import CONFIG, apply_mode
        from src.models import Scene, Script
        from src.pipeline import _resolve_requested_duration, _uses_automatic_series
        from src.script_writer import _build_long_prompt, _enforce_animation_ratio, _load_settings
        from src.tts import _mode_tts_config

        apply_mode("documentary")
        settings = _load_settings()
        self.assertTrue(settings.is_documentary)
        self.assertGreaterEqual(settings.duration, 600)
        self.assertEqual(settings.wpm, 115)
        self.assertEqual(settings.n_exercises, 0)
        self.assertEqual(_mode_tts_config()["edge_rate"], "-18%")
        self.assertFalse(_uses_automatic_series("documentary"))
        self.assertTrue(_uses_automatic_series("long"))
        self.assertEqual(_resolve_requested_duration("documentary", 120), 600)
        self.assertEqual(_resolve_requested_duration("documentary", 900), 900)

        prompt = _build_long_prompt(settings, "Lịch sử Unix", "Research Notes", None)
        self.assertIn("phim tài liệu", prompt.lower())
        self.assertIn("Không tạo bài tập", prompt)
        self.assertIn("không vượt 20%", prompt)

        scenes = [
            Scene(
                narration=f"Đây là lời kể đủ dài cho cảnh lịch sử công nghệ thứ {i}.",
                visual_type="animation",
                heading=f"Mốc {i}",
                animation={"preset": "steps"},
            )
            for i in range(10)
        ]
        script = Script(topic="Unix", title="Lịch sử Unix", scenes=scenes)
        limited = _enforce_animation_ratio(script, settings)
        self.assertLessEqual(sum(sc.visual_type == "animation" for sc in limited.scenes), 2)

        workflow = Path(".github/workflows/create-video.yml").read_text(encoding="utf-8")
        self.assertIn("- documentary", workflow)
        self.assertEqual(workflow.count('cron: "'), 2)
        self.assertIn('cron: "0 0 */3 * *"', workflow)
        self.assertIn('cron: "0 12 * * *"', workflow)
        self.assertIn('MODE="short"', workflow)
        apply_mode("long")

    def test_ai_code_runner_security_and_config(self):
        from src.ai_code_runner import _clean_and_validate_code, CONFIG
        self.assertIsNotNone(CONFIG)
        # Kiểm tra code an toàn
        safe_code = "```python\nimport matplotlib.pyplot as plt\nplt.plot([1, 2])\n```"
        self.assertIsNotNone(_clean_and_validate_code(safe_code, "matplotlib"))

        # Kiểm tra code tấn công đọc file credentials bị chặn
        attack_code = "```python\nwith open('credentials/token.json') as f: pass\n```"
        self.assertIsNone(_clean_and_validate_code(attack_code, "matplotlib"))

        attack_git = "```python\nwith open('.git/config') as f: pass\n```"
        self.assertIsNone(_clean_and_validate_code(attack_git, "matplotlib"))

    def test_llm_parsing_and_rate_limits(self):
        from src.llm import _retry_after, _is_quota_or_auth_error, _validate_response_text

        # 1. Test parse thời gian chờ của Groq
        groq_err = Exception("Rate limit reached on ITPM: Limit 7000. Please try again in 19.311428571s.")
        self.assertAlmostEqual(_retry_after(groq_err), 19.311428571, places=2)

        # 2. Test parse thời gian chờ của Gemini
        gemini_err = Exception("{'retryDelay': '12670s'}")
        self.assertEqual(_retry_after(gemini_err), 12670.0)

        # 3. Test parse retry in XhYmZs
        gemini_human = Exception("Please retry in 3h31m10.32s.")
        self.assertAlmostEqual(_retry_after(gemini_human), 3 * 3600 + 31 * 60 + 10.32, places=1)

        # 4. Test quota check
        self.assertTrue(_is_quota_or_auth_error(Exception("Generate content requests per day limit exceeded")))
        self.assertTrue(_is_quota_or_auth_error(Exception("Error code: 410")))
        self.assertFalse(_is_quota_or_auth_error(Exception("Normal syntax error at line 4100")))

        # 5. Test response text validation
        self.assertEqual(_validate_response_text("test", "Đoạn văn hợp lệ rất dài giải thích về rate limit..."), "Đoạn văn hợp lệ rất dài giải thích về rate limit...")
        with self.assertRaises(Exception):
            _validate_response_text("test", "The account behind this API key doesn't have enough credits. Please top up")

    def test_db_functions(self):
        from src.db import recent_topics, get_next_pending_episode
        topics = recent_topics(30)
        self.assertIsInstance(topics, list)


if __name__ == "__main__":
    unittest.main()

