"""Offline checks for shared configuration, artifact writing and the CLI."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storm_real_pipeline.cli import anonymous_video_id, build_parser, discover_videos, merge_questions
from storm_real_pipeline.client import Settings, _format_content, load_prompt
from storm_real_pipeline.io_utils import write_json


class SharedTests(unittest.TestCase):
    def test_configuration_errors_do_not_include_secret(self):
        with patch.dict(os.environ, {"STORM_API_KEY": "private-placeholder"}, clear=True):
            with self.assertRaises(ValueError) as context:
                Settings.from_env()
            self.assertNotIn("private-placeholder", str(context.exception))
        settings = Settings("https://example.invalid/v1", "private-placeholder", "test-model")
        self.assertNotIn("private-placeholder", repr(settings))

    def test_invalid_numeric_configuration(self):
        environment = {"STORM_API_KEY": "placeholder", "STORM_BASE_URL": "https://example.invalid/v1",
                       "STORM_MODEL": "mock", "STORM_TIMEOUT": "nan"}
        with patch.dict(os.environ, environment, clear=True), self.assertRaises(ValueError):
            Settings.from_env()

    def test_vision_content_preserves_frame_order(self):
        content = _format_content(["[t=0s]", {"type": "image_base64", "data": "AA=="},
                                   {"type": "text", "text": "[t=1s]"}])
        self.assertEqual(content[0]["text"], "[t=0s]")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,AA==")
        self.assertEqual(content[2]["text"], "[t=1s]")

    def test_all_prompts_are_packaged_and_english(self):
        for name in ("segmentation.txt", "revisit.txt", "qa.txt"):
            prompt = load_prompt(name)
            self.assertGreater(len(prompt), 200)
            self.assertFalse(any("\u4e00" <= c <= "\u9fff" for c in prompt))

    def test_atomic_json_preserves_previous_value_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json(path, {"value": 1})
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_discovery_excludes_generated_videos_and_ids_are_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            (root / "source" / "input.mp4").touch()
            (root / "outputs").mkdir()
            (root / "outputs" / "generated.mp4").touch()
            videos = discover_videos(root, root / "outputs")
            self.assertEqual(len(videos), 1)
            identifier = anonymous_video_id(videos[0], root)
            self.assertRegex(identifier, r"^video_[a-f0-9]{16}$")
            self.assertNotIn("input", identifier)

    def test_merge_rejects_duplicate_ids_without_replacing_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.json", root / "b.json"]
            question = {
                "id": "episode_q01", "episode_id": "episode", "query_time": 1.0,
                "question_type": "factual_retrieval", "question_subtype": "tool_color",
                "video_evidence": "The tool is red in the sampled frame.",
                "question": "What color is the tool?", "options": ["red", "blue", "green", "white"],
                "answer_index": 0, "evidence_spans": [[1.0, 1.0]],
                "diagnostics": {"epistemic_status": "known", "uncertainty_sources": []},
                "diagnostic_rationale": {"volatility": "No visible change.", "uncertainty": "Visible color."},
            }
            for path in paths:
                write_json(path, {"episode_id": "episode", "duration_sec": 2.0,
                                  "sample_timestamps": [0.0, 1.0], "questions": [question]})
            output = root / "questions.jsonl"
            output.write_text("previous\n")
            with self.assertRaisesRegex(ValueError, "Duplicate question ID"):
                merge_questions(paths, output)
            self.assertEqual(output.read_text(), "previous\n")

    def test_default_pipeline_arguments(self):
        args = build_parser().parse_args(["run", "input.mp4"])
        self.assertEqual(args.window_sec, 300)
        self.assertEqual(args.sample_fps, 1)
        self.assertEqual(args.speed, 1.5)
        self.assertEqual(args.blank_sec, 0.2)
        self.assertIsNone(args.thinking)
        self.assertEqual(args.domain, "cook")

    def test_merge_uses_domain_validation_and_canonical_english_options(self):
        from storm_real_pipeline.domains import get_domain

        question = {
            "id": "music_episode_q01", "episode_id": "music_episode", "domain": "music",
            "query_time": 1.0, "question_type": "object_tracking", "question_subtype": "hidden_identity",
            "video_evidence": "Only part of the instrument is visible.",
            "question": "Which instrument was held during the omitted interval?",
            "options": ["Unknown from the visible evidence", "Violin", "Cello", "Guitar"],
            "answer_index": 0, "evidence_spans": [[1.0, 1.0]], "change_intensity": 1,
            "diagnostics": {"epistemic_status": "uncertain", "uncertainty_sources": ["out_of_view"]},
            "diagnostic_rationale": {"volatility": "The interval is omitted.", "uncertainty": "No direct observation."},
        }
        document = {"episode_id": "music_episode", "domain": "music",
                    "schema_version": get_domain("music").qa_schema, "duration_sec": 2.0,
                    "sample_timestamps": [0.0, 1.0], "questions": [question]}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "qa.json", Path(directory) / "questions.jsonl"
            write_json(source, document)
            self.assertEqual(merge_questions([source], output), 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["options"][0], "Cannot be determined")
            self.assertEqual(result["diagnostics"]["uncertainty_sources"], ["missing_observation"])
            self.assertEqual(result["domain"], "music")
            question["domain"] = "sports"
            write_json(source, document)
            with self.assertRaisesRegex(ValueError, "Invalid question"):
                merge_questions([source], output)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_domain_options_apply_to_every_stage(self):
        for domain in ("cook", "bike", "health", "music", "sports"):
            for command in ("run", "segment", "qa"):
                args = build_parser().parse_args([command, "input.mp4", "--domain", domain])
                self.assertEqual(args.domain, domain)
            args = build_parser().parse_args(["assemble", "segments.json", "--video", "input.mp4",
                                             "--domain", domain])
            self.assertEqual(args.domain, domain)

    def test_cross_domain_region_is_rejected_before_client_creation(self):
        from storm_real_pipeline.cli import main

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.touch()
            with patch("storm_real_pipeline.cli.make_client") as make_client:
                self.assertEqual(main(["run", str(video), "--domain", "music", "--region", "stove_area"]), 1)
                make_client.assert_not_called()

    def test_external_symlink_uses_its_logical_input_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "external.mp4"
            source.touch()
            inputs = root / "inputs"
            inputs.mkdir()
            link = inputs / "alias.mp4"
            link.symlink_to(source)
            self.assertRegex(anonymous_video_id(link, inputs), r"^video_[a-f0-9]{16}$")

    def test_failed_run_preserves_existing_merge(self):
        from storm_real_pipeline.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            source.touch()
            output_root = root / "output"
            output = output_root / "cook"
            output.mkdir(parents=True)
            export = output / "questions.jsonl"
            export.write_text("previous\n")
            with patch("storm_real_pipeline.cli.make_client"), patch(
                "storm_real_pipeline.segmentation.segment_source", side_effect=RuntimeError("test failure")
            ):
                code = main(["run", str(source), "--output-dir", str(output_root)])
            self.assertEqual(code, 1)
            self.assertEqual(export.read_text(), "previous\n")


if __name__ == "__main__":
    unittest.main()
