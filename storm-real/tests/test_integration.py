"""Exercise real sampling and encoding with mock VLM responses, without network."""

import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storm_real_pipeline.cli import main
from storm_real_pipeline.client import load_prompt
from storm_real_pipeline.domains import DOMAINS, get_domain

FFMPEG = os.environ.get("STORM_FFMPEG") or shutil.which("ffmpeg")
HAS_OPENCV = importlib.util.find_spec("cv2") is not None


class MockVLM:
    model = "offline-test-model"

    def __init__(self, domain="cook"):
        self.calls = []
        self.domain = domain
        self.region = {"cook": "stove_area", "bike": "chain_area",
                       "health": "patient_mannequin_area", "music": "instrument_playing_area",
                       "sports": "basketball_court_area"}[domain]

    def call(self, prompt, content, **kwargs):
        self.calls.append(prompt)
        if prompt == load_prompt("segmentation.txt", self.domain):
            segments = []
            intervals = [(0, 20, True), (20, 22, False), (22, 42, True),
                         (42, 44, False), (44, 64, True), (64, 66, False),
                         (66, 86, True), (86, 90, False)]
            for start, end, usable in intervals:
                segments.append({
                    "start_sec": start, "end_sec": end,
                    "region": self.region if usable else "other_area",
                    "segment_role": "observation" if usable else "navigation",
                    "activity_type": "observe" if usable else "navigate",
                    "change_types": [],
                    "description": "Observe the workspace" if usable else "Leave the workspace",
                })
            return json.dumps(segments)
        if prompt != load_prompt("qa.txt", self.domain):
            return json.dumps({"visits": [
                {"start_sec": 0, "end_sec": 20, "label": "First observation"},
                {"start_sec": 22, "end_sec": 42, "label": "Second observation"},
                {"start_sec": 44, "end_sec": 64, "label": "Third observation"},
            ]})
        metadata = json.loads(content[0]["text"])
        timestamp = min(10.0, math.floor(metadata["episode_duration"]) - 1.0)
        types = ["current_state", "factual_retrieval", "history_aggregation", "object_tracking",
                 "state_change", "temporal_reasoning"]
        questions = []
        for index in range(10):
            uncertain = index >= 8
            question_type = types[index // 2] if index < 8 else types[index - 4]
            questions.append({
                "query_time": timestamp,
                "question_type": question_type,
                "question_subtype": "visible_marker",
                "video_evidence": "A synthetic marker is visible in the sampled frame.",
                "question": f"Which marker is supported for the synthetic query {index + 1}?",
                "options": (["Cannot be determined", "red", "blue", "green"]
                            if uncertain else ["red", "blue", "green", "yellow"]),
                "answer_index": 0,
                "evidence_spans": [[timestamp, timestamp]],
                "diagnostics": {"epistemic_status": "uncertain" if uncertain else "known",
                                "uncertainty_sources": ["ambiguous_evidence"] if uncertain else []},
                "diagnostic_rationale": {"volatility": "Synthetic separated visits.",
                                         "uncertainty": "A deterministic test response."},
            })
        return json.dumps(questions)


@unittest.skipUnless(FFMPEG and HAS_OPENCV, "Requires FFmpeg and OpenCV")
class PipelineIntegrationTests(unittest.TestCase):
    def test_five_domain_pipelines_resume_and_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "private_source_name.mp4"
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", "testsrc2=size=96x64:rate=10:duration=90", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
            output_root = root / "outputs"
            for domain in DOMAINS:
                with self.subTest(domain=domain):
                    output = output_root / domain
                    client = MockVLM(domain)
                    args = ["run", str(video), "--domain", domain, "--output-dir", str(output_root),
                            "--max-frames-per-call", "128"]
                    with patch("storm_real_pipeline.cli.make_client", return_value=client):
                        self.assertEqual(main(args), 0)
                        first_calls = len(client.calls)
                        self.assertGreaterEqual(first_calls, 3)
                        self.assertEqual(main(args), 0)
                        self.assertEqual(len(client.calls), first_calls)
                    plans = list((output / "episodes").glob("*.plan.json"))
                    self.assertEqual(len(plans), 1)
                    plan = json.loads(plans[0].read_text())
                    qa = json.loads(next((output / "qa").glob("*.json")).read_text())
                    segmentation = json.loads(next((output / "segments").glob("*.json")).read_text())
                    self.assertEqual(segmentation["schema_version"], get_domain(domain).segmentation_schema)
                    self.assertEqual(plan["domain"], domain)
                    self.assertEqual(plan["region"], client.region)
                    self.assertEqual(qa["domain"], domain)
                    self.assertEqual(qa["schema_version"], get_domain(domain).qa_schema)
                    self.assertEqual(plan["episode_id"], qa["episode_id"])
                    self.assertTrue(qa["episode_id"].startswith(domain + "_video_"))
                    self.assertLessEqual(plan["total_sec"], 60)
                    self.assertEqual(len(qa["questions"]), 10)
                    for question in qa["questions"]:
                        self.assertEqual(question["domain"], domain)
                        self.assertTrue(all(end <= question["query_time"] for _, end in question["evidence_spans"]))
                        expected = sum(visit["output_start_sec"] <= question["query_time"]
                                       for visit in plan["visits"])
                        self.assertEqual(question["change_intensity"], max(1, expected))
                    self.assertEqual(len((output / "questions.jsonl").read_text().splitlines()), 10)
                    self.assertEqual(json.loads((output / "run_summary.json").read_text())["failures"], [])
            for path in output_root.rglob("*.json"):
                text = path.read_text()
                self.assertNotIn("private_source_name", text)
                self.assertNotIn(str(root), text)
            merged = output_root / "questions.jsonl"
            self.assertEqual(main(["merge", *[str(output_root / domain / "qa") for domain in DOMAINS],
                                   "--output", str(merged)]), 0)
            rows = [json.loads(line) for line in merged.read_text().splitlines()]
            self.assertEqual(len(rows), 50)
            self.assertEqual({row["domain"] for row in rows}, set(DOMAINS))
            self.assertEqual(len({row["id"] for row in rows}), 50)


if __name__ == "__main__":
    unittest.main()
