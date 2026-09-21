"""Contract, planning, resumption, and optional real FFmpeg tests."""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storm_real_pipeline import revisit
from storm_real_pipeline.domains import DOMAIN_NAMES, get_domain


def metadata(length=20.0, visits=4, domain="cook"):
    spec = get_domain(domain)
    region, activity = {
        "cook": ("stove_area", "cook"),
        "bike": ("tire_tube_area", "inflate_deflate"),
        "health": ("test_kit_area", "operate_test_kit"),
        "music": ("instrument_playing_area", "play"),
        "sports": ("basketball_ball_drill_area", "dribble_control"),
    }[domain]
    segments, cursor = [], 0.0
    for index in range(visits):
        segments.append({"start_sec": cursor, "end_sec": cursor + length,
                         "region": region,
                         "segment_role": "observation" if index % 2 == 0 else "interaction",
                         "activity_type": "observe" if index % 2 == 0 else activity,
                         "change_types": [], "description": "A visible cooking area."})
        cursor += length
        segments.append({"start_sec": cursor, "end_sec": cursor + 1.0,
                         "region": "other_area", "segment_role": "navigation",
                         "activity_type": "navigate", "change_types": [],
                         "description": "Movement between visits."})
        cursor += 1.0
    return {"schema_version": spec.segmentation_schema, "domain": domain,
            "source_video_id": "video_test", "video_id": "video_test_window000",
            "interval_convention": "[start_sec, end_sec)",
            "processed_interval": [0.0, cursor], "segments": segments,
            "video_path": "/private/participant-name/source.mp4"}


class FakeClient:
    model = "test-model"

    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def call(self, prompt, contents, **kwargs):
        self.calls.append((prompt, contents, kwargs))
        if self.error:
            raise self.error
        return self.response


class RevisitTests(unittest.TestCase):
    def setUp(self):
        self.meta = metadata()
        self.segments, _, _ = revisit.validate_segmentation_meta(self.meta)
        self.scoped = [segment for segment in self.segments if segment["region"] == "stove_area"]

    def test_only_english_schema_and_finite_contiguous_intervals(self):
        for change in (lambda m: m.update(schema_version="stream_eqa_activity_segments_v5_no_entities"),
                       lambda m: m["segments"][0].update(region="legacy_region"),
                       lambda m: m["segments"][0].update(end_sec=float("nan")),
                       lambda m: m["segments"][0].update(start_sec=True),
                       lambda m: m["segments"][1].update(start_sec=22.0)):
            value = metadata()
            change(value)
            with self.assertRaises(ValueError):
                revisit.validate_segmentation_meta(value)

    def test_observation_and_interaction_candidates_and_no_required_hidden_change(self):
        visits = revisit.pick_visits_greedy(self.scoped, 45.0, 0.2)
        valid = revisit.validate_visits(visits, self.scoped, 1.5, 0.2, 60.0)
        self.assertGreaterEqual(len(valid), 3)
        self.assertEqual({visit["candidate_role"] for visit in valid}, {"observation", "interaction"})
        self.assertAlmostEqual(revisit.estimate_output_duration(valid, 1.5, 0.2), 43.6)
        self.assertEqual([revisit._duration(visit) for visit in valid], [20.0, 20.0, 20.0, 4.5])
        self.assertEqual(sum(item["change_segment_count"] for item in revisit._hidden_changes(valid, self.segments)), 0)

    def test_greedy_honors_nondefault_timing_and_hard_limit(self):
        for target, maximum, speed, blank, expected in [(30.0, 31.0, 2.0, 0.5, 26.0), (37.0, 40.0, 1.5, 0.2, 35.4), (50.0, 52.0, 1.5, 1.0, 42.0)]:
            visits = revisit.pick_visits_greedy(self.scoped, target, blank, speed, maximum)
            self.assertGreaterEqual(len(visits), 3)
            self.assertLessEqual(revisit.estimate_output_duration(visits, speed, blank), maximum + 1e-8)
            self.assertAlmostEqual(revisit.estimate_output_duration(visits, speed, blank), expected)
            self.assertTrue(all(3.0 <= revisit._duration(visit) <= 20.0 for visit in visits))

    def test_invalid_or_duplicate_candidate_intervals_are_dropped(self):
        raw = [{"start_sec": 0.0, "end_sec": 5.0}, {"start_sec": 10.0, "end_sec": 15.0},
               {"start_sec": 19.0, "end_sec": 24.0}, {"start_sec": float("inf"), "end_sec": 50.0},
               {"start_sec": 21.0, "end_sec": 41.0}, {"start_sec": 42.0, "end_sec": 62.0}]
        valid = revisit.validate_visits(raw, self.scoped, 1.5, 0.2, 60.0)
        self.assertEqual([visit["start_sec"] for visit in valid], [0.0, 21.0, 42.0])

    def test_prompt_uses_runtime_parameters(self):
        client = FakeClient('{"visits": []}')
        revisit._llm_visits(client, self.scoped, "stove_area", {"target_sec": 31.0, "max_total_sec": 40.0, "speed": 2.0, "blank_sec": 0.5}, 84.0)
        prompt = client.calls[0][0]
        self.assertIn("31.0 seconds", prompt)
        self.assertIn("40.0 seconds", prompt)
        self.assertIn("2.0 times", prompt)
        self.assertNotIn("45", prompt)
        self.assertNotIn("$", prompt)

    def test_frame_rounding_is_explicit_and_bounded(self):
        visits = [{"start_sec": 0.0, "end_sec": 3.1}, {"start_sec": 5.0, "end_sec": 8.2}, {"start_sec": 10.0, "end_sec": 13.3}]
        indexed, blank_frames, count = revisit._timeline(visits, 29.97, 1.5, 0.2)
        self.assertEqual(blank_frames, 5)
        self.assertAlmostEqual(indexed[-1]["output_end_sec"], count / 29.97)
        self.assertLessEqual(count / 29.97, revisit.estimate_output_duration(visits, 1.5, 0.2))
        with self.assertRaises(ValueError):
            revisit._timeline(visits, 10.0, 1.5, 0.001)

    def _fixture(self, directory):
        root = Path(directory)
        segmentation, source, output = root / "seg.json", root / "source.mp4", root / "episode.mp4"
        segmentation.write_text(json.dumps(self.meta), encoding="utf-8")
        source.write_bytes(b"test source")
        return segmentation, source, output

    @staticmethod
    def _fake_assemble(source, visits, info, blank_frames, parameters, temporary, executable):
        result = temporary / "episode.mp4"
        result.write_bytes(b"test generated video")
        count = sum(visit["output_frame_count"] for visit in visits) + blank_frames * (len(visits) - 1)
        return result, {**info, "frame_count": count, "duration_sec": count / info["fps"]}

    def test_both_selectors_run_and_plan_omits_private_metadata(self):
        client = FakeClient(json.dumps({"visits": [{"start_sec": 0.0, "end_sec": 3.0}, {"start_sec": 21.0, "end_sec": 24.0}, {"start_sec": 42.0, "end_sec": 45.0}], "rationale": "/private/secret participant-name"}))
        info = {"width": 64, "height": 48, "fps": 30.0, "frame_count": 2520, "duration_sec": 84.0}
        with tempfile.TemporaryDirectory() as directory, patch.object(revisit, "_probe_video", return_value=info), patch.object(revisit, "_ffmpeg", return_value="ffmpeg"), patch.object(revisit, "_assemble", side_effect=self._fake_assemble):
            segmentation, source, output = self._fixture(directory)
            plan = revisit.construct_episode(segmentation, source, output, client)
            self.assertEqual(plan["selection_source"], "greedy")
            self.assertEqual({item["method"] for item in plan["selection_candidates"]}, {"llm", "greedy"})
            self.assertEqual(len(client.calls), 1)
            self.assertNotIn("/private/", json.dumps(plan))
            self.assertNotIn("participant-name", json.dumps(plan))
            self.assertNotIn("source_video", plan)
            self.assertNotIn("source_segmentation", plan)
            self.assertEqual(plan["source_video_id"], "video_test")
            self.assertEqual(plan["episode_id"], "video_test_window000_revisit")
            self.assertEqual(plan["source_sha256"], revisit._sha256(source))
            self.assertNotIn("change_intensity", plan)
            self.assertIn("hidden_change_segment_count", plan)
            self.assertAlmostEqual(plan["visits"][-1]["output_end_sec"], plan["total_sec"])
            self.assertEqual(json.loads(output.with_suffix(".plan.json").read_text()), plan)

    def test_resume_checks_parameters_segmentation_and_video_integrity(self):
        info = {"width": 64, "height": 48, "fps": 30.0, "frame_count": 2520, "duration_sec": 84.0}
        with tempfile.TemporaryDirectory() as directory, patch.object(revisit, "_ffmpeg", return_value="ffmpeg"), patch.object(revisit, "_assemble", side_effect=self._fake_assemble):
            segmentation, source, output = self._fixture(directory)
            with patch.object(revisit, "_probe_video", return_value=info):
                plan = revisit.construct_episode(segmentation, source, output, None)
            def probe(path, decode=False):
                return {**info, "frame_count": plan["output_frame_count"], "duration_sec": plan["total_sec"]} if path == output else info
            with patch.object(revisit, "_probe_video", side_effect=probe):
                self.assertEqual(revisit.construct_episode(segmentation, source, output, None), plan)
                for update in ({"domain": "bike"}, {"source_schema_version": "old"}):
                    output.with_suffix(".plan.json").write_text(json.dumps({**plan, **update}))
                    with self.assertRaises(FileExistsError):
                        revisit.construct_episode(segmentation, source, output, None)
                output.with_suffix(".plan.json").write_text(json.dumps(plan))
                with self.assertRaises(FileExistsError):
                    revisit.construct_episode(segmentation, source, output, None, target_sec=44.0)
                segmentation.write_text(json.dumps(self.meta, indent=2))
                with self.assertRaises(FileExistsError):
                    revisit.construct_episode(segmentation, source, output, None)
                segmentation.write_text(json.dumps(self.meta))
                output.write_bytes(b"corrupted")
                with self.assertRaises(FileExistsError):
                    revisit.construct_episode(segmentation, source, output, None)

    def test_llm_failure_is_redacted_and_assembly_failure_preserves_old_output(self):
        client = FakeClient(error=RuntimeError("secret request and private path"))
        info = {"width": 64, "height": 48, "fps": 30.0, "frame_count": 2520, "duration_sec": 84.0}
        with tempfile.TemporaryDirectory() as directory, patch.object(revisit, "_probe_video", return_value=info), patch.object(revisit, "_ffmpeg", return_value="ffmpeg"):
            segmentation, source, output = self._fixture(directory)
            with patch.object(revisit, "_assemble", side_effect=self._fake_assemble):
                plan = revisit.construct_episode(segmentation, source, output, client)
            self.assertEqual(plan["llm_status"], "failed")
            self.assertEqual(plan["llm_failure_reason"], "RuntimeError")
            self.assertNotIn("secret", json.dumps(plan))
            original = output.read_bytes()
            with patch.object(revisit, "_assemble", side_effect=RuntimeError("encoding failure")):
                with self.assertRaises(RuntimeError):
                    revisit.construct_episode(segmentation, source, output, None, overwrite=True)
            self.assertEqual(output.read_bytes(), original)
            self.assertFalse(list(Path(directory).glob(".storm-revisit-*")))

    def test_numeric_parameter_validation_precedes_io(self):
        for kwargs in ({"speed": 0}, {"speed": float("nan")}, {"target_sec": True}, {"blank_sec": -0.1}, {"fade_sec": float("inf")}, {"target_sec": 61}, {"max_total_sec": 3}, {"region": "legacy_region"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                revisit.construct_episode(Path("missing.json"), Path("missing.mp4"), Path("out.mp4"), None, **kwargs)

    def test_source_hash_and_anonymous_ids_are_checked_before_video_probe(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(revisit, "_probe_video") as probe:
            segmentation, source, output = self._fixture(directory)
            for update in ({"source_sha256": "0" * 64}, {"source_video_id": "../private"}, {"video_id": "/private/source"}):
                segmentation.write_text(json.dumps({**self.meta, **update}))
                with self.subTest(update=update), self.assertRaises(ValueError):
                    revisit.construct_episode(segmentation, source, output, None)
            probe.assert_not_called()

    def test_domain_metadata_schema_rejection_and_cook_compatibility(self):
        for domain in DOMAIN_NAMES:
            document = metadata(domain=domain)
            with self.subTest(domain=domain):
                revisit.validate_segmentation_meta(document, domain=domain)
                wrong_domain = "bike" if domain == "cook" else "cook"
                with self.assertRaises(ValueError):
                    revisit.validate_segmentation_meta(document, domain=wrong_domain)
                document["domain"] = wrong_domain
                with self.assertRaises(ValueError):
                    revisit.validate_segmentation_meta(document, domain=domain)
        legacy_cook = metadata()
        del legacy_cook["domain"]
        revisit.validate_segmentation_meta(legacy_cook)

    def test_two_visits_work_for_each_non_cook_domain_but_not_cook(self):
        info = {"width": 64, "height": 48, "fps": 30.0,
                "frame_count": 360, "duration_sec": 12.0}
        for domain in DOMAIN_NAMES:
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as directory, \
                 patch.object(revisit, "_probe_video", return_value=info), \
                 patch.object(revisit, "_ffmpeg", return_value="ffmpeg"), \
                 patch.object(revisit, "_assemble", side_effect=self._fake_assemble):
                segmentation, source, output = self._fixture(directory)
                segmentation.write_text(json.dumps(metadata(5.0, 2, domain)))
                if domain == "cook":
                    with self.assertRaisesRegex(ValueError, "at least 3"):
                        revisit.construct_episode(segmentation, source, output, None,
                                                  domain=domain, target_sec=7, max_total_sec=10)
                    continue
                plan = revisit.construct_episode(segmentation, source, output, None,
                                                 domain=domain, target_sec=7, max_total_sec=10)
                self.assertEqual(plan["domain"], domain)
                self.assertEqual(plan["source_schema_version"], get_domain(domain).segmentation_schema)
                self.assertEqual(plan["visit_count"], 2)
                self.assertAlmostEqual(plan["visits"][-1]["output_end_sec"], plan["total_sec"])

    def test_non_cook_priorities_follow_duration_and_precede_transition_count(self):
        for domain in ("bike", "health", "music", "sports"):
            spec = get_domain(domain)
            low = min(spec.region_priority, key=spec.region_priority.get)
            high = max(spec.region_priority, key=spec.region_priority.get)
            example = metadata(domain=domain)["segments"][0]
            candidates = [
                {**example, "start_sec": float(i * 6), "end_sec": float(i * 6 + 5),
                 "region": region, "_index": i}
                for i, region in enumerate((low, high, low, high))
            ]
            with self.subTest(domain=domain):
                self.assertEqual(revisit.choose_region(candidates, 45, 1.5, 0.2,
                                                      domain=domain), high)
                candidates[0]["end_sec"] = 20.0
                self.assertEqual(revisit.choose_region(candidates, 45, 1.5, 0.2,
                                                      domain=domain), low)

    def test_sports_never_combines_different_regions_to_meet_minimum(self):
        document = metadata(5.0, 2, "sports")
        document["segments"][2]["region"] = "soccer_ball_area"
        document["segments"][2]["activity_type"] = "kick_control"
        segments, _, _ = revisit.validate_segmentation_meta(document, domain="sports")
        with self.assertRaises(ValueError):
            revisit.choose_region(segments, 7, 1.5, 0.2, domain="sports")
        with self.assertRaisesRegex(ValueError, "one region"):
            revisit.pick_visits_greedy(segments, 7, 0.2, domain="sports")
        with self.assertRaisesRegex(ValueError, "one region"):
            revisit.validate_visits([], segments, 1.5, 0.2, 60, domain="sports")

    def test_domain_prompts_receive_all_dynamic_parameters(self):
        for domain in DOMAIN_NAMES:
            spec = get_domain(domain)
            segments, _, _ = revisit.validate_segmentation_meta(metadata(domain=domain), domain=domain)
            region = segments[0]["region"]
            scoped = [segment for segment in segments if segment["region"] == region]
            client = FakeClient('{"visits": []}')
            with self.subTest(domain=domain):
                revisit._llm_visits(client, scoped, region,
                                   {"target_sec": 31.0, "max_total_sec": 40.0,
                                    "speed": 2.0, "blank_sec": 0.5}, 84.0, domain=domain)
                prompt = client.calls[0][0]
                self.assertIn(region, prompt)
                self.assertNotIn("$", prompt)
                self.assertIn(str(spec.min_visits), prompt)
                self.assertNotIn("disable_thinking", client.calls[0][2])


FFMPEG = shutil.which(os.environ.get("STORM_FFMPEG", "ffmpeg"))
HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(FFMPEG and HAS_CV2, "Real video test requires FFmpeg and OpenCV")
class RevisitFFmpegTests(unittest.TestCase):
    def test_real_output_decode_duration_timeline_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, segmentation, output = root / "source.mp4", root / "segments.json", root / "episode.mp4"
            subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12:duration=18", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True, capture_output=True)
            segmentation.write_text(json.dumps(metadata(length=5.0, visits=3)))
            plan = revisit.construct_episode(segmentation, source, output, None, target_sec=10.0, max_total_sec=10.0)
            self.assertEqual(plan["visit_count"], 3)
            self.assertLessEqual(plan["total_sec"], 10.0)
            self.assertLessEqual(abs(plan["duration_error_sec"]), 5 / 12)
            self.assertAlmostEqual(plan["visits"][-1]["output_end_sec"], plan["total_sec"])
            self.assertEqual(revisit._probe_video(output, decode=True)["frame_count"], plan["output_frame_count"])
            self.assertEqual(revisit.construct_episode(segmentation, source, output, None, target_sec=10.0, max_total_sec=10.0), plan)
            self.assertFalse(list(root.glob(".storm-revisit-*")))


if __name__ == "__main__":
    unittest.main()
