"""Segmentation contracts without video codecs or a live model API."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from storm_real_pipeline import segmentation as seg
from storm_real_pipeline.domains import DOMAIN_NAMES, get_domain


def interval(start=0.0, end=2.0, **updates):
    result = {
        "start_sec": start, "end_sec": end, "region": "stove_area",
        "segment_role": "observation", "activity_type": "observe",
        "change_types": [], "description": "A pan is visible on the stove.",
    }
    result.update(updates)
    return result


class WindowTests(unittest.TestCase):
    def test_default_windows_merge_short_tail(self):
        self.assertEqual(seg.build_processing_windows(640), [(0.0, 300.0), (300.0, 640.0)])
        self.assertEqual(seg.build_processing_windows(645), [(0.0, 300.0), (300.0, 600.0), (600.0, 645.0)])
        self.assertEqual(seg.build_processing_windows(300), [(0.0, 300.0)])

    def test_zero_tail_does_not_create_empty_window(self):
        self.assertEqual(seg.build_processing_windows(600, min_tail_sec=0), [(0.0, 300.0), (300.0, 600.0)])

    def test_rejects_nonfinite_and_invalid_parameters(self):
        for kwargs in ({"duration_sec": float("nan")}, {"duration_sec": 0},
                       {"duration_sec": 60, "window_sec": 0},
                       {"duration_sec": 60, "min_tail_sec": -1},
                       {"duration_sec": 60, "min_tail_sec": 301}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                seg.build_processing_windows(**kwargs)


class SchemaTests(unittest.TestCase):
    def test_interaction_changes_are_retained(self):
        item = interval(segment_role="interaction", activity_type="cook",
                        change_types=["state_change", "state_change"])
        result = seg.validate_segments([item], 0, 2, [0, 1, 2])
        self.assertEqual(result[0]["change_types"], ["state_change"])
        self.assertEqual(item["change_types"], ["state_change", "state_change"])

    def test_rejects_schema_and_timeline_violations(self):
        bad_cases = [
            [], [interval(region=[])], [interval(activity_type="cook")],
            [interval(change_types=["state_change"])],
            [interval(change_types=[{}])], [interval(start_sec=True)],
            [interval(end_sec=float("nan"))], [interval(end_sec=1.4)],
            [interval(description="")], [interval(description="x" * 151)],
            [interval(extra="field")], [interval(0, 1), interval(2, 3)],
            [interval(0, 2), interval(1, 3)], [interval(1, 3)],
        ]
        for items in bad_cases:
            with self.subTest(items=items), self.assertRaises(seg.SegmentSchemaError):
                seg.validate_segments(items, 0, 3, [0, 1, 2, 3])

    def test_small_boundary_offsets_are_canonicalized(self):
        result = seg.validate_segments([interval(0.01, 1.99)], 0, 2, [0, 1, 2])
        self.assertEqual((result[0]["start_sec"], result[0]["end_sec"]), (0, 2))

    def test_each_domain_uses_its_own_region_and_activity_enums(self):
        examples = {
            "cook": ("stove_area", "cook"),
            "bike": ("tire_tube_area", "inflate_deflate"),
            "health": ("test_kit_area", "operate_test_kit"),
            "music": ("instrument_playing_area", "play"),
            "sports": ("basketball_ball_drill_area", "dribble_control"),
        }
        for domain, (region, activity) in examples.items():
            with self.subTest(domain=domain):
                item = interval(region=region, segment_role="interaction",
                                activity_type=activity, change_types=["state_change"])
                result = seg.validate_segments([item], 0, 2, [0, 1, 2], domain=domain)
                self.assertEqual(result[0]["region"], region)
                wrong_domain = "bike" if domain == "cook" else "cook"
                with self.assertRaises(seg.SegmentSchemaError):
                    seg.validate_segments([item], 0, 2, [0, 1, 2], domain=wrong_domain)
                invalid = {**item, "activity_type": "observe"}
                with self.assertRaises(seg.SegmentSchemaError):
                    seg.validate_segments([invalid], 0, 2, [0, 1, 2], domain=domain)


class SamplingTests(unittest.TestCase):
    def test_frames_stay_in_memory_and_capture_is_released(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [10.0, 20.0]
        cap.read.return_value = (True, object())
        with tempfile.TemporaryDirectory() as temp:
            frames_dir = Path(temp) / "unused_frames"
            with patch.object(seg, "cv2") as cv2, patch.object(seg, "_blur_and_encode", return_value="jpeg"):
                cv2.VideoCapture.return_value = cap
                frames, end = seg.sample_frames(Path("source.mp4"), 1, frames_dir, 75, 0, 2)
            self.assertEqual(frames, [(0.0, "jpeg"), (1.0, "jpeg")])
            self.assertEqual(end, 2.0)
            self.assertFalse(frames_dir.exists())
            cap.release.assert_called_once()

    def test_decode_failure_is_not_silently_truncated(self):
        cap = Mock()
        cap.get.side_effect = [10.0, 20.0]
        cap.read.return_value = (False, None)
        with patch.object(seg, "cv2") as cv2:
            cv2.VideoCapture.return_value = cap
            with self.assertRaises(OSError):
                seg.sample_frames(Path("source.mp4"), 1, end_sec=2)
        cap.release.assert_called_once()


class GenerationTests(unittest.TestCase):
    def test_batch_retry_and_full_coverage(self):
        client = Mock()
        client.call.side_effect = ["invalid JSON", json.dumps([interval(0, 2)]),
                                   "```json\n" + json.dumps([interval(2, 4)]) + "\n```"]
        frames = [(float(index), "jpeg") for index in range(4)]
        with patch.object(seg, "sample_frames", return_value=(frames, 4.0)), \
             patch.object(seg, "load_prompt", return_value="prompt"):
            result = seg.segment_video(client, Path("source.mp4"), 0, 4, max_frames_per_call=2)
        self.assertEqual([(item["start_sec"], item["end_sec"]) for item in result], [(0, 2), (2, 4)])
        self.assertEqual(client.call.call_count, 3)
        self.assertIn("previous response failed", client.call.call_args_list[1].args[1][1])

    def test_exhausted_schema_retries_raise(self):
        client = Mock()
        client.call.return_value = "[]"
        with patch.object(seg, "sample_frames", return_value=([(0.0, "jpeg")], 2.0)), \
             patch.object(seg, "load_prompt", return_value="prompt"):
            with self.assertRaises(seg.SegmentSchemaError):
                seg.segment_video(client, Path("source.mp4"), 0, 2)
        self.assertEqual(client.call.call_count, 3)

    def test_domain_retry_policy_and_prompt_routing(self):
        for domain in DOMAIN_NAMES:
            client = Mock()
            client.call.return_value = "[]"
            with self.subTest(domain=domain), \
                 patch.object(seg, "sample_frames", return_value=([(0.0, "jpeg")], 2.0)), \
                 patch.object(seg, "load_prompt", return_value="domain prompt") as prompt:
                with self.assertRaises(seg.SegmentSchemaError):
                    seg.segment_video(client, Path("source.mp4"), 0, 2, domain=domain)
                self.assertEqual(client.call.call_count, get_domain(domain).segmentation_retries)
                prompt.assert_called_once_with("segmentation.txt", domain=domain)


class SourceOutputTests(unittest.TestCase):
    def test_anonymous_output_and_cache_validation(self):
        client = Mock(model="test-model")
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(seg, "_video_metadata", return_value=(10.0, 20)), \
             patch.object(seg, "load_prompt", return_value="prompt"), \
             patch.object(seg, "segment_video", return_value=[interval()]) as generate:
            source = Path(temp) / "original_private_name.mp4"
            source.write_bytes(b"video content")
            output_dir = Path(temp) / "out"
            paths = seg.segment_source(source, output_dir, client, video_id="video_0001")
            self.assertEqual(paths[0].name, "video_0001_window001.json")
            text = paths[0].read_text()
            self.assertNotIn(source.name, text)
            self.assertNotIn(str(source.parent), text)
            document = json.loads(text)
            self.assertEqual(document["source_video_id"], "video_0001")
            self.assertEqual(document["sample_timestamps"], [0.0, 1.0])
            seg.segment_source(source, output_dir, client, video_id="video_0001")
            self.assertEqual(generate.call_count, 1)

            for mutate in (
                lambda value: value["segments"][0].update(end_sec=1),
                lambda value: value["parameters"].update(sample_fps=2),
                lambda value: value.update(window_interval=[0, 1]),
                lambda value: value.update(schema_version="old"),
                lambda value: value.update(domain="bike"),
                lambda value: value.update(video_path="private.mp4"),
            ):
                corrupt = copy.deepcopy(document)
                mutate(corrupt)
                paths[0].write_text(json.dumps(corrupt))
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    seg.segment_source(source, output_dir, client, video_id="video_0001")

            paths[0].write_text(text)
            client.model = "another-model"
            with self.assertRaisesRegex(ValueError, "overwrite"):
                seg.segment_source(source, output_dir, client, video_id="video_0001")
            client.model = "test-model"
            with patch.object(seg, "load_prompt", return_value="changed prompt"):
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    seg.segment_source(source, output_dir, client, video_id="video_0001")
            source.write_bytes(b"changed video content")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                seg.segment_source(source, output_dir, client, video_id="video_0001")
            seg.segment_source(source, output_dir, client, video_id="video_0001", overwrite=True)
            self.assertEqual(generate.call_count, 2)

    def test_invalid_cache_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.json"
            path.write_text("{broken")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                seg._validate_cached_result(path, {}, [])

    def test_identifiers_cannot_escape_output_directory(self):
        for video_id in ("../private", "video/name", "", ".", "video name"):
            with self.subTest(video_id=video_id), self.assertRaises(ValueError):
                seg.segment_source(Path("unused"), Path("unused"), Mock(), video_id=video_id)

    def test_domain_artifact_metadata_and_cross_domain_cache_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "private_source.mp4"
            source.write_bytes(b"source")
            for domain in DOMAIN_NAMES:
                spec = get_domain(domain)
                region = next(region for region in sorted(spec.regions) if region != "other_area")
                client = Mock(model="test-model")
                with self.subTest(domain=domain), \
                     patch.object(seg, "_video_metadata", return_value=(10.0, 20)), \
                     patch.object(seg, "load_prompt", return_value="prompt"), \
                     patch.object(seg, "segment_video", return_value=[interval(region=region)]) as generate:
                    paths = seg.segment_source(source, root / domain, client,
                                               video_id="video_001", domain=domain)
                    document = json.loads(paths[0].read_text())
                    self.assertEqual(document["domain"], domain)
                    self.assertEqual(document["schema_version"], spec.segmentation_schema)
                    self.assertEqual(document["parameters"]["domain"], domain)
                    self.assertEqual(generate.call_args.kwargs["domain"], domain)
                    seg.segment_source(source, root / domain, client,
                                       video_id="video_001", domain=domain)
                    self.assertEqual(generate.call_count, 1)
                    wrong_domain = "bike" if domain == "cook" else "cook"
                    with self.assertRaisesRegex(ValueError, "overwrite"):
                        seg.segment_source(source, root / domain, client,
                                           video_id="video_001", domain=wrong_domain)


if __name__ == "__main__":
    unittest.main()
