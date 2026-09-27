from __future__ import annotations

import json
import re
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from tools.storm.benchgen.stream_eqa.contracts import (
    ContractError,
    QUESTION_TYPES,
    validate_episode_document,
)
from tools.storm.benchgen.stream_eqa.event_adapter import build_events
from tools.storm.benchgen.stream_eqa.exporter import export_run, validate_dataset
from tools.storm.benchgen.stream_eqa.keyclip_exporter import (
    CLIP_KEYS,
    CLIP_PLAN_KEYS,
    VideoInfo,
    build_clip_specs,
    export_keyclip_run,
    validate_keyclip_dataset,
)
from tools.storm.benchgen.stream_eqa.questions import build_questions


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class StreamEqaExportTest(unittest.TestCase):
    @staticmethod
    def _fake_clip_renderer(
        source: Path,
        target: Path,
        start_sec: float,
        end_sec: float,
        speed: float,
        fps: int,
    ) -> VideoInfo:
        del source
        frames = round((end_sec - start_sec) * fps / speed)
        target.write_bytes(f"synthetic-key-clip:{frames}".encode())
        return VideoInfo(frames=frames, duration_sec=frames / fps)

    def _run_fixture(self, root: Path) -> tuple[Path, Path]:
        run = root / "run_20260803_000000"
        episode_name = "00000_FloorPlan1_seed3"
        episode = run / episode_name
        episode.mkdir(parents=True)
        plan_id = "0123456789abcdef"
        targets = [
            ("Bowl|one", "Bowl", "station-1"),
            ("Bread|one", "Bread", "station-2"),
            ("Pan|one", "Pan", "station-3"),
            ("Cup|one", "Cup", "station-4"),
            ("Plate|one", "Plate", "station-5"),
        ]
        event_targets = [*targets, targets[3], targets[4], targets[2], targets[0], targets[1]]
        event_types = ["disappear"] * 5 + ["appear"] * 5
        trigger_times = [3.5, 9.3, 15.6, 21.1, 26.9, 32.7, 38.8, 44.3, 50.0, 56.2]
        events = []
        opportunities = []
        executions = []
        for index, ((target_id, target_type, station_id), event_type, trigger) in enumerate(
            zip(event_targets, event_types, trigger_times)
        ):
            events.append({
                "index": index,
                "event_type": event_type,
                "event_family": "presence_change",
                "event_action": event_type,
                "target_id": target_id,
                "target_type": target_type,
                "station_id": station_id,
                "trigger_time_s": trigger,
                "parameters": [],
            })
            opportunities.append({
                "index": index,
                "target_id": target_id,
                "before_station_id": station_id,
                "after_station_id": station_id,
                "before_time_s": trigger - 1.05,
                "hidden_time_s": trigger - 0.10,
                "trigger_time_s": trigger,
                "after_time_s": trigger + 1.20,
            })
            executions.append({
                "index": index,
                "event_type": event_type,
                "target_id": target_id,
                "applied": True,
                "observed": True,
                "note": "disabled" if event_type == "disappear" else "enabled",
                "before_evidence": {"visible": event_type == "disappear"},
                "after_evidence": {"visible": event_type == "appear"},
            })
        plan = {
            "plan_id": plan_id,
            "recipe": {"scene": "FloorPlan1", "seed": 3},
            "event_program": {"events": events},
            "mobility": {"opportunities": opportunities},
            "trajectory": {"fps": 30, "frames": [None] * 1801},
        }
        profile = {
            "stations": [
                {
                    "station_id": station_id,
                    "surface_type": "DiningTable" if target_type != "Pan" else "StoveBurner",
                    "target_pixels": 2000 + index,
                }
                for index, (_, target_type, station_id) in enumerate(targets)
            ],
        }
        _write(run / "manifest.json", {
            "accepted": True,
            "episodes": [{"directory": episode_name, "plan_id": plan_id}],
        })
        _write(run / "batch_validation.json", {"accepted": True})
        _write(episode / "episode_validation.json", {"accepted": True})
        _write(episode / "execution_trace.json", {
            "accepted": True, "failed_frames": [], "events": executions,
        })
        _write(episode / "episode_plan.json", plan)
        _write(episode / "scene_profile.json", profile)
        video = episode / "rollout.mp4"
        video.write_bytes(b"synthetic-video-fixture")
        (episode / "rollout_highlighted.mp4").write_bytes(b"audit-only")
        return run, video

    def test_export_matches_stream_eqa_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, source_video = self._run_fixture(root)
            output = root / "qa_results"
            summary = export_run(run, output)
            self.assertEqual(summary["video_count"], 1)
            self.assertEqual(summary["question_count"], 6)
            self.assertEqual(
                summary["question_type_counts"],
                {question_type: 1 for question_type in QUESTION_TYPES},
            )
            episode_jsons = list(output.glob("FloorPlan1/*.json"))
            episode_videos = list(output.glob("FloorPlan1/*.mp4"))
            self.assertEqual(len(episode_jsons), 1)
            self.assertEqual(len(episode_videos), 1)
            self.assertEqual(source_video.stat().st_ino, episode_videos[0].stat().st_ino)
            document = json.loads(episode_jsons[0].read_text(encoding="utf-8"))
            self.assertEqual(len(document["events"]), 10)
            self.assertEqual(len(document["questions"]), 6)
            self.assertEqual(
                {item["question_type"] for item in document["questions"]},
                set(QUESTION_TYPES),
            )
            public_text = json.dumps(document, ensure_ascii=False)
            self.assertNotIn("_target_id", public_text)
            for internal_term in (
                "simulator", "object id", "物体实例", "禁用", "启用", "该事件",
                "目标掩码",
            ):
                self.assertNotIn(internal_term, public_text)
            self.assertIsNone(re.search(r"\d+\.\d{2}秒", public_text))
            for question in document["questions"]:
                self.assertTrue(question["question"].endswith("？"))
                self.assertGreaterEqual(len(question["question"]), 20)
            tracking = next(
                item for item in document["questions"]
                if item["question_type"] == "object_tracking"
            )
            self.assertIn("同一位置", tracking["video_evidence"])
            self.assertNotIn("标识", tracking["video_evidence"])
            self.assertFalse(any(output.rglob("*highlighted*")))
            self.assertEqual(validate_dataset(output), summary)
            with self.assertRaises(FileExistsError):
                export_run(run, output)

    def test_natural_questions_vary_across_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._run_fixture(root)
            episode = next(path for path in run.iterdir() if path.is_dir())
            load = lambda name: json.loads(
                (episode / name).read_text(encoding="utf-8")
            )
            events = build_events(
                load("episode_plan.json"),
                load("execution_trace.json"),
                load("scene_profile.json"),
            )
            prompts: dict[str, set[str]] = defaultdict(set)
            answers: dict[str, set[str]] = defaultdict(set)
            for index in range(24):
                generated = build_questions(
                    f"natural_episode_{index}", events, duration=60.03,
                )
                for question in generated:
                    question_type = question["question_type"]
                    prompts[question_type].add(question["question"])
                    answers[question_type].add(
                        question["options"][question["answer_index"]]
                    )
            for question_type in QUESTION_TYPES:
                self.assertGreaterEqual(
                    len(prompts[question_type]), 6, question_type,
                )
            self.assertTrue(any(
                "还没有重新出现" in answer
                for answer in answers["current_state"]
            ))
            self.assertTrue(any(
                "重新出现在" in answer
                for answer in answers["current_state"]
            ))
            self.assertEqual(
                answers["state_change"],
                {"它从原来的位置消失了", "它重新出现在原来的位置"},
            )
            self.assertTrue(any(
                "最早消失" in prompt or "最先从画面中不见" in prompt
                for prompt in prompts["temporal_reasoning"]
            ))
            self.assertTrue(any(
                "最早重新出现" in prompt or "最先回到原来的位置" in prompt
                for prompt in prompts["temporal_reasoning"]
            ))
            self.assertTrue(any(
                "消失" in prompt for prompt in prompts["history_aggregation"]
            ))
            self.assertTrue(any(
                "回来" in prompt or "重新出现" in prompt
                for prompt in prompts["history_aggregation"]
            ))
            self.assertGreaterEqual(
                len(answers["history_aggregation"]), 3,
            )

    def test_keyclip_export_matches_reference_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._run_fixture(root)
            episode = next(path for path in run.iterdir() if path.is_dir())
            load = lambda name: json.loads(
                (episode / name).read_text(encoding="utf-8")
            )
            events = build_events(
                load("episode_plan.json"),
                load("execution_trace.json"),
                load("scene_profile.json"),
            )
            specs = build_clip_specs(
                events,
                source_frame_count=1801,
                fps=30,
            )
            self.assertEqual(len(specs), 6)
            self.assertEqual(
                [len(spec.event_indices) for spec in specs],
                [2, 1, 2, 2, 1, 2],
            )

            output = root / "qa_results"
            summary = export_keyclip_run(
                run,
                output,
                renderer=self._fake_clip_renderer,
            )
            self.assertEqual(summary["full_episode_count"], 1)
            self.assertEqual(summary["key_clip_count"], 6)
            self.assertEqual(summary["event_count"], 10)
            self.assertEqual(summary["question_count"], 37)
            self.assertEqual(summary["clips_per_episode"], {6: 1})
            self.assertEqual(summary["question_type_counts"], {
                "current_state": 6,
                "factual_retrieval": 10,
                "history_aggregation": 6,
                "object_tracking": 3,
                "state_change": 6,
                "temporal_reasoning": 6,
            })
            plan_path = next(output.glob("FloorPlan1/*.clips.json"))
            clip_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(set(clip_plan), CLIP_PLAN_KEYS)
            self.assertEqual(clip_plan["speed"], 1.5)
            self.assertEqual(clip_plan["blank_sec"], 0.2)
            self.assertEqual(len(clip_plan["clips"]), 6)
            self.assertTrue(all(
                set(clip) == CLIP_KEYS for clip in clip_plan["clips"]
            ))
            documents = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(output.glob("FloorPlan1/*.json"))
                if not path.name.endswith(".clips.json")
            ]
            self.assertEqual([len(doc["events"]) for doc in documents], [2, 1, 2, 2, 1, 2])
            self.assertEqual([len(doc["questions"]) for doc in documents], [7, 5, 7, 6, 6, 6])
            self.assertEqual(sum(len(doc["events"]) for doc in documents), 10)
            for document in documents:
                public_text = json.dumps(document, ensure_ascii=False)
                self.assertNotIn("simulator", public_text)
                self.assertNotIn("object id", public_text)
                self.assertNotIn("该事件", public_text)
                self.assertNotRegex(public_text, r"\d+\.\d{2}秒")
                self.assertNotIn("一直保持片段开始时的状态", public_text)
                self.assertEqual(
                    len(document["questions"]),
                    len({question["question"] for question in document["questions"]}),
                )
                spans = {
                    (event["start_sec"], event["end_sec"])
                    for event in document["events"]
                }
                for question in document["questions"]:
                    self.assertTrue(all(
                        tuple(span) in spans
                        for span in question["evidence_spans"]
                    ))
                if len(document["events"]) == 2:
                    temporal = next(
                        question for question in document["questions"]
                        if question["question_type"] == "temporal_reasoning"
                    )
                    self.assertTrue(all(
                        event["subject"] in temporal["options"]
                        for event in document["events"]
                    ))
                tracking_questions = [
                    question for question in document["questions"]
                    if question["question_type"] == "object_tracking"
                ]
                for tracking in tracking_questions:
                    self.assertIn("符合下面哪种情况", tracking["question"])
            self.assertFalse((output / "summary.json").exists())
            self.assertFalse((output / "questions.jsonl").exists())
            self.assertEqual(validate_keyclip_dataset(output), summary)

    def test_contract_rejects_duplicate_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._run_fixture(root)
            output = root / "qa_results"
            export_run(run, output)
            json_path = next(output.glob("FloorPlan1/*.json"))
            video_path = json_path.with_suffix(".mp4")
            document = json.loads(json_path.read_text(encoding="utf-8"))
            document["questions"][0]["options"][1] = document["questions"][0]["options"][0]
            with self.assertRaises(ContractError):
                validate_episode_document(document, json_path, video_path)


if __name__ == "__main__":
    unittest.main()
