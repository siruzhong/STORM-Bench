"""Offline tests for QA evidence, quota, provenance, and resume contracts."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storm_real_pipeline import qa


def candidate(index=0, question_type="current_state", status="known"):
    return {
        "id": "pending",
        "episode_id": "ignored_model_identifier",
        "query_time": 2.0,
        "question_type": question_type,
        "question_subtype": "visible_state",
        "video_evidence": "A red cup is visible on the counter at two seconds.",
        "question": f"Which visible feature answers question {index}?",
        "options": ["Red", "Blue", "Green", "Cannot be determined from the visible frames"],
        "answer_index": 0 if status == "known" else 3,
        "evidence_spans": [[1.0, 1.0], [2.0, 2.0]],
        "diagnostics": {
            "epistemic_status": status,
            "uncertainty_sources": [] if status == "known" else ["missing_observation"],
        },
        "diagnostic_rationale": {
            "volatility": "The query occurs after one visible visit.",
            "uncertainty": "The answer follows only from sampled visual evidence.",
        },
    }


def quota_candidates(domain="cook"):
    types = sorted(qa.QUESTION_TYPES)
    questions = [candidate(i, types[i // 2], "known" if i < 8 else "uncertain") for i in range(10)]
    if domain != "cook":
        for question in questions:
            question["options"][3] = "Cannot be determined"
    return questions


class FakeClient:
    model = "mock-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, system_prompt, contents, **kwargs):
        self.calls.append((system_prompt, contents, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


class ValidationTests(unittest.TestCase):
    def validate(self, question):
        return qa.validate_question(question, "episode_001", 4.0, [0.0, 1.0, 2.0, 3.0])

    def test_valid_known_and_uncertain(self):
        for status in ("known", "uncertain"):
            with self.subTest(status=status):
                result = self.validate(candidate(status=status))
                self.assertIsNotNone(result)
                self.assertEqual(result["episode_id"], "episode_001")
                self.assertEqual(result["diagnostics"]["epistemic_status"], status)

    def test_rejects_nonfinite_boolean_or_string_timestamps(self):
        for value in (math.nan, math.inf, -math.inf, True, "2", 10 ** 1000):
            with self.subTest(value=repr(value)[:20]):
                question = candidate()
                question["query_time"] = value
                self.assertIsNone(self.validate(question))
                question = candidate()
                question["evidence_spans"] = [[value, value]]
                self.assertIsNone(self.validate(question))

    def test_rejects_future_evidence_and_unsampled_endpoints(self):
        for spans in ([[3.0, 3.0]], [[1.5, 2.0]], [[2.0, 1.0]]):
            question = candidate()
            question["evidence_spans"] = spans
            self.assertIsNone(self.validate(question))
        question = candidate()
        question["query_time"] = 2.01
        question["evidence_spans"] = [[1.01, 2.01]]
        result = self.validate(question)
        self.assertEqual(result["query_time"], 2.0)
        self.assertEqual(result["evidence_spans"], [[1.0, 2.0]])

    def test_invalid_status_is_not_silently_known(self):
        for value in ("Unknown", "uncertainty", None, [], 1):
            question = candidate()
            question["diagnostics"]["epistemic_status"] = value
            self.assertIsNone(self.validate(question))
        question = candidate()
        del question["diagnostics"]
        self.assertIsNone(self.validate(question))

    def test_rejects_invalid_uncertainty_sources_and_missing_abstention(self):
        for sources in ([], ["invented_source"], ["missing_observation", "invalid"], [{}]):
            question = candidate(status="uncertain")
            question["diagnostics"]["uncertainty_sources"] = sources
            self.assertIsNone(self.validate(question))
        question = candidate(status="uncertain")
        question["answer_index"] = 0
        self.assertIsNone(self.validate(question))
        question = candidate()
        question["diagnostics"]["uncertainty_sources"] = ["missing_observation"]
        self.assertIsNone(self.validate(question))

    def test_rejects_empty_text_and_duplicate_options(self):
        for field in ("question", "video_evidence", "question_subtype"):
            question = candidate()
            question[field] = "  "
            self.assertIsNone(self.validate(question))
        question = candidate()
        question["options"][1] = "  rEd "
        self.assertIsNone(self.validate(question))

    def test_json_parser_rejects_nan_and_explanatory_prose(self):
        self.assertEqual(qa.parse_json_response("```json\n[]\n```", list), [])
        for text in ('[{"query_time": NaN}]', "Here is your JSON: []"):
            with self.assertRaises(ValueError):
                qa.parse_json_response(text, list)


class GenerationTests(unittest.TestCase):
    frames = [(0.0, "jpeg0"), (1.0, "jpeg1"), (2.0, "jpeg2"), (3.0, "jpeg3")]

    def test_quota_filling_and_casefold_deduplication(self):
        questions = quota_candidates()
        duplicate = copy.deepcopy(questions[0])
        duplicate["question"] = "  " + duplicate["question"].upper() + "  "
        client = FakeClient([[questions[0], duplicate, *questions[1:8]], questions[8:]])
        result = qa.generate_questions(client, "episode_001", 4.0, self.frames)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0]["id"], "episode_001_q01")
        second_metadata = json.loads(client.calls[1][1][0]["text"])
        self.assertEqual(second_metadata["generation_mode"], "uncertain_only")
        self.assertEqual(second_metadata["remaining_uncertain_needed"], 2)
        self.assertTrue(all(sum(q["question_type"] == t for q in result) <= 2 for t in qa.QUESTION_TYPES))

    def test_five_invalid_rounds_fail_without_fabricating_answers(self):
        client = FakeClient(["invalid"] * 5)
        with self.assertRaisesRegex(RuntimeError, "after 5 attempts"):
            qa.generate_questions(client, "episode_001", 4.0, self.frames)
        self.assertEqual(len(client.calls), 5)

    def test_transport_error_is_not_a_schema_retry(self):
        client = FakeClient([RuntimeError("service unavailable")])
        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            qa.generate_questions(client, "episode_001", 4.0, self.frames)
        self.assertEqual(len(client.calls), 1)

    def test_prompt_retains_core_evidence_rules(self):
        client = FakeClient([quota_candidates()])
        qa.generate_questions(client, "episode_001", 4.0, self.frames, thinking="disabled")
        prompt, content, arguments = client.calls[0]
        for token in ("at least 8", "2 epistemic_status=uncertain", "at most 2", "minimal sufficient evidence",
                      "Category agreement alone", "Do not use a plan", "sparse sample"):
            self.assertIn(token, prompt)
        self.assertFalse(any("\u4e00" <= char <= "\u9fff" for char in prompt))
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 4)
        self.assertEqual(arguments["thinking"], "disabled")


class DomainValidationTests(unittest.TestCase):
    def validate(self, question, domain):
        return qa.validate_question(question, "episode_001", 4.0, [0.0, 1.0, 2.0, 3.0], domain=domain)

    def test_domain_is_attached_and_conflicting_label_rejected(self):
        for domain in ("cook", "bike", "health", "music", "sports"):
            with self.subTest(domain=domain):
                question = candidate()
                self.assertEqual(self.validate(question, domain)["domain"], domain)
                question["domain"] = "bike" if domain == "cook" else "cook"
                self.assertIsNone(self.validate(question, domain))

    def test_bike_requires_exact_uncertain_option(self):
        question = candidate(status="uncertain")
        self.assertIsNone(self.validate(question, "bike"))
        question["options"][3] = "Cannot be determined"
        self.assertIsNotNone(self.validate(question, "bike"))
        question["options"][3] = "Cannot be determined."
        self.assertIsNone(self.validate(question, "bike"))

    def test_health_music_sports_normalize_english_answers_and_sources(self):
        for domain in ("health", "music", "sports"):
            with self.subTest(domain=domain):
                question = candidate(status="uncertain")
                question["options"][3] = "There is insufficient information to resolve this identity."
                question["diagnostics"]["uncertainty_sources"] = "identity_ambiguity"
                original = copy.deepcopy(question)
                validated = self.validate(question, domain)
                self.assertEqual(validated["options"][3], "Cannot be determined")
                self.assertEqual(validated["diagnostics"]["uncertainty_sources"], ["ambiguous_evidence"])
                self.assertEqual(question, original, "Validation must not mutate model candidates")
                question["diagnostics"]["uncertainty_sources"] = ["occlusion", "out_of_view", "attribute_ambiguity"]
                self.assertEqual(self.validate(question, domain)["diagnostics"]["uncertainty_sources"],
                                 ["partial_observation", "missing_observation", "ambiguous_attribute"])

    def test_canonicalization_rejects_duplicates_and_unknown_source_aliases(self):
        question = candidate(status="uncertain")
        question["options"][2] = "Cannot be determined"
        self.assertIsNone(self.validate(question, "health"))
        question = candidate(status="uncertain")
        question["diagnostics"]["uncertainty_sources"] = ["made_up_alias"]
        self.assertIsNone(self.validate(question, "music"))

    def test_all_domains_preserve_their_prompts_and_answer_meaning(self):
        expected_titles = {"cook": "long-term video memory", "bike": "Bike Repair",
                           "health": "Health Procedures", "music": "Music Performance", "sports": "Sports"}
        for domain, title in expected_titles.items():
            with self.subTest(domain=domain):
                client = FakeClient([quota_candidates(domain)])
                result = qa.generate_questions(client, "episode_001", 4.0, GenerationTests.frames, domain=domain)
                self.assertIn(title, client.calls[0][0])
                self.assertTrue(all(q["domain"] == domain for q in result))
                positions = [q["answer_index"] for q in result]
                if domain != "cook":
                    counts = [positions.count(i) for i in range(4)]
                    self.assertLessEqual(max(counts) - min(counts), 1)
                    repeated = qa.generate_questions(FakeClient([quota_candidates(domain)]), "episode_001", 4.0,
                                                     GenerationTests.frames, domain=domain)
                    self.assertEqual(positions, [q["answer_index"] for q in repeated])
                else:
                    self.assertEqual(positions, [0] * 8 + [3] * 2)
                for question in result:
                    if question["diagnostics"]["epistemic_status"] == "known":
                        self.assertEqual(question["options"][question["answer_index"]], "Red")
                    elif domain != "cook":
                        self.assertEqual(question["options"][question["answer_index"]], "Cannot be determined")

    def test_health_music_sports_retain_twelve_round_budget(self):
        for domain in ("health", "music", "sports"):
            with self.subTest(domain=domain), self.assertLogs(qa.logger, "WARNING"):
                client = FakeClient(["invalid"] * 11 + [quota_candidates(domain)])
                result = qa.generate_questions(client, "episode_001", 4.0, GenerationTests.frames, domain=domain)
                self.assertEqual(len(client.calls), 12)
                self.assertEqual(len(result), 10)

    def test_domain_specific_uncertain_supplement_instructions(self):
        for domain in ("bike", "health", "music", "sports"):
            with self.subTest(domain=domain):
                candidates = quota_candidates(domain)
                client = FakeClient([candidates[:8], candidates[8:]])
                qa.generate_questions(client, "episode_001", 4.0, GenerationTests.frames, domain=domain)
                metadata = json.loads(client.calls[1][1][0]["text"])
                self.assertEqual(metadata["generation_mode"], "uncertain_only")
                self.assertIn("Cannot be determined", metadata["instruction"])
                self.assertIn("ONLY 2" if domain == "bike" else "Generate 4 candidate", metadata["instruction"])


@unittest.skipIf(qa.cv2 is None, "OpenCV is unavailable in this test environment")
class SamplingTests(unittest.TestCase):
    def test_complete_episode_sampling_and_optional_helper_bound(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "episode_001.mp4"
            writer = qa.cv2.VideoWriter(str(video), qa.cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
            self.assertTrue(writer.isOpened(), "The test environment must support MP4 video encoding")
            try:
                for index in range(16):
                    writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
            finally:
                writer.release()
            frames, duration = qa.sample_frames(video, 1.0, 80)
            self.assertAlmostEqual(duration, 4.0)
            self.assertEqual([t for t, _ in frames], [0.0, 1.0, 2.0, 3.0])
            self.assertTrue(all(payload for _, payload in frames))
            bounded_frames, bounded_duration = qa.sample_frames(video, 1.0, 80, 2.0)
            self.assertEqual(bounded_duration, 2.0)
            self.assertEqual([t for t, _ in bounded_frames], [0.0, 1.0])


class PlanTests(unittest.TestCase):
    def test_prefers_exact_output_frame_times(self):
        plan = {
            "speed": 1.5, "blank_sec": 0.2,
            "visits": [
                {"start_sec": 10.0, "end_sec": 13.0, "output_start_sec": 0.0, "output_end_sec": 2.04},
                {"start_sec": 20.0, "end_sec": 23.0, "output_start_sec": 2.28, "output_end_sec": 4.32},
            ],
        }
        spans = qa.visit_output_spans(plan)
        self.assertEqual(spans[1]["out_start_sec"], 2.28)
        questions = [{"query_time": value} for value in (0.0, 2.1, 2.2, 2.28, 4.0)]
        qa.attach_change_intensity(questions, plan)
        self.assertEqual([q["change_intensity"] for q in questions], [1, 1, 1, 2, 2])

    def test_legacy_span_reconstruction_and_explicit_output_spans(self):
        plan = {"speed": 1.5, "blank_sec": 0.2,
                "visits": [{"start_sec": 10.0, "end_sec": 13.0}, {"start_sec": 20.0, "end_sec": 23.0}]}
        self.assertAlmostEqual(qa.visit_output_spans(plan)[1]["out_start_sec"], 2.2)
        plan["output_spans"] = [
            {"out_start_sec": 0.0, "out_end_sec": 2.04},
            {"out_start_sec": 2.28, "out_end_sec": 4.32},
        ]
        self.assertEqual(qa.visit_output_spans(plan)[1]["out_start_sec"], 2.28)

    def test_invalid_plan_is_not_silently_single_visit(self):
        for plan in ({"visits": []}, {"visits": [{"output_start_sec": math.nan, "output_end_sec": 2.0}]},
                     {"visits": [{"start_sec": 2.0, "end_sec": 1.0}]},
                     {"speed": 0, "visits": [{"start_sec": 0.0, "end_sec": 1.0}]}):
            with self.assertRaises(ValueError):
                qa.visit_output_spans(plan)
        questions = [{"query_time": 3.0}]
        qa.attach_change_intensity(questions, None)
        self.assertEqual(questions[0]["change_intensity"], 1)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "episode_001.mp4"
        self.video.write_bytes(b"fake video bytes for hashing")
        self.output = self.root / "episode_001.json"
        self.plan = self.root / "episode_001.plan.json"
        self.plan.write_text(json.dumps({"visits": [
            {"start_sec": 100.0, "end_sec": 103.0, "output_start_sec": 0.0, "output_end_sec": 4.0},
        ], "hidden_changes": ["PRIVATE_METADATA_MUST_NOT_REACH_MODEL"]}))
        self.client = FakeClient([quota_candidates()] * 10)
        patcher = patch.object(qa, "sample_frames", return_value=(GenerationTests.frames, 4.0))
        self.sample = patcher.start()
        self.addCleanup(patcher.stop)

    def run_episode(self, **kwargs):
        return qa.generate_episode_qa(
            self.video, self.output, self.client,
            episode_id="episode_001", plan_path=self.plan, **kwargs,
        )

    def test_valid_resume_and_private_metadata_exclusion(self):
        first = self.run_episode()
        second = self.run_episode()
        self.assertEqual(first, second)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.sample.call_count, 1)
        self.assertNotIn("video_path", first)
        payload = json.dumps(first)
        self.assertNotIn(str(self.root), payload)
        self.assertNotIn(self.video.name, payload)
        self.assertNotIn("PRIVATE_METADATA", payload)
        self.assertNotIn("PRIVATE_METADATA", json.dumps(self.client.calls))

    def test_changed_video_plan_model_or_config_regenerates(self):
        self.run_episode()
        self.video.write_bytes(b"new video content")
        self.run_episode()
        plan = json.loads(self.plan.read_text())
        plan["hidden_changes"] = ["different plan contents"]
        self.plan.write_text(json.dumps(plan))
        self.run_episode()
        self.client.model = "another-model"
        self.run_episode()
        self.run_episode(jpeg_quality=85)
        self.assertEqual(len(self.client.calls), 5)

    def test_malformed_or_schema_invalid_cached_output_regenerates(self):
        result = self.run_episode()
        self.output.write_text("{broken")
        self.run_episode()
        result["questions"][0]["diagnostics"]["epistemic_status"] = "invalid"
        self.output.write_text(json.dumps(result))
        self.run_episode()
        self.assertEqual(len(self.client.calls), 3)

    def test_configuration_and_missing_plan_fail_early(self):
        with self.assertRaisesRegex(ValueError, "questions_per_type"):
            self.run_episode(questions_per_type=1)
        with self.assertRaisesRegex(ValueError, "sample_fps"):
            self.run_episode(sample_fps=math.nan)
        self.plan.write_text("[]")
        with self.assertRaisesRegex(ValueError, "supplied revisit plan"):
            self.run_episode()
        self.assertEqual(len(self.client.calls), 0)

    def test_rendered_video_hash_must_match_supplied_plan(self):
        plan = json.loads(self.plan.read_text())
        plan["output_sha256"] = qa._sha256(self.video)
        self.plan.write_text(json.dumps(plan))
        self.run_episode()
        self.video.write_bytes(b"content from a different episode")
        with self.assertRaisesRegex(ValueError, "does not match the episode video"):
            self.run_episode()
        self.assertEqual(len(self.client.calls), 1)

    def test_domain_and_schema_are_validated_on_plan_output_and_resume(self):
        for domain in ("bike", "health", "music", "sports"):
            with self.subTest(domain=domain):
                spec = qa.get_domain(domain)
                plan = json.loads(self.plan.read_text())
                plan.update(domain=domain, source_schema_version=spec.segmentation_schema, language="en")
                self.plan.write_text(json.dumps(plan))
                self.client = FakeClient([quota_candidates(domain)] * 2)
                first = self.run_episode(domain=domain)
                self.assertEqual(first["schema_version"], spec.qa_schema)
                self.assertEqual(first["domain"], domain)
                self.assertTrue(all(q["domain"] == domain for q in first["questions"]))
                self.assertEqual(first["request"]["domain"], domain)
                self.assertEqual(self.run_episode(domain=domain), first)
                self.assertEqual(len(self.client.calls), 1)
                first["questions"][0]["domain"] = "cook"
                self.output.write_text(json.dumps(first))
                self.run_episode(domain=domain)
                self.assertEqual(len(self.client.calls), 2)

    def test_wrong_domain_missing_plan_or_schema_is_rejected_before_model_call(self):
        with self.assertRaisesRegex(ValueError, "does not belong to the bike domain"):
            self.run_episode(domain="bike")
        with self.assertRaisesRegex(ValueError, "valid music revisit plan is required"):
            qa.generate_episode_qa(self.video, self.output, self.client, episode_id="episode_001", domain="music")
        plan = json.loads(self.plan.read_text())
        plan["domain"] = "health"
        plan["source_schema_version"] = qa.get_domain("sports").segmentation_schema
        self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "incompatible source segmentation schema"):
            self.run_episode(domain="health")
        self.assertEqual(len(self.client.calls), 0)


if __name__ == "__main__":
    unittest.main()
