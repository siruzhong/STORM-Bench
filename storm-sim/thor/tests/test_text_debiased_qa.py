from __future__ import annotations

import unittest
from collections import Counter, defaultdict

from tools.storm.benchgen.stream_eqa import text_debiased_qa as v2


def _event(index: int, target: str, event_type: str, start: float, end: float):
    return {
        "start_sec": start,
        "end_sec": end,
        "event_type": event_type,
        "subject": target,
        "_target_id": f"{target}|instance",
        "_index": index,
        "_surface": "操作台",
    }


class TextDebiasedQATest(unittest.TestCase):
    def test_current_state_uses_real_joint_visibility_configurations(self):
        events = [
            _event(0, "Apple", "disappearance", 4.0, 5.0),
            _event(1, "Bowl", "disappearance", 9.0, 10.0),
            _event(2, "Bread", "disappearance", 14.0, 15.0),
            _event(3, "Cup", "disappearance", 19.0, 20.0),
            _event(4, "Apple", "appearance", 29.0, 30.0),
            _event(5, "Bowl", "appearance", 34.0, 35.0),
            _event(6, "Bread", "appearance", 39.0, 40.0),
            _event(7, "Cup", "appearance", 44.0, 45.0),
        ]
        classes = []
        for ordinal in range(4):
            question = v2._build_current_known(
                "episode", "FloorPlan1", events, ordinal, 60,
                v2.EVIDENCE_OPTIONS[0],
            )
            classes.append(question["_semantic_class"])
            self.assertEqual(len(question["_distractors"]), 3)
            self.assertNotIn("agent's hand", " ".join(question["_distractors"]))
            self.assertNotIn("broken", " ".join(question["_distractors"]))
        self.assertEqual(set(classes), {f"joint_visibility_{index}" for index in range(4)})

    def test_stratified_positions_are_balanced_and_time_decorrelated(self):
        sizes = {
            ("current_state", "known"): 53,
            ("current_state", "uncertain"): 1,
            ("factual_retrieval", "known"): 52,
            ("factual_retrieval", "uncertain"): 2,
            ("history_aggregation", "known"): 39,
            ("history_aggregation", "uncertain"): 12,
            ("state_change", "known"): 48,
            ("state_change", "uncertain"): 3,
            ("object_tracking", "known"): 26,
            ("object_tracking", "uncertain"): 24,
            ("temporal_reasoning", "known"): 27,
            ("temporal_reasoning", "uncertain"): 13,
        }
        questions = []
        cursor = 0
        for (question_type, status), count in sizes.items():
            for index in range(count):
                evidence = v2.EVIDENCE_OPTIONS[index % len(v2.EVIDENCE_OPTIONS)]
                uncertain = status == "uncertain"
                questions.append({
                    "episode_id": f"episode_{cursor:03d}",
                    "question_type": question_type,
                    "question_subtype": v2.SUBTYPES[question_type],
                    "query_time": float(5 + (cursor * 7) % 55),
                    "question": f"question {cursor}",
                    "diagnostics": {"epistemic_status": status, "uncertainty_sources": ["missing_observation"] if uncertain else []},
                    "_correct": evidence if uncertain else "concrete correct",
                    "_distractors": ["a", "b", "c"] if uncertain else ["a", "b", evidence],
                })
                cursor += 1
        summary = v2._assign_balanced_options(questions, v2.DEFAULT_SEED)
        self.assertEqual(Counter(question["answer_index"] for question in questions), Counter({0: 75, 1: 75, 2: 75, 3: 75}))
        for counts in summary.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        bins = defaultdict(Counter)
        for question in questions:
            bins[int(question["query_time"] // 10)][question["answer_index"]] += 1
        self.assertLessEqual(sum(max(counts.values()) for counts in bins.values()) / 300, 0.30)

    def test_evidence_wording_is_status_independent(self):
        self.assertGreaterEqual(len(v2.EVIDENCE_OPTIONS), 4)
        for option in v2.EVIDENCE_OPTIONS:
            self.assertNotIn("Cannot be determined from the visible frames", option)
            self.assertNotIn("off-screen", option.lower())
            self.assertNotIn("provably", option.lower())

    def test_prompt_builder_uses_an_explicit_field_allowlist(self):
        question = {
            "question": "Which state is visible?",
            "options": ["one", "two", "three", "four"],
            "answer_index": 2,
            "video_evidence": "private answer-bearing metadata",
            "diagnostics": {"epistemic_status": "known"},
            "diagnostic_rationale": {"uncertainty": "private"},
        }
        prompt = v2.build_model_prompt(question)
        self.assertIn("Which state is visible?", prompt)
        self.assertIn("C. three", prompt)
        self.assertNotIn("answer_index", prompt)
        self.assertNotIn("private", prompt)
        self.assertNotIn("known", prompt)


if __name__ == "__main__":
    unittest.main()
