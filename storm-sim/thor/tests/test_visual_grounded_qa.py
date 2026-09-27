from __future__ import annotations

import unittest

from tools.storm.benchgen.stream_eqa import visual_grounded_qa as v21


def _event(index: int, target: str, event_type: str, start: float, end: float):
    return {
        "start_sec": start,
        "end_sec": end,
        "event_type": event_type,
        "subject": target,
        "_target_id": f"{target}|instance",
        "_index": index,
        "_surface": "countertop",
    }


def _events():
    return [
        _event(0, "Apple", "disappearance", 2.0, 5.0),
        _event(1, "Bowl", "disappearance", 8.0, 11.0),
        _event(2, "Bread", "disappearance", 14.0, 17.0),
        _event(3, "Cup", "disappearance", 20.0, 23.0),
        _event(4, "Apple", "appearance", 32.0, 35.0),
        _event(5, "Bowl", "appearance", 37.0, 40.0),
        _event(6, "Bread", "appearance", 43.0, 46.0),
        _event(7, "Cup", "appearance", 49.0, 52.0),
    ]


class VisualGroundedQATest(unittest.TestCase):
    def test_known_draft_has_four_concrete_options_and_no_refusal(self):
        events = _events()
        question = v21._build_factual_known(
            "episode", "FloorPlan1", events, 0, 60, v21.REFUSAL_OPTIONS[0],
        )
        self.assertEqual(len(question["_distractors"]), 3)
        self.assertNotIn(v21.REFUSAL_OPTIONS[0], question["_distractors"])
        self.assertNotIn(question["_correct"], v21.REFUSAL_OPTIONS)
        self.assertTrue(all(start < end for start, end in question["evidence_spans"]))

    def test_current_state_uses_four_real_objects_and_unique_state(self):
        events = _events()
        semantic_classes = set()
        for ordinal in range(2):
            question = v21._build_current_known(
                "episode", "FloorPlan1", events, ordinal, 60, v21.REFUSAL_OPTIONS[0],
            )
            semantic_classes.add(question["_semantic_class"])
            self.assertIn("which listed object", question["question"])
            self.assertNotIn("supported", question["question"].lower())
            self.assertEqual(len(question["_distractors"]), 3)
            self.assertEqual(len({question["_correct"], *question["_distractors"]}), 4)
            self.assertTrue(all(start < end for start, end in question["evidence_spans"]))
        self.assertEqual(semantic_classes, {"unique_object_absent", "unique_object_visible"})

    def test_uncertain_question_never_mentions_time_after_query(self):
        events = _events()
        builders = (
            lambda: v21._build_location_uncertain(
                "episode", "FloorPlan1", events, "factual_retrieval", 0, 60,
                v21.REFUSAL_OPTIONS[0], ["missing_observation"],
            ),
            lambda: v21._build_history_uncertain(
                "episode", "FloorPlan1", events, 0, 60,
                v21.REFUSAL_OPTIONS[0], ["missing_observation"],
            ),
            lambda: v21._build_temporal_uncertain(
                "episode", "FloorPlan1", events, 0, 60,
                v21.REFUSAL_OPTIONS[0], ["missing_observation"],
            ),
        )
        for builder in builders:
            question = builder()
            self.assertEqual(question["_correct"], v21.REFUSAL_OPTIONS[0])
            self.assertTrue(all(value <= question["query_time"] for value in v21._mentioned_times(question)))
            self.assertTrue(all(start < end <= question["query_time"] for start, end in question["evidence_spans"]))

    def test_history_uses_same_cardinality_object_sets(self):
        events = _events()
        pair = v21._build_history_known(
            "episode", "FloorPlan1", events, 0, 60, v21.REFUSAL_OPTIONS[0],
        )
        triple = v21._build_history_known(
            "episode", "FloorPlan1", events, 1, 60, v21.REFUSAL_OPTIONS[0],
        )
        self.assertEqual(len(pair["_correct"].split(" and ")), 2)
        self.assertTrue(all(len(option.split(" and ")) == 2 for option in [pair["_correct"], *pair["_distractors"]]))
        self.assertEqual(len(triple["_correct"].split(" and ")), 3)
        self.assertTrue(all(len(option.split(" and ")) == 3 for option in [triple["_correct"], *triple["_distractors"]]))
        self.assertIn("Which set of objects", pair["question"])

    def test_model_prompt_keeps_private_fields_out(self):
        prompt = v21.build_model_prompt({
            "question": "Which object changes?",
            "options": ["apple", "bowl", "bread", "cup"],
            "answer_index": 1,
            "diagnostics": {"epistemic_status": "known"},
            "video_evidence": "private",
        })
        self.assertNotIn("answer_index", prompt)
        self.assertNotIn("private", prompt)
        self.assertNotIn("known", prompt)


if __name__ == "__main__":
    unittest.main()
