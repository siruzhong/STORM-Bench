from __future__ import annotations

import unittest

from tools.storm.benchgen.stream_eqa import visual_contrast_qa as v22


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


class VisualContrastQATest(unittest.TestCase):
    def setUp(self):
        v22._configure_base()

    def test_state_change_has_four_object_labels(self):
        question = v22._build_state_known(
            "episode", "FloorPlan1", _events(), 0, 60, v22.base.REFUSAL_OPTIONS[0],
        )
        options = [question["_correct"], *question["_distractors"]]
        self.assertEqual(set(options), {"apple", "bowl", "bread", "cup"})
        self.assertNotIn("disappears", " ".join(options).lower())
        self.assertNotIn("reappears", " ".join(options).lower())
        self.assertEqual(question["question_subtype"], v22.SUBTYPES["state_change"])
        self.assertTrue(all(start < end for start, end in question["evidence_spans"]))

    def test_temporal_reasoning_has_four_object_labels(self):
        question = v22._build_temporal_known(
            "episode", "FloorPlan1", _events(), 1, 60, v22.base.REFUSAL_OPTIONS[0],
        )
        options = [question["_correct"], *question["_distractors"]]
        self.assertEqual(set(options), {"apple", "bowl", "bread", "cup"})
        self.assertEqual(question["question_subtype"], v22.SUBTYPES["temporal_reasoning"])
        self.assertIn("immediately before", question["question"])
        self.assertTrue(all(start < end for start, end in question["evidence_spans"]))

    def test_configuration_changes_export_identity_without_touching_counts(self):
        self.assertEqual(v22.base.VERSION, v22.VERSION)
        self.assertEqual(v22.base.ID_VERSION_TAG, "v22e1")
        self.assertEqual(v22.base.QA_SOURCE, v22.QA_SOURCE)
        self.assertEqual(v22.base.REPORT_FILENAME, v22.REPORT_FILENAME)


if __name__ == "__main__":
    unittest.main()
