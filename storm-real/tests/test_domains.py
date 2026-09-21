"""Check domain coverage, prompt contracts, and portable prompt resources."""

import re
import unittest
from dataclasses import FrozenInstanceError
from importlib.resources import files
from string import Template

from storm_real_pipeline.domains import DOMAIN_NAMES, DOMAINS, get_domain


def prompt(domain, stage):
    return files("storm_real_pipeline").joinpath(
        "prompts", domain, stage + ".txt"
    ).read_text(encoding="utf-8")


class DomainCatalogTests(unittest.TestCase):
    def test_all_domains_have_distinct_schemas(self):
        self.assertEqual(DOMAIN_NAMES, ("cook", "bike", "health", "music", "sports"))
        self.assertEqual(get_domain().name, "cook")
        self.assertEqual(get_domain(" Sports ").name, "sports")
        self.assertEqual(len({spec.segmentation_schema for spec in DOMAINS.values()}), 5)
        self.assertEqual(len({spec.qa_schema for spec in DOMAINS.values()}), 5)
        for name, spec in DOMAINS.items():
            with self.subTest(domain=name):
                self.assertEqual(spec.name, name)
                self.assertIsInstance(spec.regions, frozenset)
                self.assertIsInstance(spec.activities, frozenset)
                self.assertIn("other_area", spec.regions)
                self.assertTrue({"observe", "navigate", "unusable"} <= spec.activities)
                self.assertTrue(set(spec.region_priority) <= spec.regions - {"other_area"})
                self.assertTrue(all(value > 0 for value in spec.region_priority.values()))
                self.assertEqual(spec.region_aliases, {})
                with self.assertRaises(FrozenInstanceError):
                    spec.min_visits = 99
        for name in ("unknown", "", "../cook", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_domain(name)

    def test_latest_visit_and_retry_thresholds(self):
        expected = {
            "cook": (3, 3, 5, False),
            "bike": (2, 3, 5, True),
            "health": (2, 5, 12, True),
            "music": (2, 5, 12, True),
            "sports": (2, 5, 12, True),
        }
        for name, values in expected.items():
            spec = get_domain(name)
            self.assertEqual(
                (spec.min_visits, spec.segmentation_retries,
                 spec.qa_generation_attempts, spec.exact_uncertain_answer),
                values,
            )

    def test_discipline_regions_remain_separate(self):
        sports = get_domain("sports")
        expected_counts = {"basketball": 4, "soccer": 5, "climbing": 4}
        for prefix, count in expected_counts.items():
            regions = {value for value in sports.regions if value.startswith(prefix + "_")}
            self.assertEqual(len(regions), count)
            self.assertTrue(regions <= sports.region_priority.keys())
        self.assertEqual(len(sports.regions), 14)
        self.assertEqual(sports.region_priority["basketball_ball_drill_area"], 7)
        self.assertEqual(sports.region_priority["soccer_cone_drill_area"], 7)
        self.assertEqual(sports.region_priority["climbing_hold_sequence_area"], 7)


class DomainPromptTests(unittest.TestCase):
    def test_every_enum_is_defined_in_its_segmentation_prompt(self):
        for name, spec in DOMAINS.items():
            text = prompt(name, "segmentation")
            with self.subTest(domain=name):
                for value in spec.regions | spec.activities:
                    self.assertRegex(text, r"\b" + re.escape(value) + r"\b")
                for field in ("start_sec", "end_sec", "region", "segment_role",
                              "activity_type", "change_types", "description"):
                    self.assertIn(field, text)
                for change in ("position_change", "state_change", "identity_replacement"):
                    self.assertIn(change, text)
                self.assertIn("interaction", text)
                self.assertIn("observation", text)
                self.assertIn("150 characters", text)

    def test_qa_prompts_cover_types_and_uncertainty_contract(self):
        question_types = {
            "factual_retrieval", "current_state", "state_change", "object_tracking",
            "temporal_reasoning", "history_aggregation",
        }
        for name, spec in DOMAINS.items():
            text = prompt(name, "qa")
            with self.subTest(domain=name):
                for value in question_types:
                    self.assertIn(value, text)
                for field in ("query_time", "answer_index", "evidence_spans",
                              "epistemic_status", "uncertainty_sources"):
                    self.assertIn(field, text)
                self.assertIn("known|uncertain", text)
                if spec.exact_uncertain_answer:
                    self.assertIn('correct option must be exactly "Cannot be determined"', text)
                else:
                    self.assertIn("question-specific wording", text)

    def test_revisit_templates_accept_runtime_parameters(self):
        common = {
            "region": "example_area", "min_visits": 2, "max_visits": 10,
            "min_visit_sec": 3, "max_visit_sec": 20, "target_sec": 45,
            "max_total_sec": 60, "speed": 1.5, "blank_sec": 0.2,
            "src_sum_lo": 64, "src_sum_hi": 68,
            "target_lower_sec": 42, "target_upper_sec": 48,
        }
        for name in DOMAIN_NAMES:
            source = prompt(name, "revisit")
            with self.subTest(domain=name):
                used = {
                    match.group("named") or match.group("braced")
                    for match in Template.pattern.finditer(source)
                    if match.group("named") or match.group("braced")
                }
                self.assertTrue(used <= common.keys())
                self.assertTrue({"region", "min_visits", "max_visits", "target_sec"} <= used)
                rendered = Template(source).substitute(common)
                self.assertNotIn("${", rendered)
                self.assertIn("example_area", rendered)
                self.assertIn("observation", rendered)
                self.assertIn("interaction", rendered)
                self.assertIn("true", rendered)
                if name != "cook":
                    self.assertIn("42 to 48 seconds", rendered)

    def test_domain_specific_evidence_rules_are_preserved(self):
        for stage in ("segmentation", "revisit", "qa"):
            text = prompt("sports", stage)
            for discipline in ("basketball", "soccer", "climbing"):
                self.assertIn(discipline, text)
        self.assertIn("never mix region families", prompt("sports", "segmentation"))
        self.assertIn("Never combine basketball, soccer, and climbing", prompt("sports", "revisit"))
        for name in ("health", "music", "sports"):
            self.assertIn("Avoid direct same/different questions", prompt(name, "qa"))
        self.assertIn("jersey/color alone", prompt("sports", "qa"))
        self.assertIn("Never make a medical diagnosis", prompt("health", "segmentation"))
        self.assertIn("never infer clinical success", prompt("health", "qa"))
        self.assertIn("Do not split every note", prompt("music", "segmentation"))
        self.assertIn("never infer identity merely", prompt("music", "qa").lower())
        self.assertIn("Tool contact without a visible outcome", prompt("bike", "segmentation"))

    def test_prompts_contain_no_private_paths_tokens_or_nonenglish_process(self):
        private_markers = re.compile(
            r"/(?:Users|home|root|mnt|tmp)/"
            r"|\b(?:\d{1,3}\.){3}\d{1,3}\b"
            r"|\bsk-[A-Za-z0-9_-]{16,}\b"
            r"|\bBearer\s+[A-Za-z0-9._-]{16,}\b"
            r"|[\u3400-\u9fff]"
        )
        for name in DOMAIN_NAMES:
            for stage in ("segmentation", "revisit", "qa"):
                with self.subTest(domain=name, stage=stage):
                    text = prompt(name, stage)
                    self.assertGreater(len(text), 100)
                    self.assertIsNone(private_markers.search(text))


if __name__ == "__main__":
    unittest.main()
