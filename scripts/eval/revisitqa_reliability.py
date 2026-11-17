"""RevisitQA scoring: every item offers three substantive options plus one abstention option.

Main metrics (ranking uses known_abstention_hmean):
  known_accuracy           Known items answered with the correct substantive option (abstaining counts as wrong).
  abstention_recall        Uncertain items on which the model abstains (1 - overconfidence).
  known_abstention_hmean   H = 2 * KA * AR / (KA + AR): 0 for both always-abstain and never-abstain policies.
Diagnostics:
  false_abstention_rate    Known items on which the model abstains.
  abstention_precision     Abstentions whose gold label is Uncertain.
  coverage / risk          Fraction answered, and error rate among answered items.
Reference only (appendix):
  effective_reliability    Phi_1 of Whitehead et al. (2022): +1 correct answer, -1 wrong answer, 0 abstention.
                           It assumes every item is answerable, so correct abstention on Uncertain items earns
                           nothing and always-abstaining scores 0, tying with models whose answers are unreliable.
  accuracy                 Abstention treated as the label of Uncertain items; dominated by the Known share.
Unparseable outputs count as wrong answers.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Sequence

INTENSITY_BINS = (("low", 1, 3), ("medium", 4, 6), ("high", 7, 10))
EVIDENCE_AGE_BINS = (("lt10s", 0.0, 10.0), ("ge10s", 10.0, float("inf")))
SCHEMA_VERSION = "revisitqa_selective_qa"


def _intensity_bin(value) -> str | None:
    for name, low, high in INTENSITY_BINS:
        if value is not None and low <= int(value) <= high:
            return name
    return None


def _evidence_age(row: dict) -> float | None:
    spans = row.get("evidence_spans") or []
    if not spans or row.get("query_time") is None:
        return None
    return float(row["query_time"]) - max(float(span[1]) for span in spans)


def evidence_window(row: dict, frames: int = 16) -> str | None:
    """'earlier' if an evidence span starts before the latest `frames` 1 FPS frames up to t_q, else 'last16'."""
    spans = row.get("evidence_spans") or []
    if not spans or row.get("query_time") is None:
        return None
    first_visible = math.floor(float(row["query_time"]) + 1e-6) - (frames - 1)
    return "earlier" if min(float(span[0]) for span in spans) < first_visible - 1e-6 else "last16"


def _outcome(gt: dict, pred: dict) -> dict:
    abstain_choice = gt.get("abstention_choice")
    if abstain_choice is None:
        raise ValueError(f"{gt.get('id')}: manifest has no abstention option; rebuild with revisitqa_unify_abstention.py")
    status = gt["diagnostics"]["epistemic_status"]
    choice = pred.get("pred_index")
    abstained = choice == abstain_choice
    correct = choice == gt["answer"]
    return {
        "known": status == "known",
        "abstained": abstained,
        "invalid": choice is None,
        "correct": correct,
        # Effective reliability: abstaining scores 0 regardless of status.
        "er": 0 if abstained else (1 if correct else -1),
    }


def _hmean(a, b):
    if a is None or b is None:
        return None
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def _summarize(outcomes: Sequence[dict]) -> dict:
    known = [o for o in outcomes if o["known"]]
    uncertain = [o for o in outcomes if not o["known"]]
    answered = [o for o in outcomes if not o["abstained"]]
    abstained = [o for o in outcomes if o["abstained"]]

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    return {
        "count": len(outcomes),
        "known_count": len(known),
        "uncertain_count": len(uncertain),
        "accuracy": mean(o["correct"] for o in outcomes),
        "known_accuracy": mean(o["correct"] for o in known),
        "abstention_recall": mean(o["abstained"] for o in uncertain),
        "false_abstention_rate": mean(o["abstained"] for o in known),
        "abstention_precision": mean(not o["known"] for o in abstained),
        "coverage": mean(not o["abstained"] for o in outcomes),
        "risk": mean(not o["correct"] for o in answered),
        "effective_reliability": mean(o["er"] for o in outcomes),
        # Harmonic mean of known_accuracy and abstention_recall: 0 for both always-abstain and never-abstain policies.
        "known_abstention_hmean": _hmean(mean(o["correct"] for o in known), mean(o["abstained"] for o in uncertain)),
        "invalid_rate": mean(o["invalid"] for o in outcomes),
    }


def _bootstrap(outcomes_by_episode: dict[str, list[dict]], replicates: int, seed: int) -> dict:
    keys = ("effective_reliability", "known_abstention_hmean", "known_accuracy", "abstention_recall", "accuracy")
    episodes = sorted(outcomes_by_episode)
    if replicates <= 0 or len(episodes) < 2:
        return {}
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(replicates):
        drawn = [o for _ in episodes for o in outcomes_by_episode[rng.choice(episodes)]]
        summary = _summarize(drawn)
        for key in keys:
            if summary[key] is not None:
                samples[key].append(summary[key])
    result = {}
    for key, values in samples.items():
        values.sort()
        result[key] = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
    return result


def score_selective(
    gt_rows: Sequence[dict],
    predictions: Sequence[dict],
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260817,
) -> dict:
    """Score predictions against normalized ground truth (rows from data_io.normalize_mcq_sample)."""
    gt_by_id = {str(row["id"]): row for row in gt_rows}
    pairs = [(gt_by_id[str(pred["id"])], pred) for pred in predictions]
    outcomes = [(gt, _outcome(gt, pred)) for gt, pred in pairs]
    result = {"schema_version": SCHEMA_VERSION, **_summarize([o for _, o in outcomes])}

    def grouped(key: Callable[[dict], str | None]) -> dict:
        groups = defaultdict(list)
        for gt, outcome in outcomes:
            name = key(gt)
            if name is not None:
                groups[name].append(outcome)
        return {name: _summarize(items) for name, items in sorted(groups.items())}

    def age_bin(gt):
        age = _evidence_age(gt)
        for name, low, high in EVIDENCE_AGE_BINS:
            if age is not None and low <= age < high:
                return name
        return None

    result["by_intensity"] = grouped(lambda gt: _intensity_bin(gt.get("change_intensity")))
    result["by_question_type"] = grouped(lambda gt: gt.get("question_type"))
    result["by_evidence_age"] = grouped(age_bin)
    result["by_evidence_age_ge16s"] = grouped(lambda gt: "ge16s" if (_evidence_age(gt) or 0) >= 16 else None)
    by_episode = defaultdict(list)
    for gt, outcome in outcomes:
        by_episode[str(gt.get("episode_id"))].append(outcome)
    result["ci95"] = _bootstrap(by_episode, bootstrap_replicates, bootstrap_seed)
    choices = defaultdict(int)
    for _, pred in pairs:
        choices[pred.get("pred_choice") or "INVALID"] += 1
    result["choice_distribution"] = dict(sorted(choices.items()))
    return result


def format_headline(score: dict) -> str:
    def pct(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"

    return (
        f"H={pct(score['known_abstention_hmean'])} known_acc={pct(score['known_accuracy'])} "
        f"abst_recall={pct(score['abstention_recall'])} false_abst={pct(score['false_abstention_rate'])} "
        f"ER={score['effective_reliability']:+.4f} acc={pct(score['accuracy'])} invalid={pct(score['invalid_rate'])}"
    )
