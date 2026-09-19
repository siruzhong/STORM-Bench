"""Pure scoring and sampling utilities for the independent STORM path."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Sequence


CHANGE_INTENSITY_BUCKETS = ("1-3", "4-6", "7-10")
EPISTEMIC_STATUSES = ("known", "uncertain")
UNCERTAINTY_SOURCE_CHOICES = (
    ("missing_observation", "Missing observation: key frames are missing, the target is off-screen, or the needed moment was not recorded"),
    ("partial_observation", "Partial observation: only part of the target or process is visible, including occlusion"),
    ("low_visual_quality", "Low visual quality: the relevant content is on screen but too dark, blurry, or low-resolution to judge"),
    ("ambiguous_evidence", "Ambiguous evidence: timestamps, identity, or observations conflict and do not yield one interpretation"),
    ("ambiguous_attribute", "Ambiguous attribute: the target's category, color, or state cannot be resolved"),
    ("multiple_candidates", "Multiple candidates: more than one option is compatible with the visible frames"),
)
UNCERTAINTY_SOURCE_LABELS = tuple(label for label, _description in UNCERTAINTY_SOURCE_CHOICES)
CONTROLLED_UNCERTAINTY_SOURCES = set(UNCERTAINTY_SOURCE_LABELS)


def change_intensity_value(row: dict) -> int:
    value = row.get("change_intensity")
    if value is None:
        value = (row.get("diagnostics") or {}).get("change_intensity")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Sample {row.get('id')} has invalid change_intensity={value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Sample {row.get('id')} has invalid change_intensity={value!r}")
        value = int(value)
    if value < 1:
        raise ValueError(f"Sample {row.get('id')} has invalid change_intensity={value!r}")
    return value


def change_intensity_bucket(intensity: int) -> str:
    if intensity <= 3:
        return "1-3"
    if intensity <= 6:
        return "4-6"
    return "7-10"


def volatility_bucket(row: dict) -> str:
    return change_intensity_bucket(change_intensity_value(row))


def epistemic_statuses(rows: Sequence[dict]) -> tuple[str, ...]:
    observed = {str(row["diagnostics"]["epistemic_status"]) for row in rows}
    return tuple(status for status in EPISTEMIC_STATUSES if status in observed) or tuple(sorted(observed))


def audit_ground_truth(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("STORM ground truth is empty")
    seen: set[str] = set()
    sources: list[str] = []
    cell_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate ground-truth id: {sample_id!r}")
        seen.add(sample_id)
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"Sample {sample_id} has no diagnostics object")
        change_intensity_value(row)
        status = diagnostics.get("epistemic_status")
        if status not in EPISTEMIC_STATUSES:
            raise ValueError(f"Sample {sample_id} has invalid epistemic_status={status!r}")
        raw_sources = diagnostics.get("uncertainty_sources", [])
        if not isinstance(raw_sources, list):
            raise ValueError(f"Sample {sample_id} has invalid uncertainty_sources")
        if status == "known" and raw_sources:
            raise ValueError(f"Sample {sample_id} marks known but lists uncertainty_sources")
        if status == "uncertain" and not raw_sources:
            raise ValueError(f"Sample {sample_id} marks uncertain without uncertainty_sources")
        sources.extend(str(source) for source in raw_sources)
        cell_counts[(volatility_bucket(row), str(status))] += 1

    statuses = epistemic_statuses(rows)
    source_mode = (
        "controlled_vocabulary"
        if all(source in CONTROLLED_UNCERTAINTY_SOURCES for source in sources)
        else "free_text"
    )
    warnings = []
    if source_mode != "controlled_vocabulary":
        warnings.append("uncertainty_sources contains labels outside the controlled vocabulary.")
    required_cells = [
        (bucket, status)
        for bucket in CHANGE_INTENSITY_BUCKETS
        for status in EPISTEMIC_STATUSES
    ]
    missing_cells = [
        f"volatility={bucket}|status={status}"
        for bucket, status in required_cells
        if cell_counts[(bucket, status)] == 0
    ]
    if missing_cells:
        warnings.append("STORM-BR is unavailable because one or more required cells are empty.")
    return {
        "status_labels": list(statuses),
        "volatility_axis": "change_intensity",
        "volatility_buckets": list(CHANGE_INTENSITY_BUCKETS),
        "uncertainty_source_mode": source_mode,
        "required_cell_count": len(required_cells),
        "cell_counts": {
            f"volatility={bucket}|status={status}": cell_counts[(bucket, status)]
            for bucket, status in required_cells
        },
        "missing_cells": missing_cells,
        "headline_eligible": not missing_cells,
        "warnings": warnings,
    }


def stratified_sample(rows: Sequence[dict], limit: int | None, seed: int) -> list[dict]:
    """Round-robin over observed volatility/status cells for diagnostic smoke tests."""
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(volatility_bucket(row), row["diagnostics"]["epistemic_status"])].append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    selected: list[dict] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            group = groups[key]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _harmonic_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    if any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    denominator = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / denominator if denominator else None


def _prediction_index(prediction: dict) -> int | None:
    value = prediction.get("pred_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _prediction_status(prediction: dict) -> str | None:
    value = prediction.get("pred_epistemic_status", prediction.get("predicted_epistemic_status"))
    return value if isinstance(value, str) and value else None


def _prediction_sources(prediction: dict) -> tuple[str, ...]:
    value = prediction.get(
        "pred_uncertainty_sources", prediction.get("predicted_uncertainty_sources")
    )
    if value is None:
        value = prediction.get(
            "pred_uncertainty_source", prediction.get("predicted_uncertainty_source")
        )
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(
        dict.fromkeys(
            str(source) for source in value if str(source) in CONTROLLED_UNCERTAINTY_SOURCES
        )
    )


def _source_was_probed(prediction: dict) -> bool:
    return any(
        key in prediction
        for key in (
            "pred_uncertainty_sources",
            "predicted_uncertainty_sources",
            "pred_uncertainty_source",
            "predicted_uncertainty_source",
            "pred_uncertainty_raw",
        )
    )


def _ground_truth_sources(row: dict) -> tuple[str, ...]:
    raw = (row.get("diagnostics") or {}).get("uncertainty_sources") or []
    return tuple(str(source) for source in raw)


def _answer_index(row: dict) -> int:
    return int(row["answer"] if "answer" in row else row["answer_index"])


def _volatility_tier(row: dict) -> int:
    return CHANGE_INTENSITY_BUCKETS.index(volatility_bucket(row))


def _set_f1(predicted: Sequence[str], expected: Sequence[str]) -> float:
    predicted_set = set(predicted)
    expected_set = set(expected)
    if not predicted_set and not expected_set:
        return 1.0
    denominator = len(predicted_set) + len(expected_set)
    return 2.0 * len(predicted_set & expected_set) / denominator if denominator else 0.0


def _score_core(rows: Sequence[dict], predictions: Sequence[dict], statuses: Sequence[str]) -> dict:
    prediction_by_id = {str(row["id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Duplicate prediction ids")
    expected_ids = {str(row["id"]) for row in rows}
    if set(prediction_by_id) != expected_ids:
        missing = sorted(expected_ids - set(prediction_by_id))[:5]
        extra = sorted(set(prediction_by_id) - expected_ids)[:5]
        raise ValueError(f"Prediction coverage mismatch: missing={missing}, extra={extra}")

    buckets = CHANGE_INTENSITY_BUCKETS
    samples = []
    for row in rows:
        prediction = prediction_by_id[str(row["id"])]
        pred_index = _prediction_index(prediction)
        pred_status = _prediction_status(prediction)
        pred_sources = _prediction_sources(prediction)
        gt_sources = _ground_truth_sources(row)
        gt_status = row["diagnostics"]["epistemic_status"]
        answer_correct = pred_index == _answer_index(row)
        status_correct = pred_status == gt_status
        source_asked = pred_status == "uncertain" and _source_was_probed(prediction)
        source_correct = bool(
            source_asked and gt_status == "uncertain" and set(pred_sources) & set(gt_sources)
        )
        cause_f1 = _set_f1(pred_sources, gt_sources) if gt_status == "uncertain" else None
        attribution_joint = (
            float(answer_correct and status_correct) * cause_f1
            if gt_status == "uncertain"
            else float(answer_correct and status_correct)
        )
        age = max(
            0.0,
            float(row.get("query_time", 0.0))
            - max(float(span[1]) for span in row.get("evidence_spans", [[0.0, 0.0]])),
        )
        bucket = volatility_bucket(row)
        samples.append(
            {
                "answer_correct": float(answer_correct),
                "status_correct": float(status_correct),
                "joint_correct": float(answer_correct and status_correct),
                "volatility_bucket": bucket,
                "status": gt_status,
                "pred_status": pred_status,
                "pred_sources": pred_sources,
                "gt_sources": gt_sources,
                "source_asked": source_asked,
                "source_correct": float(source_correct),
                "cause_f1": cause_f1,
                "attribution_joint": attribution_joint,
                "answer_invalid": pred_index is None,
                "status_invalid": pred_status not in statuses,
                "evidence_age": age,
                "time_weight": 1.0 + min(age / 5.0, 1.0),
                "difficulty_weight": (
                    1.0
                    + 0.5 * _volatility_tier(row)
                    + 0.5 * (gt_status != "known")
                ),
            }
        )

    volatility_groups = {}
    for bucket in buckets:
        group = [sample for sample in samples if sample["volatility_bucket"] == bucket]
        volatility_groups[bucket] = {
            "count": len(group),
            "accuracy": _mean(sample["answer_correct"] for sample in group),
        }
    status_groups = {}
    for status in statuses:
        group = [sample for sample in samples if sample["status"] == status]
        status_groups[status] = {
            "count": len(group),
            "answer_accuracy": _mean(sample["answer_correct"] for sample in group),
            "status_accuracy": _mean(sample["status_correct"] for sample in group),
            "joint_accuracy": _mean(sample["joint_correct"] for sample in group),
        }
    cell_groups = {}
    cell_grid = {}
    for bucket in buckets:
        cell_grid[bucket] = {}
        for status in statuses:
            group = [
                sample
                for sample in samples
                if sample["volatility_bucket"] == bucket and sample["status"] == status
            ]
            if not group:
                continue
            cell = {
                "change_intensity_bucket": bucket,
                "epistemic_status": status,
                "count": len(group),
                "answer_accuracy": _mean(sample["answer_correct"] for sample in group),
                "status_accuracy": _mean(sample["status_correct"] for sample in group),
                "joint_accuracy": _mean(sample["joint_correct"] for sample in group),
                "attribution_joint_accuracy": _mean(
                    sample["attribution_joint"] for sample in group
                ),
            }
            cell_groups[f"volatility={bucket}|status={status}"] = cell
            cell_grid[bucket][status] = cell
    cell_grid = {bucket: statuses for bucket, statuses in cell_grid.items() if statuses}

    cell_values = [group["joint_accuracy"] for group in cell_groups.values()]
    attribution_cell_values = [
        group["attribution_joint_accuracy"] for group in cell_groups.values()
    ]
    required_cell_count = len(buckets) * len(EPISTEMIC_STATUSES)
    cells_complete = len(cell_values) == required_cell_count
    storm_score_arithmetic = _mean(cell_values)
    storm_score_observed = _harmonic_mean(cell_values)
    storm_score = storm_score_observed if cells_complete else None
    storm_score_attr_observed = _harmonic_mean(attribution_cell_values)
    storm_score_attr = storm_score_attr_observed if cells_complete else None
    volatility = _mean(
        group["accuracy"] for group in volatility_groups.values() if group["accuracy"] is not None
    )
    uncertainty = _mean(
        group["joint_accuracy"]
        for group in status_groups.values()
        if group["joint_accuracy"] is not None
    )
    known_only_answer = {}
    for bucket in buckets:
        group = [
            sample
            for sample in samples
            if sample["volatility_bucket"] == bucket and sample["status"] == "known"
        ]
        known_only_answer[bucket] = {
            "count": len(group),
            "accuracy": _mean(sample["answer_correct"] for sample in group),
        }
    low_known = known_only_answer["1-3"]["accuracy"]
    high_known = known_only_answer["7-10"]["accuracy"]
    known_only_accuracy_drop = (
        None if low_known is None or high_known is None else low_known - high_known
    )
    known_volatility_score = _mean(
        group["accuracy"]
        for group in known_only_answer.values()
        if group["accuracy"] is not None
    )
    overconfidence_by_bucket = {}
    underconfidence_by_bucket = {}
    for bucket in buckets:
        group = [
            sample
            for sample in samples
            if sample["volatility_bucket"] == bucket and sample["status"] == "uncertain"
        ]
        overconfidence_by_bucket[bucket] = {
            "count": len(group),
            "rate": _mean(float(sample["pred_status"] == "known") for sample in group),
        }
        known_group = [
            sample
            for sample in samples
            if sample["volatility_bucket"] == bucket and sample["status"] == "known"
        ]
        underconfidence_by_bucket[bucket] = {
            "count": len(known_group),
            "rate": _mean(
                float(sample["pred_status"] == "uncertain") for sample in known_group
            ),
        }
    uncertain_samples = [sample for sample in samples if sample["status"] == "uncertain"]
    overconfidence_rate = _mean(
        float(sample["pred_status"] == "known") for sample in uncertain_samples
    )
    known_samples = [sample for sample in samples if sample["status"] == "known"]
    underconfidence_rate = _mean(
        float(sample["pred_status"] == "uncertain") for sample in known_samples
    )
    source_scored = [
        sample
        for sample in samples
        if sample["source_asked"] and sample["status"] == "uncertain"
    ]
    uncertainty_source_accuracy = _mean(sample["source_correct"] for sample in source_scored)
    uncertainty_cause_sample_f1 = _mean(
        sample["cause_f1"] for sample in uncertain_samples
    )
    cause_label_scores = {}
    for source in UNCERTAINTY_SOURCE_LABELS:
        true_positive = sum(
            source in sample["pred_sources"] and source in sample["gt_sources"]
            for sample in uncertain_samples
        )
        false_positive = sum(
            source in sample["pred_sources"] and source not in sample["gt_sources"]
            for sample in uncertain_samples
        )
        false_negative = sum(
            source not in sample["pred_sources"] and source in sample["gt_sources"]
            for sample in uncertain_samples
        )
        denominator = 2 * true_positive + false_positive + false_negative
        cause_label_scores[source] = {
            "support": true_positive + false_negative,
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            ),
            "recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            ),
            "f1": 2 * true_positive / denominator if denominator else 0.0,
        }
    supported_cause_scores = [
        score["f1"] for score in cause_label_scores.values() if score["support"] > 0
    ]
    uncertainty_cause_macro_f1 = _mean(supported_cause_scores)
    cause_micro_tp = sum(
        len(set(sample["pred_sources"]) & set(sample["gt_sources"]))
        for sample in uncertain_samples
    )
    cause_micro_fp = sum(
        len(set(sample["pred_sources"]) - set(sample["gt_sources"]))
        for sample in uncertain_samples
    )
    cause_micro_fn = sum(
        len(set(sample["gt_sources"]) - set(sample["pred_sources"]))
        for sample in uncertain_samples
    )
    cause_micro_denominator = 2 * cause_micro_tp + cause_micro_fp + cause_micro_fn
    uncertainty_cause_micro_f1 = (
        2 * cause_micro_tp / cause_micro_denominator
        if cause_micro_denominator
        else None
    )
    source_groups = {}
    for source in UNCERTAINTY_SOURCE_LABELS:
        group = [sample for sample in samples if source in sample["gt_sources"]]
        asked = [sample for sample in group if sample["source_asked"]]
        if not group:
            continue
        source_groups[source] = {
            "count": len(group),
            "answer_accuracy": _mean(sample["answer_correct"] for sample in group),
            "status_accuracy": _mean(sample["status_correct"] for sample in group),
            "joint_accuracy": _mean(sample["joint_correct"] for sample in group),
            "source_accuracy": _mean(
                float(source in sample["pred_sources"]) for sample in group
            ),
            "source_accuracy_probed_only": _mean(
                float(source in sample["pred_sources"]) for sample in asked
            ),
            "source_probed": len(asked),
        }
    low_uncertain = cell_grid.get("1-3", {}).get("uncertain", {}).get("joint_accuracy")
    high_uncertain = cell_grid.get("7-10", {}).get("uncertain", {}).get("joint_accuracy")
    low_known_joint = cell_grid.get("1-3", {}).get("known", {}).get("joint_accuracy")
    high_known_joint = cell_grid.get("7-10", {}).get("known", {}).get("joint_accuracy")
    high_risk_values = [
        value
        for value in (high_known_joint, high_uncertain)
        if value is not None
    ]
    evidence_age_groups = {}
    for label, lower, upper in (
        ("0-2s", 0.0, 2.0),
        (">2-5s", 2.0, 5.0),
        (">5s", 5.0, None),
    ):
        group = [
            sample
            for sample in samples
            if sample["evidence_age"] >= lower
            and (upper is None or sample["evidence_age"] <= upper)
            and not (lower > 0.0 and sample["evidence_age"] == lower)
        ]
        group_cells = []
        for bucket in buckets:
            for status in EPISTEMIC_STATUSES:
                cell_group = [
                    sample
                    for sample in group
                    if sample["volatility_bucket"] == bucket and sample["status"] == status
                ]
                if cell_group:
                    group_cells.append(_mean(sample["joint_correct"] for sample in cell_group))
        evidence_age_groups[label] = {
            "count": len(group),
            "answer_accuracy": _mean(sample["answer_correct"] for sample in group),
            "joint_accuracy": _mean(sample["joint_correct"] for sample in group),
            "storm_br": (
                _harmonic_mean(group_cells) if len(group_cells) == required_cell_count else None
            ),
            "storm_br_observed": _harmonic_mean(group_cells),
            "observed_cell_count": len(group_cells),
        }
    return {
        "count": len(samples),
        "answer_accuracy": _mean(sample["answer_correct"] for sample in samples),
        "status_accuracy": _mean(sample["status_correct"] for sample in samples),
        "status_balanced_accuracy": _mean(
            group["status_accuracy"]
            for group in status_groups.values()
            if group["status_accuracy"] is not None
        ),
        "storm_score": storm_score,
        "storm_score_name": "STORM-BR",
        "storm_score_arithmetic": storm_score_arithmetic,
        "storm_score_observed": storm_score_observed,
        "storm_score_attribution": storm_score_attr,
        "storm_score_attribution_observed": storm_score_attr_observed,
        "score_cells_complete": cells_complete,
        "required_cell_count": required_cell_count,
        "observed_cell_count": len(cell_values),
        "volatility_score": volatility,
        "known_volatility_score": known_volatility_score,
        "epistemic_uncertainty_score": uncertainty,
        "worst_observed_cell_joint": min(cell_values) if cell_values else None,
        "known_only_accuracy_drop": known_only_accuracy_drop,
        "known_joint_volatility_drop": (
            None
            if low_known_joint is None or high_known_joint is None
            else low_known_joint - high_known_joint
        ),
        "uncertain_joint_volatility_drop": (
            None
            if low_uncertain is None or high_uncertain is None
            else low_uncertain - high_uncertain
        ),
        "high_risk_reliability": (
            _harmonic_mean(high_risk_values) if len(high_risk_values) == 2 else None
        ),
        "overconfidence_rate": overconfidence_rate,
        "underconfidence_rate": underconfidence_rate,
        "uncertainty_source_accuracy": uncertainty_source_accuracy,
        "uncertainty_source_accuracy_scope": "legacy_probed_only",
        "uncertainty_source_probed": len(source_scored),
        "uncertainty_source_coverage": (
            len(source_scored) / len(uncertain_samples) if uncertain_samples else None
        ),
        "uncertainty_cause_sample_f1": uncertainty_cause_sample_f1,
        "uncertainty_cause_macro_f1": uncertainty_cause_macro_f1,
        "uncertainty_cause_micro_f1": uncertainty_cause_micro_f1,
        "by_uncertainty_cause_label": cause_label_scores,
        "time_aware_answer_accuracy": _weighted_mean(
            [sample["answer_correct"] for sample in samples],
            [sample["time_weight"] for sample in samples],
        ),
        "difficulty_aware_joint_accuracy": _weighted_mean(
            [sample["joint_correct"] for sample in samples],
            [sample["difficulty_weight"] for sample in samples],
        ),
        "invalid_answer_predictions": sum(sample["answer_invalid"] for sample in samples),
        "invalid_status_predictions": sum(sample["status_invalid"] for sample in samples),
        "volatility_axis": "change_intensity",
        "by_volatility": volatility_groups,
        "by_change_intensity": volatility_groups,
        "by_epistemic_status": status_groups,
        "by_known_only_change_intensity": known_only_answer,
        "by_overconfidence_change_intensity": overconfidence_by_bucket,
        "by_underconfidence_change_intensity": underconfidence_by_bucket,
        "by_uncertainty_source": source_groups,
        "by_evidence_age": evidence_age_groups,
        "by_observed_cell": cell_groups,
        "by_cell": cell_grid,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    statuses: Sequence[str],
    replicates: int,
    seed: int,
) -> dict:
    if replicates <= 0:
        return {}
    prediction_by_id = {str(row["id"]): row for row in predictions}
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        clusters[str(row.get("episode_id", row["id"]))].append(row)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    tracked = (
        "answer_accuracy",
        "storm_score",
        "storm_score_attribution",
        "volatility_score",
        "known_volatility_score",
        "epistemic_uncertainty_score",
        "worst_observed_cell_joint",
        "known_only_accuracy_drop",
        "known_joint_volatility_drop",
        "uncertain_joint_volatility_drop",
        "high_risk_reliability",
        "overconfidence_rate",
        "underconfidence_rate",
        "uncertainty_source_accuracy",
        "uncertainty_cause_sample_f1",
        "uncertainty_cause_macro_f1",
        "uncertainty_cause_micro_f1",
    )
    values: dict[str, list[float]] = defaultdict(list)
    for replicate in range(replicates):
        sampled_rows = []
        sampled_predictions = []
        for draw, cluster_id in enumerate(rng.choices(cluster_ids, k=len(cluster_ids))):
            for row in clusters[cluster_id]:
                copied_row = dict(row)
                copied_prediction = dict(prediction_by_id[str(row["id"])])
                synthetic_id = f"{replicate}:{draw}:{row['id']}"
                copied_row["id"] = synthetic_id
                copied_prediction["id"] = synthetic_id
                sampled_rows.append(copied_row)
                sampled_predictions.append(copied_prediction)
        score = _score_core(sampled_rows, sampled_predictions, statuses)
        for metric in tracked:
            if score[metric] is not None:
                values[metric].append(float(score[metric]))
    return {
        metric: {
            "low": _percentile(metric_values, 0.025),
            "high": _percentile(metric_values, 0.975),
            "valid_replicates": len(metric_values),
        }
        for metric, metric_values in values.items()
        if metric_values
    }


def score_predictions(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260817,
) -> dict:
    contract = audit_ground_truth(rows)
    statuses = tuple(contract["status_labels"])
    result = _score_core(rows, predictions, statuses)
    result.update(
        {
            "schema_version": "storm_balanced_reliability_v2",
            "dataset_contract": contract,
            "headline_definition": (
                "harmonic mean of all six 3x2 cell joint accuracies; "
                f"cells = change_intensity {list(CHANGE_INTENSITY_BUCKETS)} x known/uncertain; "
                "known cell = answer correct AND status=known; "
                "uncertain cell = answer correct AND status=uncertain; "
                "all six cells are required and any zero-valued cell makes STORM-BR zero"
            ),
            "attribution_definition": (
                "STORM-BR-Attr replaces each uncertain sample's joint correctness with "
                "answer correct x status correct x multi-label uncertainty-cause set F1; "
                "all ground-truth uncertain samples are scored, including predicted-known cases"
            ),
            "confidence_intervals_95": _cluster_bootstrap(
                rows, predictions, statuses, bootstrap_replicates, bootstrap_seed
            ),
            "bootstrap": {
                "unit": "episode_id",
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
            },
        }
    )
    return result


def score_answer_predictions(rows: Sequence[dict], predictions: Sequence[dict]) -> dict:
    """Score answer-side diagnostics without fabricating epistemic predictions."""
    contract = audit_ground_truth(rows)
    statuses = tuple(contract["status_labels"])
    placeholders = [dict(prediction, pred_epistemic_status=None) for prediction in predictions]
    core = _score_core(rows, placeholders, statuses)
    return {
        "schema_version": "storm_balanced_reliability_v2",
        "score_scope": "answer_only",
        "count": core["count"],
        "answer_accuracy": core["answer_accuracy"],
        "volatility_score": core["volatility_score"],
        "known_volatility_score": core["known_volatility_score"],
        "known_only_accuracy_drop": core["known_only_accuracy_drop"],
        "time_aware_answer_accuracy": core["time_aware_answer_accuracy"],
        "invalid_answer_predictions": core["invalid_answer_predictions"],
        "by_change_intensity": core["by_change_intensity"],
        "by_known_only_change_intensity": core["by_known_only_change_intensity"],
        "dataset_contract": contract,
        "storm_score": None,
        "omitted_metrics": [
            "epistemic_uncertainty_score",
            "storm_score",
            "storm_score_attribution",
            "status_accuracy",
            "overconfidence_rate",
            "underconfidence_rate",
            "uncertainty_source_accuracy",
            "uncertainty_cause_sample_f1",
            "uncertainty_cause_macro_f1",
            "difficulty_aware_joint_accuracy",
        ],
    }


def accuracy_summary(predictions: Sequence[dict]) -> dict:
    from collections import Counter

    valid = [row for row in predictions if row.get("pred_index") is not None]
    correct = sum(bool(row.get("correct")) for row in predictions)
    by_type = {}
    for question_type in sorted({row.get("question_type") for row in predictions} - {None}):
        subset = [row for row in predictions if row.get("question_type") == question_type]
        type_correct = sum(bool(row.get("correct")) for row in subset)
        by_type[question_type] = {
            "correct": type_correct,
            "total": len(subset),
            "accuracy": type_correct / len(subset) if subset else 0.0,
        }
    return {
        "correct": correct,
        "total": len(predictions),
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "valid_predictions": len(valid),
        "invalid_predictions": len(predictions) - len(valid),
        "choice_distribution": dict(Counter(row.get("pred_choice") or "INVALID" for row in predictions)),
        "by_question_type": by_type,
    }


def streaming_audit(predictions: Sequence[dict], protocol: str | None = None) -> dict:
    from scripts.eval.storm_streaming import NATIVE_RECURRENT_STATE

    if not predictions or "stream_frames_seen" not in predictions[0]:
        return {}
    return {
        "episodes": len({row.get("episode_id") for row in predictions}),
        "max_stream_frames_seen": max(row["stream_frames_seen"] for row in predictions),
        "max_buffered_frames": max(row["buffered_frames"] for row in predictions),
        "memory_pressure_predictions": sum(
            row.get("buffer_capacity") is not None and row["stream_frames_seen"] > row["buffer_capacity"]
            for row in predictions
        ),
        "future_leakage_violations": sum(bool(row.get("future_leakage")) for row in predictions),
        "native_recurrent_state": NATIVE_RECURRENT_STATE,
        "visual_state_persistent": (
            protocol == "online"
            if protocol is not None
            else any(bool(row.get("visual_state_persistent")) for row in predictions)
        ),
    }


def attach_storm_scores(
    predictions: Sequence[dict],
    gt_rows: Sequence[dict],
    signature_data: dict | None = None,
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260817,
) -> dict:
    score = score_predictions(
        gt_rows,
        predictions,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    if signature_data is not None:
        score["run"] = signature_data
        for key in (
            "dataset_sha256",
            "dataset_ids_sha256",
            "dataset_count",
            "dataset_episode_count",
        ):
            if key in signature_data:
                score[key] = signature_data[key]
    audit = streaming_audit(predictions, (signature_data or {}).get("protocol"))
    if audit:
        score["streaming_audit"] = audit
    score.update(accuracy_summary(predictions))
    return score
