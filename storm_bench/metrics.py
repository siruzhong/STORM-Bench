"""Deterministic STORM-Bench metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _status(value: str) -> str:
    value = value.strip().lower()
    aliases = {"k": "known", "u": "uncertain", "cannot_be_determined": "uncertain"}
    value = aliases.get(value, value)
    if value not in {"known", "uncertain"}:
        raise ValueError(f"invalid status: {value}")
    return value


def _index(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().upper()
        if len(value) == 1 and value in "ABCD":
            return ord(value) - ord("A")
        value = int(value)
    if not isinstance(value, int) or not 0 <= value < 4:
        raise ValueError(f"invalid task answer: {value}")
    return value


def _f1(predicted: set[str], target: set[str]) -> float:
    if not predicted and not target:
        return 1.0
    if not predicted or not target:
        return 0.0
    precision = len(predicted & target) / len(predicted)
    recall = len(predicted & target) / len(target)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def storm_br(records: Iterable[dict[str, Any]], smooth: float = 1.0) -> float:
    """Laplace-smoothed harmonic mean over occupied intensity/status cells."""
    cells: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in records:
        cell = (min((int(row["change_intensity"]) - 1) // 3, 2), _status(row["status"]))
        cells[cell][1] += 1
        cells[cell][0] += int(row["joint_correct"])
    scores = [(correct + smooth) / (count + 2 * smooth) for correct, count in cells.values()]
    if not scores:
        return 0.0
    return len(scores) / sum(1.0 / score for score in scores)


def evaluate(questions: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    questions = list(questions)
    by_id = {row["id"]: row for row in questions}
    pred_by_id = {row["id"]: row for row in predictions}
    if set(by_id) != set(pred_by_id):
        raise ValueError("question and prediction ids must match exactly")
    records = []
    for item_id, question in by_id.items():
        pred = pred_by_id[item_id]
        answer = _index(pred.get("task_answer", pred.get("answer_index")))
        status = _status(pred["status"])
        target_status = question["diagnostics"]["epistemic_status"]
        correct_answer = answer == question["answer_index"]
        joint = correct_answer and status == target_status
        target_sources = set(question["diagnostics"].get("uncertainty_sources", []))
        predicted_sources = set(pred.get("uncertainty_sources", []))
        records.append({"id": item_id, "change_intensity": question["change_intensity"], "status": target_status, "answer_correct": int(correct_answer), "status_correct": int(status == target_status), "joint_correct": int(joint), "cause_f1": _f1(predicted_sources, target_sources) if target_status == "uncertain" else None})
    n = len(records)
    known = [r for r in records if r["status"] == "known"]
    uncertain = [r for r in records if r["status"] == "uncertain"]
    def mean(rows: list[dict[str, Any]], key: str) -> float | None:
        return sum(row[key] for row in rows) / len(rows) if rows else None
    return {"n": n, "accuracy": mean(records, "answer_correct"), "joint_accuracy": mean(records, "joint_correct"), "known_joint_accuracy": mean(known, "joint_correct"), "uncertain_joint_accuracy": mean(uncertain, "joint_correct"), "overconfidence": sum(r["status"] == "known" and not r["status_correct"] for r in uncertain) / len(uncertain) if uncertain else None, "storm_br": storm_br(records), "storm_br_percent": 100 * storm_br(records), "records": records}
