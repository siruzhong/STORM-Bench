"""Strict JSONL loading and validation for the public benchmark schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

STATUS = {"known", "uncertain"}
QUESTION_TYPES = {"current_state", "factual_retrieval", "history_aggregation", "state_change", "object_tracking", "temporal_reasoning"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: expected a JSON object")
                rows.append(value)
    return rows


def _require(row: dict[str, Any], key: str, path: str) -> Any:
    if key not in row:
        raise ValueError(f"{path}: missing required field {key!r}")
    return row[key]


def validate_questions(rows: Iterable[dict[str, Any]], source: str = "questions") -> list[dict[str, Any]]:
    rows = list(rows)
    ids = set()
    for n, row in enumerate(rows, 1):
        label = f"{source}:{n}"
        item_id = _require(row, "id", label)
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise ValueError(f"{label}: id must be a unique non-empty string")
        ids.add(item_id)
        query_time = _require(row, "query_time", label)
        if not isinstance(query_time, (int, float)) or query_time < 0:
            raise ValueError(f"{label}: query_time must be non-negative")
        options = _require(row, "options", label)
        if not isinstance(options, list) or len(options) != 4 or len(set(options)) != 4:
            raise ValueError(f"{label}: options must contain four distinct strings")
        answer = _require(row, "answer_index", label)
        if not isinstance(answer, int) or not 0 <= answer < 4:
            raise ValueError(f"{label}: answer_index must be in [0, 3]")
        diagnostics = _require(row, "diagnostics", label)
        status = diagnostics.get("epistemic_status") if isinstance(diagnostics, dict) else None
        if status not in STATUS:
            raise ValueError(f"{label}: diagnostics.epistemic_status must be known or uncertain")
        sources = diagnostics.get("uncertainty_sources", [])
        if not isinstance(sources, list) or (status == "uncertain" and not sources):
            raise ValueError(f"{label}: uncertainty_sources must be a list; uncertain items need one")
        intensity = _require(row, "change_intensity", label)
        if not isinstance(intensity, int) or not 1 <= intensity <= 10:
            raise ValueError(f"{label}: change_intensity must be an integer in [1, 10]")
        if row.get("question_type") not in QUESTION_TYPES:
            raise ValueError(f"{label}: unsupported question_type")
        for start, end in row.get("evidence_spans", []):
            if start < 0 or start > end or end > query_time:
                raise ValueError(f"{label}: evidence span is outside the observed prefix")
        if status == "uncertain" and answer != 3:
            raise ValueError(f"{label}: uncertain items must use option index 3 for abstention")
    return rows


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    return validate_questions(read_jsonl(path), str(path))
