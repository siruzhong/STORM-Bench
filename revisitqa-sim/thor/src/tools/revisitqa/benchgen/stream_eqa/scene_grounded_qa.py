"""Build v2.3 experiment 2 with room-based current-state and factual questions.

The other four question types use the v2.2 event builders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import visual_contrast_qa_exp2 as exp2


base = exp2.base

VERSION = "scene_grounded_v2_3_exp2"
ID_VERSION_TAG = "v23e2"
QA_SOURCE = "simulation_ground_truth_scene_grounded_v2_3_exp2"
REPORT_FILENAME = "QA_v2.3_experiment_2_scene_grounding_report.md"
DEFAULT_SEED = 20260825

ROOM_LABELS = {
    "kitchen": "kitchen",
    "living_room": "living room",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
}
ROOM_OPTIONS = list(ROOM_LABELS.values())

SUBTYPES = dict(exp2.SUBTYPES)
SUBTYPES.update({
    "current_state": "visible_room_at_query_v23e2",
    "factual_retrieval": "activity_room_retrieval_v23e2",
})


def _room_answer(scene: str) -> str:
    room = base._room_for_scene(scene)
    try:
        return ROOM_LABELS[room]
    except KeyError as error:
        raise base.ExportError(f"unsupported AI2-THOR room type: {room}") from error


def _scene_question(
    *,
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    maximum: int,
    refusal_option: str,
    question_type: str,
    question: str,
    semantic_prefix: str,
) -> dict[str, Any]:
    query = float(maximum)
    if query <= 0:
        raise base.ExportError("scene question needs a positive final query time")
    correct = _room_answer(scene)
    return base._draft(
        episode_id=episode_id,
        question_type=question_type,
        query_time=query,
        question=question.format(query=query),
        concrete_options=list(ROOM_OPTIONS),
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[0.0, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"{semantic_prefix}:{correct}",
    )


def _build_current_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Identify the room visible at the final current state."""
    del ordinal
    return _scene_question(
        episode_id=episode_id,
        scene=scene,
        events=events,
        maximum=maximum,
        refusal_option=refusal_option,
        question_type="current_state",
        question="At {query:.1f}s, what type of room is the agent currently observing?",
        semantic_prefix="visible_room_at_query",
    )


def _build_factual_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Retrieve the persistent room setting from the video."""
    del ordinal
    return _scene_question(
        episode_id=episode_id,
        scene=scene,
        events=events,
        maximum=maximum,
        refusal_option=refusal_option,
        question_type="factual_retrieval",
        question="Which type of room is the setting of the activity shown through {query:.1f}s?",
        semantic_prefix="activity_room",
    )


_EXP2_REPORT = exp2._report_markdown


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _EXP2_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.2 Experiment 2", "Sim QA v2.3 Experiment 2")
    return text + "\n".join([
        "",
        "## 9. v2.3 Experiment 2: Scene Visual Grounding",
        "",
        "- v2.3 Experiment 1 shows that Qwen3.5-9B cannot reliably track the return order of small objects across a full video even when offline and online evidence are aligned.",
        "- This experiment returns to the v2.2 baseline and rewrites only 53 Known `current_state` and 52 Known `factual_retrieval` items.",
        "- The two types ask for the room visible at the query time and the room containing the complete activity, respectively.",
        "- Options are fixed to kitchen, living room, bedroom, and bathroom; each scene type has 75 items and answer positions remain globally balanced.",
        "- Room type is a large-scale visual fact that persists through the video, does not depend on small-object timing, and cannot be inferred text-only from a question without the scene name.",
        "- `history_aggregation`, `object_tracking`, `state_change`, and `temporal_reasoning` remain unchanged from v2.2 Experiment 2 to preserve event-level reasoning difficulty.",
        "",
    ])


def _configure_base() -> None:
    exp2._configure_base()
    base.VERSION = VERSION
    base.ID_VERSION_TAG = ID_VERSION_TAG
    base.QA_SOURCE = QA_SOURCE
    base.REPORT_FILENAME = REPORT_FILENAME
    base.SUBTYPES = dict(SUBTYPES)
    base._build_current_known = _build_current_known
    base._build_factual_known = _build_factual_known
    base._report_markdown = _report_markdown


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_base()
    summary = (
        base.validate_export(args.output)
        if args.validate_only
        else base.export_clone(
            args.source_dataset,
            args.output,
            link_mode=args.link_mode,
            seed=args.seed,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
