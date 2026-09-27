"""Replay one immutable plan in AI2-THOR for validation and capture."""
from __future__ import annotations

import random
from pathlib import Path

import imageio.v2 as imageio

from ...events import Event
from ...evidence import evidence_passes, frame_evidence
from ...rollout import highlight_bboxes, highlight_instance_masks
from ..adapters.ai2thor import Ai2ThorSession
from ..artifacts.store import ArtifactStore
from ..domain.config import BenchmarkConfig
from ..domain.contracts import (
    EpisodePlan,
    EventExecution,
    ExecutionTrace,
    ValidationCheck,
    ValidationReport,
)


def _frame_index(time_s: float, fps: int, frame_count: int) -> int:
    return min(frame_count - 1, max(0, int(round(time_s * fps))))


def _visible_quality(evidence, config: BenchmarkConfig) -> bool:
    return evidence_passes(
        evidence,
        min_pixels=config.quality.min_target_pixels,
        min_dimension=config.quality.min_target_dimension,
        min_edge_margin=config.quality.min_edge_margin,
        max_center_distance=config.quality.max_center_distance,
    )


def _observed(
    event_type: str, before: dict, after: dict, config: BenchmarkConfig,
) -> bool:
    if event_type == "disappear":
        return _visible_quality(before, config) and not after.get("visible")
    if event_type == "appear":
        return not before.get("visible") and _visible_quality(after, config)
    return False


def _description(event_type: str, object_type: str, observed: bool) -> str:
    label = object_type.replace("_", " ").lower()
    if not observed:
        return f"No reliable visual change involving the {label} was observed."
    if event_type == "disappear":
        return f"The visible {label} was no longer present when the view returned."
    return f"A {label} became visible when the view returned."


class ThorReplay:
    def __init__(self, session: Ai2ThorSession):
        self.session = session

    def _events(self, plan: EpisodePlan) -> list[Event]:
        result = []
        for intent in plan.event_program.events:
            result.append(Event(intent.spec(), intent.index))
        return result

    def _reset_and_bind(
        self, plan: EpisodePlan, config: BenchmarkConfig,
    ) -> list[Event]:
        initial = self.session.reset(plan.recipe.scene, config)
        events = self._events(plan)
        for event in events:
            event.bind(initial)
        return events

    def preflight(
        self, plan: EpisodePlan, config: BenchmarkConfig,
    ) -> tuple[ValidationReport, ExecutionTrace]:
        events = self._reset_and_bind(plan, config)
        event_by_index = {item.index: item for item in events}
        opportunity_by_index = {
            item.index: item for item in plan.mobility.opportunities}
        frames = plan.trajectory.frames
        executions = []
        checks = []
        rng = random.Random(plan.recipe.seed ^ 0x93F117)
        failed_frames = []
        for intent in plan.event_program.events:
            event = event_by_index[intent.index]
            opportunity = opportunity_by_index[intent.index]
            before_index = _frame_index(
                opportunity.before_time_s, plan.recipe.fps, len(frames))
            trigger_index = _frame_index(
                opportunity.trigger_time_s, plan.recipe.fps, len(frames))
            after_index = _frame_index(
                opportunity.after_time_s, plan.recipe.fps, len(frames))
            before_render = self.session.teleport(
                frames[before_index], config.runtime.force_action)
            if not before_render.metadata.get("lastActionSuccess"):
                failed_frames.append(before_index)
            before = frame_evidence(before_render, [intent.target_id])
            hidden_render = self.session.teleport(
                frames[trigger_index], config.runtime.force_action)
            if not hidden_render.metadata.get("lastActionSuccess"):
                failed_frames.append(trigger_index)
            hidden = frame_evidence(hidden_render, [intent.target_id])
            applied = event.apply(self.session.controller, rng, strict=True)
            after_render = self.session.teleport(
                frames[after_index], config.runtime.force_action)
            if not after_render.metadata.get("lastActionSuccess"):
                failed_frames.append(after_index)
            after = frame_evidence(after_render, [intent.target_id])
            observed = _observed(intent.event_type, before, after, config)
            hidden_ok = not hidden.get("visible")
            checks.extend((
                ValidationCheck(
                    name=f"event_{intent.index:02d}_applied",
                    passed=applied, value=event.note, limit="applied"),
                ValidationCheck(
                    name=f"event_{intent.index:02d}_hidden",
                    passed=hidden_ok,
                    value=hidden.get("mask_pixels", 0), limit=0),
                ValidationCheck(
                    name=f"event_{intent.index:02d}_observed",
                    passed=observed,
                    value={"before": before, "after": after},
                    limit="rendered before/after evidence"),
            ))
            executions.append(EventExecution(
                index=intent.index,
                event_type=intent.event_type,
                target_id=intent.target_id,
                applied=applied,
                observed=observed,
                note=event.note,
                before_evidence=before,
                after_evidence=after,
            ))
        if failed_frames:
            checks.append(ValidationCheck(
                name="teleport_success", passed=False,
                value=failed_frames, limit="no failed frames"))
        accepted = bool(checks) and all(item.passed for item in checks)
        trace = ExecutionTrace(
            plan_id=plan.plan_id,
            accepted=accepted,
            events=tuple(executions),
            failed_frames=tuple(sorted(set(failed_frames))),
        )
        return ValidationReport(
            stage="rendered_preflight",
            accepted=accepted,
            checks=tuple(checks),
            failure_code=None if accepted else "rendered_preflight_failed",
        ), trace

    def capture(
        self,
        plan: EpisodePlan,
        config: BenchmarkConfig,
        output_dir: Path,
    ) -> ExecutionTrace:
        events = self._reset_and_bind(plan, config)
        event_by_index = {item.index: item for item in events}
        intent_by_index = {
            item.index: item for item in plan.event_program.events}
        opportunity_by_index = {
            item.index: item for item in plan.mobility.opportunities}
        triggers = {
            _frame_index(intent.trigger_time_s, plan.recipe.fps,
                         len(plan.trajectory.frames)): intent.index
            for intent in plan.event_program.events
        }
        evidence_frames = {}
        for opportunity in plan.mobility.opportunities:
            evidence_frames[_frame_index(
                opportunity.before_time_s, plan.recipe.fps,
                len(plan.trajectory.frames))] = (opportunity.index, "before")
            evidence_frames[_frame_index(
                opportunity.after_time_s, plan.recipe.fps,
                len(plan.trajectory.frames))] = (opportunity.index, "after")
        before_evidence = {}
        after_evidence = {}
        apply_status = {}
        apply_notes = {}
        failed_frames = []
        active_ids = set()
        last_boxes = {}
        box_until = {}
        rng = random.Random(plan.recipe.seed ^ 0x93F117)
        normal_path = output_dir / "rollout.mp4"
        highlighted_path = output_dir / "rollout_highlighted.mp4"
        normal_writer = imageio.get_writer(
            normal_path, fps=plan.recipe.fps,
            codec=config.runtime.video_codec, quality=8,
        )
        highlighted_writer = imageio.get_writer(
            highlighted_path, fps=plan.recipe.fps,
            codec=config.runtime.video_codec, quality=8,
        )
        try:
            for frame in plan.trajectory.frames:
                event_index = triggers.get(frame.index)
                if event_index is not None:
                    event = event_by_index[event_index]
                    applied = event.apply(
                        self.session.controller, rng, strict=True)
                    apply_status[event_index] = applied
                    apply_notes[event_index] = event.note
                    target_id = intent_by_index[event_index].target_id
                    active_ids.add(target_id)
                    if intent_by_index[event_index].event_type == "disappear":
                        box_until[target_id] = (
                            frame.index + int(1.4 * plan.recipe.fps))
                rendered = self.session.teleport(
                    frame, config.runtime.force_action)
                if not rendered.metadata.get("lastActionSuccess"):
                    failed_frames.append(frame.index)
                for target_id in {
                        item.target_id for item in plan.event_program.events}:
                    evidence = frame_evidence(rendered, [target_id])
                    if evidence.get("bbox"):
                        last_boxes[target_id] = evidence["bbox"]
                evidence_request = evidence_frames.get(frame.index)
                if evidence_request is not None:
                    requested_index, phase = evidence_request
                    target_id = intent_by_index[requested_index].target_id
                    evidence = frame_evidence(rendered, [target_id])
                    if phase == "before":
                        before_evidence[requested_index] = evidence
                    else:
                        after_evidence[requested_index] = evidence
                normal_writer.append_data(rendered.frame)
                highlighted = highlight_instance_masks(
                    rendered.frame, rendered, active_ids,
                    alpha=config.runtime.highlight_alpha,
                )
                missing_boxes = [
                    last_boxes[target_id]
                    for target_id in active_ids
                    if target_id not in rendered.instance_masks
                    and frame.index <= box_until.get(target_id, -1)
                    and target_id in last_boxes
                ]
                if missing_boxes:
                    highlighted = highlight_bboxes(
                        highlighted, missing_boxes, thickness=4)
                highlighted_writer.append_data(highlighted)
        finally:
            normal_writer.close()
            highlighted_writer.close()
        executions = []
        descriptions = []
        for intent in plan.event_program.events:
            before = before_evidence.get(intent.index, {})
            after = after_evidence.get(intent.index, {})
            observed = _observed(intent.event_type, before, after, config)
            applied = bool(apply_status.get(intent.index))
            description = _description(
                intent.event_type, intent.target_type, observed)
            executions.append(EventExecution(
                index=intent.index,
                event_type=intent.event_type,
                target_id=intent.target_id,
                applied=applied,
                observed=observed,
                note=apply_notes.get(intent.index, "event not triggered"),
                before_evidence=before,
                after_evidence=after,
            ))
            descriptions.append({
                "index": intent.index,
                "time_s": intent.trigger_time_s,
                "description": description,
                "observed": observed,
                "before_evidence": before,
                "after_evidence": after,
            })
        accepted = (
            not failed_frames
            and len(executions) == len(plan.event_program.events)
            and all(item.applied and item.observed for item in executions)
        )
        trace = ExecutionTrace(
            plan_id=plan.plan_id,
            accepted=accepted,
            events=tuple(executions),
            failed_frames=tuple(sorted(set(failed_frames))),
        )
        store = ArtifactStore(output_dir.parent)
        store.write_json(output_dir / "observed_events.json", descriptions)
        store.write_json(output_dir / "execution_trace.json", trace)
        return trace

    def close(self) -> None:
        self.session.close()
