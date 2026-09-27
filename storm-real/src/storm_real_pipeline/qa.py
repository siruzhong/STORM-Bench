"""Construct temporally grounded QA directly from an assembled episode.

The revisit plan is used only for the cumulative visit count. It is never
included in the model's visual-evidence request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .client import VLMClient, load_prompt
from .domains import get_domain
from .io_utils import write_json

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
QUESTION_TYPES = frozenset({
    "factual_retrieval", "current_state", "state_change", "object_tracking",
    "temporal_reasoning", "history_aggregation",
})
EPISTEMIC_STATUS = frozenset({"known", "uncertain"})
UNCERTAINTY_SOURCES = frozenset({
    "missing_observation", "partial_observation", "low_visual_quality",
    "ambiguous_evidence", "ambiguous_attribute", "multiple_candidates",
})
MIN_KNOWN_QUESTIONS = 8
MIN_UNCERTAIN_QUESTIONS = 2
QUESTION_GENERATION_ATTEMPTS = 5
UNKNOWN_ANSWER_MARKERS = (
    "cannot be determined", "cannot determine", "unable to determine",
    "cannot be established", "cannot be resolved", "cannot be inferred",
    "cannot be confirmed", "cannot be identified", "cannot be recovered",
    "insufficient evidence", "not enough evidence", "insufficient information",
    "not enough information", "not shown in the video", "not visible",
    "not observed", "not recorded", "does not establish", "does not reveal",
    "does not provide enough", "does not support a definite",
    "remains unresolved", "remains ambiguous", "lacks enough",
)
_TIMESTAMP_TOLERANCE = 0.05
_CANONICALIZING_DOMAINS = frozenset({"health", "music", "sports"})
_EXACT_UNCERTAIN_ANSWER = "Cannot be determined"
_UNCERTAINTY_SOURCE_ALIASES = {
    "occlusion": "partial_observation",
    "out_of_view": "missing_observation",
    "identity_ambiguity": "ambiguous_evidence",
    "conflicting_evidence": "ambiguous_evidence",
    "insufficient_temporal_resolution": "ambiguous_evidence",
    "attribute_ambiguity": "ambiguous_attribute",
}
_UNRESOLVED_ANSWER_MARKERS = (
    "cannot be determined", "can not be determined", "cannot determine",
    "not enough information", "insufficient information", "not visible", "unknown", "uncertain",
)


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _check_parameters(sample_fps: float, jpeg_quality: int, questions_per_type: int) -> None:
    if not _finite_number(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be a finite positive number")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer from 1 to 100")
    if type(questions_per_type) is not int or questions_per_type != 2:
        raise ValueError("questions_per_type must be 2 to satisfy the 8 Known / 2 Uncertain quotas")


def video_metadata(video_path: Path) -> tuple[float, float, int]:
    """Return duration, source FPS, and frame count; reject unusable metadata."""
    if cv2 is None:
        raise RuntimeError("Video sampling requires opencv-python-headless")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError("Cannot open the episode video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError("The episode video has invalid FPS or frame-count metadata")
        frame_count = int(count)
        return frame_count / fps, fps, frame_count
    finally:
        capture.release()


def sample_frames(
    video_path: Path,
    sample_fps: float = 1.0,
    jpeg_quality: int = 80,
    max_duration: float = 0.0,
) -> tuple[list[tuple[float, str]], float]:
    """Sample timestamped JPEG frames without segmentation-stage smoothing.

    ``max_duration=0`` samples the complete episode, as the public entry point
    always does. The optional bound is retained for callers of this helper.
    """
    _check_parameters(sample_fps, jpeg_quality, 2)
    if not _finite_number(max_duration) or max_duration < 0:
        raise ValueError("max_duration must be finite and nonnegative")
    duration, source_fps, frame_count = video_metadata(video_path)
    effective_duration = duration if max_duration == 0 else min(duration, max_duration)
    max_frame = min(frame_count, int(math.ceil(effective_duration * source_fps)))
    step = max(1, round(source_fps / sample_fps))
    capture = cv2.VideoCapture(str(video_path))
    frames: list[tuple[float, str]] = []
    try:
        if not capture.isOpened():
            raise OSError("Cannot open the episode video for sampling")
        for frame_index in range(0, max_frame, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode the requested frame at index {frame_index}")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise RuntimeError(f"Cannot encode the requested frame at index {frame_index}")
            timestamp = round(frame_index / source_fps, 2)
            if frames and timestamp <= frames[-1][0]:
                raise ValueError("sample_fps is too high for two-decimal timestamp precision")
            frames.append((timestamp, base64.b64encode(encoded).decode("ascii")))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("No frames were sampled from the episode video")
    return frames, effective_duration


def vision_content(frames: list[tuple[float, str]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for timestamp, image_b64 in frames:
        content.append({"type": "text", "text": f"[t={timestamp:.2f}s]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })
    return content


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def parse_json_response(raw: str, expected_type: type) -> Any:
    """Accept a JSON document or a JSON code fence, without surrounding prose."""
    if not isinstance(raw, str):
        raise TypeError("The model response must be text")
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(text, parse_constant=reject_constant)
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected {expected_type.__name__}, received {type(value).__name__}")
    return value


def validate_question(
    question: Any,
    episode_id: str,
    duration: float,
    sample_timestamps: list[float],
    *,
    domain: str = "cook",
) -> dict[str, Any] | None:
    """Return a canonical question, or ``None`` for invalid evidence/schema.

    Timestamp tolerance only corrects numeric representation around a supplied
    sampled frame. Every returned evidence endpoint is an actual frame time,
    and no endpoint may exceed the canonical query timestamp.
    """
    spec = get_domain(domain)
    if not isinstance(question, dict) or not _finite_number(duration) or duration <= 0:
        return None
    if "domain" in question and question["domain"] != spec.name:
        return None
    required = {
        "query_time", "question_type", "question_subtype", "video_evidence",
        "question", "options", "answer_index", "evidence_spans", "diagnostics",
        "diagnostic_rationale",
    }
    if not required.issubset(question):
        return None
    question_type = question["question_type"]
    if not isinstance(question_type, str) or question_type not in QUESTION_TYPES:
        return None
    for key in ("question_subtype", "video_evidence", "question"):
        if not _nonempty_text(question[key]):
            return None
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", question["question_subtype"]) is None:
        return None
    options = question["options"]
    if not isinstance(options, list) or len(options) != 4 or not all(map(_nonempty_text, options)):
        return None
    if len({_normalized_question(option) for option in options}) != 4:
        return None
    options = [option.strip() for option in options]
    answer_index = question["answer_index"]
    if type(answer_index) is not int or not 0 <= answer_index < 4:
        return None
    if (
        not isinstance(sample_timestamps, list) or not sample_timestamps
        or any(not _finite_number(t) or not 0 <= t <= duration for t in sample_timestamps)
        or any(a >= b for a, b in zip(sample_timestamps, sample_timestamps[1:]))
    ):
        return None

    def align(value: Any) -> float | None:
        if not _finite_number(value) or value < 0 or value > duration + _TIMESTAMP_TOLERANCE:
            return None
        nearest = min(sample_timestamps, key=lambda timestamp: abs(timestamp - value))
        if abs(nearest - value) > _TIMESTAMP_TOLERANCE + 1e-9:
            return None
        return float(nearest)

    query_time = align(question["query_time"])
    if query_time is None:
        return None
    raw_spans = question["evidence_spans"]
    if not isinstance(raw_spans, list) or not raw_spans:
        return None
    spans: list[list[float]] = []
    for span in raw_spans:
        if not isinstance(span, list) or len(span) != 2:
            return None
        start, end = align(span[0]), align(span[1])
        if start is None or end is None or span[0] > span[1] or start > end or end > query_time:
            return None
        spans.append([start, end])

    diagnostics = question["diagnostics"]
    if not isinstance(diagnostics, dict):
        return None
    status = diagnostics.get("epistemic_status")
    if not isinstance(status, str) or status not in EPISTEMIC_STATUS:
        return None
    sources = diagnostics.get("uncertainty_sources")
    if spec.name in _CANONICALIZING_DOMAINS:
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(sources, list):
            sources = [
                _UNCERTAINTY_SOURCE_ALIASES.get(source.strip(), source.strip())
                if isinstance(source, str) else source for source in sources
            ]
    if not isinstance(sources, list) or any(
        not isinstance(source, str) or source not in UNCERTAINTY_SOURCES for source in sources
    ):
        return None
    if status == "known" and sources:
        return None
    if status == "uncertain":
        if not sources:
            return None
        if spec.exact_uncertain_answer:
            if options[answer_index] != _EXACT_UNCERTAIN_ANSWER:
                if spec.name not in _CANONICALIZING_DOMAINS:
                    return None
                normalized_answer = re.sub(r"[^a-z]+", " ", options[answer_index].casefold()).strip()
                if not any(marker in normalized_answer for marker in _UNRESOLVED_ANSWER_MARKERS):
                    return None
                options[answer_index] = _EXACT_UNCERTAIN_ANSWER
            # Canonicalization must not turn distinct candidates into duplicate options.
            if len({_normalized_question(option) for option in options}) != 4:
                return None
        elif not any(marker in options[answer_index].casefold() for marker in UNKNOWN_ANSWER_MARKERS):
            return None
    rationale = question["diagnostic_rationale"]
    if not isinstance(rationale, dict) or any(
        not _nonempty_text(rationale.get(key)) for key in ("volatility", "uncertainty")
    ):
        return None
    return {
        "id": "",
        "episode_id": episode_id,
        "domain": spec.name,
        "query_time": query_time,
        "question_type": question_type,
        "question_subtype": question["question_subtype"].strip(),
        "video_evidence": question["video_evidence"].strip(),
        "question": question["question"].strip(),
        "options": [option.strip() for option in options],
        "answer_index": answer_index,
        "evidence_spans": spans,
        "diagnostics": {
            "epistemic_status": status,
            "uncertainty_sources": list(dict.fromkeys(sources)),
        },
        "diagnostic_rationale": {key: rationale[key].strip() for key in ("volatility", "uncertainty")},
    }


def generate_questions(
    client: VLMClient,
    episode_id: str,
    duration: float,
    frames: list[tuple[float, str]],
    questions_per_type: int = 2,
    disable_thinking: bool = False,
    *,
    thinking: str | None = None,
    domain: str = "cook",
) -> list[dict[str, Any]]:
    """Generate candidates using the domain's prompt and quota-filling budget."""
    spec = get_domain(domain)
    _check_parameters(1.0, 80, questions_per_type)
    if not _finite_number(duration) or duration <= 0 or not frames:
        raise ValueError("Question generation requires a positive duration and sampled frames")
    if any(not _nonempty_text(encoded) for _, encoded in frames):
        raise ValueError("Every sampled frame must contain a JPEG payload")
    sample_timestamps = [timestamp for timestamp, _ in frames]
    if (
        any(not _finite_number(t) or not 0 <= t <= duration for t in sample_timestamps)
        or any(a >= b for a, b in zip(sample_timestamps, sample_timestamps[1:]))
    ):
        raise ValueError("Frame timestamps must be finite, ordered, unique, and within the episode")
    accepted: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_questions: set[str] = set()
    max_total = len(QUESTION_TYPES) * questions_per_type
    prompt = load_prompt("qa.txt", spec.name)
    generation_attempts = spec.qa_generation_attempts

    def status_counts() -> tuple[int, int]:
        known = sum(q["diagnostics"]["epistemic_status"] == "known" for q in accepted)
        return known, len(accepted) - known

    for attempt in range(1, generation_attempts + 1):
        known_count, uncertain_count = status_counts()
        remaining_known = max(0, MIN_KNOWN_QUESTIONS - known_count)
        remaining_uncertain = max(0, MIN_UNCERTAIN_QUESTIONS - uncertain_count)
        if remaining_known == remaining_uncertain == 0:
            break
        if remaining_known == 0:
            generation_mode = "uncertain_only"
            if spec.name in _CANONICALIZING_DOMAINS:
                available_types = [t for t in sorted(QUESTION_TYPES) if counts[t] < questions_per_type]
                instruction = (
                    f"Generate {max(4, remaining_uncertain * 2)} candidate questions, ALL uncertain, "
                    f"so validation can retain the needed {remaining_uncertain}. "
                    f"Use ONLY these question types with remaining slots: {available_types}. "
                    f"Make the correct option exactly '{_EXACT_UNCERTAIN_ANSWER}', include a nonempty "
                    "list of legal uncertainty_sources, and align query_time and evidence endpoints "
                    "to supplied timestamps, with evidence ending no later than query_time. "
                    "Do not return any known question."
                )
            elif spec.exact_uncertain_answer:
                instruction = (
                    f"Generate ONLY {remaining_uncertain} new uncertain questions. "
                    f"Make the correct option exactly '{_EXACT_UNCERTAIN_ANSWER}', "
                    "include legal uncertainty_sources, use types with remaining slots, "
                    "and do not return known questions."
                )
            else:
                instruction = (
                    f"Generate ONLY {remaining_uncertain} new uncertain questions. "
                    "The correct option must state, in concise question-specific wording, "
                    "that the requested fact cannot be established from available visual evidence. "
                    "Include legal uncertainty_sources, vary the wording, use types with remaining slots, "
                    "and do not return known questions."
                )
        elif remaining_uncertain == 0:
            generation_mode = "known_only"
            instruction = (
                f"Generate ONLY {remaining_known} new known questions whose answers are "
                "uniquely supported by visible frames. Use types with remaining slots."
            )
        else:
            generation_mode = "fill_both_quotas"
            instruction = (
                f"Generate at least {remaining_known} new known and {remaining_uncertain} new uncertain "
                "questions. While the uncertain quota remains unmet, include at least one valid uncertain "
                "question. "
            )
            instruction += (
                f"Its correct option must be exactly '{_EXACT_UNCERTAIN_ANSWER}'."
                if spec.exact_uncertain_answer else
                "Its correct option must explicitly state why available visual evidence cannot "
                "establish the fact; use varied question-specific wording."
            )
        payload = {
            "episode_id": episode_id,
            "domain": spec.name,
            "episode_duration": round(duration, 2),
            "max_questions_per_type": questions_per_type,
            "required_total_known": MIN_KNOWN_QUESTIONS,
            "required_total_uncertain": MIN_UNCERTAIN_QUESTIONS,
            "generation_mode": generation_mode,
            "remaining_known_needed": remaining_known,
            "remaining_uncertain_needed": remaining_uncertain,
            "remaining_slots_by_type": {
                question_type: questions_per_type - counts[question_type]
                for question_type in sorted(QUESTION_TYPES)
            },
            "already_accepted": [{
                "question_type": q["question_type"],
                "epistemic_status": q["diagnostics"]["epistemic_status"],
                "question": q["question"],
            } for q in accepted],
            "instruction": (
                "The timestamped images following this metadata are the complete visible evidence. "
                "Use only timestamps explicitly shown with those images. Do not repeat accepted "
                "questions or fabricate evidence to satisfy quotas. " + instruction
            ),
        }
        contents = text_content(json.dumps(payload, ensure_ascii=False, indent=2))
        contents.extend(vision_content(frames))
        logger.info("QA round %s/%s: Known=%s, Uncertain=%s", attempt, generation_attempts,
                    known_count, uncertain_count)
        # Transport errors are the shared client's responsibility and must not
        # be mistaken for malformed candidate JSON or silently consume a quota round.
        raw = client.call(prompt, contents, thinking=thinking, disable_thinking=disable_thinking)
        try:
            candidates = parse_json_response(raw, list)
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid QA response; requesting another generation round: %s", exc)
            continue
        for candidate in candidates:
            question = validate_question(candidate, episode_id, duration, sample_timestamps, domain=spec.name)
            if question is None:
                logger.debug("Discarded a candidate that failed QA schema or evidence validation")
                continue
            question_type = question["question_type"]
            normalized_text = _normalized_question(question["question"])
            if counts[question_type] >= questions_per_type or normalized_text in seen_questions:
                continue
            known_count, uncertain_count = status_counts()
            status = question["diagnostics"]["epistemic_status"]
            if generation_mode == "known_only" and status != "known":
                continue
            if generation_mode == "uncertain_only" and status != "uncertain":
                continue
            reserved = (
                max(0, MIN_UNCERTAIN_QUESTIONS - uncertain_count) if status == "known"
                else max(0, MIN_KNOWN_QUESTIONS - known_count)
            )
            if len(accepted) >= max_total - reserved:
                continue
            counts[question_type] += 1
            seen_questions.add(normalized_text)
            accepted.append(question)
    known_count, uncertain_count = status_counts()
    if known_count < MIN_KNOWN_QUESTIONS or uncertain_count < MIN_UNCERTAIN_QUESTIONS:
        raise RuntimeError(
            f"QA quota not met after {generation_attempts} attempts: "
            f"Known={known_count}/{MIN_KNOWN_QUESTIONS}, Uncertain={uncertain_count}/{MIN_UNCERTAIN_QUESTIONS}"
        )
    if spec.name != "cook":
        # Preserve the original transfer-domain balancing while keeping each
        # answer tied to its option after the deterministic swap.
        target_positions = [index % 4 for index in range(len(accepted))]
        random.Random(episode_id).shuffle(target_positions)
        for question, target_index in zip(accepted, target_positions):
            current_index = question["answer_index"]
            if current_index != target_index:
                options = question["options"]
                options[current_index], options[target_index] = options[target_index], options[current_index]
                question["answer_index"] = target_index
    for index, question in enumerate(accepted, start=1):
        question["id"] = f"{episode_id}_q{index:02d}"
    return accepted


def visit_output_spans(plan: dict[str, Any]) -> list[dict[str, float | int]]:
    """Read exact rendered visit times, falling back to legacy source intervals.

    Invalid supplied plans raise ``ValueError`` instead of silently becoming a
    single-visit episode. Gaps, including black frames, are not new visits.
    """
    if not isinstance(plan, dict):
        raise ValueError("The revisit plan must be an object")
    visits = plan.get("visits")
    if not isinstance(visits, list) or not visits or any(not isinstance(v, dict) for v in visits):
        raise ValueError("The revisit plan must contain a nonempty visits array")
    spans: list[dict[str, float | int]] = []

    def append_span(start: Any, end: Any, index: int) -> None:
        if (
            not _finite_number(start) or not _finite_number(end)
            or start < 0 or end <= start
            or (spans and start < spans[-1]["out_end_sec"] - 1e-9)
        ):
            raise ValueError("Visit output spans must be finite, positive, ordered, and non-overlapping")
        spans.append({"visit_index": index, "out_start_sec": float(start), "out_end_sec": float(end)})

    # The release renderer records the actual output-frame timeline on visits.
    exact_keys = ("output_start_sec", "output_end_sec")
    if any(any(key in visit for key in exact_keys) for visit in visits):
        for index, visit in enumerate(visits, 1):
            append_span(visit.get(exact_keys[0]), visit.get(exact_keys[1]), index)
        return spans
    if "output_spans" in plan:
        explicit = plan["output_spans"]
        if not isinstance(explicit, list) or len(explicit) != len(visits):
            raise ValueError("plan.output_spans must match the number of visits")
        for index, span in enumerate(explicit, 1):
            if not isinstance(span, dict):
                raise ValueError("Each plan.output_spans entry must be an object")
            append_span(span.get("output_start_sec", span.get("out_start_sec")),
                        span.get("output_end_sec", span.get("out_end_sec")), index)
        return spans
    speed = plan.get("speed", 1.0)
    blank = plan.get("blank_output_sec", plan.get("blank_sec", 0.0))
    if not _finite_number(speed) or speed <= 0 or not _finite_number(blank) or blank < 0:
        raise ValueError("Revisit speed must be positive and the black-frame duration nonnegative")
    offset = 0.0
    previous_source_end = -1.0
    for index, visit in enumerate(visits, 1):
        start, end = visit.get("start_sec"), visit.get("end_sec")
        if (
            not _finite_number(start) or not _finite_number(end)
            or start < 0 or end <= start or start < previous_source_end
        ):
            raise ValueError("Legacy visit source intervals must be finite, ordered, and non-overlapping")
        output_end = offset + (end - start) / speed
        append_span(offset, output_end, index)
        previous_source_end = end
        offset = output_end + blank
    return spans


def attach_change_intensity(questions: list[dict[str, Any]], plan: dict[str, Any] | None) -> None:
    """Attach the count of visits that have begun by each query (minimum one)."""
    spans = visit_output_spans(plan) if plan is not None else []
    for question in questions:
        target_time = question.get("query_time")
        if not _finite_number(target_time) or target_time < 0:
            raise ValueError("Cannot attach change intensity to an invalid query timestamp")
        question["change_intensity"] = max(1, sum(
            span["out_start_sec"] <= target_time + 0.01 for span in spans
        ))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan(plan_path: Path | None, *, domain: str = "cook") -> tuple[dict[str, Any] | None, str | None]:
    spec = get_domain(domain)
    if plan_path is None:
        if spec.name != "cook":
            raise ValueError(f"A valid {spec.name} revisit plan is required for QA generation")
        return None, None
    try:
        data = plan_path.read_bytes()
        plan = parse_json_response(data.decode("utf-8"), dict)
        visit_output_spans(plan)
        if plan.get("domain", "cook") != spec.name:
            raise ValueError(f"The revisit plan does not belong to the {spec.name} domain")
        if spec.name != "cook" or "source_schema_version" in plan:
            if plan.get("source_schema_version") != spec.segmentation_schema:
                raise ValueError("The revisit plan has an incompatible source segmentation schema")
        if "language" in plan and plan["language"] != "en":
            raise ValueError("The revisit plan must use the English schema")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot use the supplied revisit plan: {exc}") from exc
    return plan, hashlib.sha256(data).hexdigest()


def _valid_cached_result(
    result: Any, request: dict[str, Any], episode_id: str,
    plan: dict[str, Any] | None,
    *,
    domain: str = "cook",
) -> bool:
    spec = get_domain(domain)
    if not isinstance(result, dict) or result.get("request") != request:
        return False
    if (result.get("episode_id") != episode_id or "video_path" in result
            or result.get("domain") != spec.name
            or result.get("schema_version") != spec.qa_schema
            or result.get("language") != "en"
            or result.get("qa_source") != "direct_video_frames"
            or result.get("sample_fps") != request["sample_fps"]):
        return False
    duration = result.get("duration_sec")
    timestamps = result.get("sample_timestamps")
    questions = result.get("questions")
    if not isinstance(questions, list) or not 10 <= len(questions) <= 12:
        return False
    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, question in enumerate(questions, 1):
        normalized = validate_question(question, episode_id, duration, timestamps, domain=spec.name)
        if normalized is None:
            return False
        normalized["id"] = f"{episode_id}_q{index:02d}"
        text = _normalized_question(normalized["question"])
        if text in seen:
            return False
        seen.add(text)
        canonical.append(normalized)
    attach_change_intensity(canonical, plan)
    if canonical != questions:
        return False
    if spec.name != "cook":
        target_positions = [index % 4 for index in range(len(canonical))]
        random.Random(episode_id).shuffle(target_positions)
        if [q["answer_index"] for q in canonical] != target_positions:
            return False
    counts = Counter(q["question_type"] for q in canonical)
    statuses = Counter(q["diagnostics"]["epistemic_status"] for q in canonical)
    return (
        all(count <= request["questions_per_type"] for count in counts.values())
        and statuses["known"] >= MIN_KNOWN_QUESTIONS
        and statuses["uncertain"] >= MIN_UNCERTAIN_QUESTIONS
    )


def generate_episode_qa(
    video_path: Path,
    output_path: Path,
    client: VLMClient,
    *,
    episode_id: str,
    domain: str = "cook",
    plan_path: Path | None = None,
    sample_fps: float = 1.0,
    jpeg_quality: int = 80,
    questions_per_type: int = 2,
    thinking: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate or resume one anonymous episode's directly grounded QA record.

    A cache is reused only after validating its full QA schema and quotas, the
    generation parameters, prompt/model, and SHA-256 hashes of video and plan.
    No source paths, filenames, or plan contents are copied into the release
    record or sent to the model. The episode ID and content hashes identify the
    video. A supplied plan's rendered-video hash must match the current video.
    """
    spec = get_domain(domain)
    video_path, output_path = Path(video_path), Path(output_path)
    plan_path = Path(plan_path) if plan_path is not None else None
    _check_parameters(sample_fps, jpeg_quality, questions_per_type)
    if not isinstance(episode_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", episode_id) is None:
        raise ValueError("episode_id must contain only ASCII letters, digits, underscores, and hyphens")
    if not video_path.is_file():
        raise FileNotFoundError("The episode video does not exist")
    if output_path.resolve() == video_path.resolve() or (
        plan_path is not None and output_path.resolve() == plan_path.resolve()
    ):
        raise ValueError("The QA output must not overwrite its video or revisit plan")
    plan, plan_sha = _load_plan(plan_path, domain=spec.name)
    video_sha = _sha256(video_path)
    if plan is not None:
        for field in ("output_sha256", "video_sha256"):
            if field in plan and plan[field] != video_sha:
                raise ValueError("The supplied revisit plan does not match the episode video content")
    request = {
        "cache_schema_version": 2,
        "schema_version": spec.qa_schema,
        "domain": spec.name,
        "algorithm": "direct-video-qa-v2",
        "video_sha256": video_sha,
        "plan_sha256": plan_sha,
        "prompt_sha256": hashlib.sha256(load_prompt("qa.txt", spec.name).encode("utf-8")).hexdigest(),
        "model": client.model,
        "sample_fps": sample_fps,
        "jpeg_quality": jpeg_quality,
        "questions_per_type": questions_per_type,
        "generation_attempts": spec.qa_generation_attempts,
        "thinking": thinking,
    }
    if output_path.exists() and not overwrite:
        try:
            existing = parse_json_response(output_path.read_text(encoding="utf-8"), dict)
            if _valid_cached_result(existing, request, episode_id, plan, domain=spec.name):
                logger.info("Reusing validated QA for episode %s", episode_id)
                return existing
            logger.info("Regenerating stale or invalid QA for episode %s", episode_id)
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            logger.warning("Regenerating unreadable QA for episode %s: %s", episode_id, exc)
    frames, duration = sample_frames(video_path, sample_fps, jpeg_quality)
    if plan is not None:
        spans = visit_output_spans(plan)
        if float(spans[-1]["out_end_sec"]) > duration + 0.1:
            raise ValueError("The supplied revisit plan extends beyond the episode video")
    else:
        logger.warning("Episode %s has no revisit plan; change_intensity defaults to one", episode_id)
    questions = generate_questions(
        client, episode_id, duration, frames, questions_per_type, thinking=thinking, domain=spec.name,
    )
    attach_change_intensity(questions, plan)
    result = {
        "schema_version": spec.qa_schema,
        "domain": spec.name,
        "language": "en",
        "episode_id": episode_id,
        "duration_sec": duration,
        "sample_fps": sample_fps,
        "qa_source": "direct_video_frames",
        "sample_timestamps": [timestamp for timestamp, _ in frames],
        "questions": questions,
        "request": request,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    logger.info("Saved %s validated questions for episode %s", len(questions), episode_id)
    return result
