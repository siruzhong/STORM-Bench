# Reference-aligned QA export

Use a frozen AI2THOR `questions.jsonl` as the reference profile. Select supported
VirtualHome questions before running a model. Do not use prediction accuracy to
choose questions, change answers, or force a target score.

The exporter needs a retimed evidence bundle, a native-frame visibility audit,
Python, NumPy, and SciPy. Run from the VirtualHome release directory:

```bash
PYTHONPATH=src python scripts/export_reference_aligned_qa.py \
  --evidence "$EVIDENCE_BUNDLE" \
  --visibility "$VISIBILITY_AUDIT" \
  --reference "$REFERENCE_QUESTIONS" \
  --work "$WORK_DIR" \
  --output "$DATASET_ROOT"
python scripts/materialize_aligned_videos.py "$DATASET_ROOT" \
  --source "$ORIGINAL_EPISODE_VIDEOS"
```

The source video directory contains `<episode_id>.mp4`. The exporter does not
render, re-encode, or modify videos. The materializer verifies original SHA-256
values after linking or copying each video.

## Selection rules

- Ten questions per episode; type and type/status totals are apportioned from
  the reference. Each episode has one or two uncertain questions.
- Intensity and normalized query-time marginals are soft objectives. They choose
  among supported questions and do not change observed values. A feasible solver
  result is accepted at the time limit; optimality is not assumed.
- Known object-visibility facts use the actual 1 FPS evaluator clock. A stable
  visible or absent state requires two consecutive samples. Visibility needs
  at least 128 target-mask pixels; absence needs zero. Intermediate counts are
  unclassified. Pixel visibility alone does not certify readable appliance power
  or door state, so those physical states are not used as known answers.
- A fully hidden operation requires zero target pixels throughout its native
  frame interval, including frames skipped by the evaluator.
- `change_intensity` is `max(1, count(event.start <= query_time + 1e-6))`, matching
  the AI2THOR generator's count of injected events that have started. It does not
  count observer-induced visibility transitions.
- Uncertainty-source labels describe the actual evidence limitations. Unsupported
  source combinations are reported as distribution gaps rather than assigned
  solely to satisfy a quota.
- The default `--uncertainty-policy reference` follows the supplied AI2THOR
  reference: known questions have four concrete alternatives; uncertain questions
  have three concrete alternatives and an uncertainty alternative. A negative
  observation such as "None of these objects returned to view" is a determinate
  alternative, not an uncertainty label. Grounded answers and status labels are
  not changed when the option policy changes.
- `--uncertainty-policy all_questions` reproduces the earlier candidate, which
  included an uncertainty alternative in every question. Preserve evaluated
  exports and write a new dataset when changing this policy. Reference matching
  retains the reference's option-presence cue and is not evidence of no leakage.

## Export contract

```text
dataset/
  meta_data/qa_results/Scene<scene>_Room<room>/<episode>.json
  meta_data/gene_videos/Scene<scene>_Room<room>/<episode>.mp4
  questions.jsonl
  model_inputs.jsonl
  evaluation_gt.jsonl
  evaluation_labels.json
  evaluation_splits.json
  evaluation_protocol.json
  distribution_alignment.json
  qa_revision_manifest.json
  summary.json
  release_status.json
  video_manifest.json
  private/qa_provenance.jsonl
```

`questions.jsonl` and per-episode documents use the reference's public question
keys. Fine-grained frame witnesses and simulator provenance live in the private
ledger. Do not concatenate annotations or diagnostic fields into model prompts.

`model_inputs.jsonl` reproduces the AI2THOR prompt function and its field
allowlist. The existing STORM evaluator should use **`evaluation_gt.jsonl`** with
the dataset root as `--video-root`. These canonical rows contain the options in
their final display order and bypass the evaluator's automatic legacy shuffle.
Their prompt also includes a uniform query-time anchor for all conditions.
Do not give that evaluator `questions.jsonl` directly: it would shuffle the
balanced answers again and cannot infer VirtualHome's nested video paths.

The primary protocol uses only frames at or before `query_time`. Report known and
uncertain splits separately, plus the combined score. Full-video offline runs
are separate future-context diagnostics. Keep the uniform four-choice baseline
at 25%; do not interpret that theoretical baseline as a measured text-only score.

Structural validation, video hash verification, full RGB semantic review, and
evaluation are separate status fields. A structurally valid export remains a
candidate until the declared semantic review and evaluation are complete.
