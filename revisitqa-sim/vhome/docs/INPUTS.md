# Input contracts

All JSON files use UTF-8. Times are seconds from clip start; object IDs retain their integer type. Supply inputs from the same rollout export. The generator expects evidence records, not arbitrary videos or captions.

## Rollout source for `collect`

`SOURCE/manifest.json` is an object with an `episodes` list. Each episode has:

- `episode_id`: unique string.
- `capture`: absolute capture directory or a path relative to `SOURCE`.
- `scene`, `room_id`, `room`: scene ID, room ID, and readable room category.
- `duration_seconds`: clip duration.
- `hashes`: object containing the SHA-256 of `video.mp4`.

Each capture directory contains:

```text
events.json
stages.json
frame_manifest.jsonl
qa.json
evidence/STAGE_KEY.json
evidence/STAGE_KEY.png
INSTANCE_MASK_FILES
```

`events.json` is a list. Each event contains `event_index`, `object_id`, `object`, `object_description`, `verb`, `start`, `end`, `before_time`, `discovery_time`, `visibility`, `before_state`, `after_state`, and `stage_keys` with `before` and `after` keys. Event indices are expected to match their zero-based list positions for the historical retiming stage. Verbs include `Open`, `Close`, `SwitchOn`, `SwitchOff`, and object-handling actions. Descriptions must distinguish targets within the episode.

`stages.json` maps stage keys to `time` and `visible_pixels`. Each frame-manifest row has `frame`, `rgb`, and `mask`. Frame numbers are contiguous and zero-based. `rgb` and `mask` are paths relative to the capture directory. `qa.json` supplies `sample_fps`, the native capture frame rate used by the masks; this is distinct from the 1 FPS evaluator sampling rate.

Each stage evidence JSON contains `graph.nodes`, `graph.edges`, `color`, and `instance_colors`, with optional `color_collision_ids`. Nodes have `id`, `class_name`, and `states`; edges have `from_id`, `to_id`, and `relation_type`. Colors are normalized RGB triples. The target's color must map uniquely to its object ID.

## Evidence bundle for `run` or `generate`

The collector produces an object with an `episodes` list. Each episode has:

- `metadata`: the manifest episode record above.
- `events`: its event list.
- `observations`: before/after graph and visibility records collected from stage evidence.
- `qa`: the capture QA metadata, including `sample_fps`.
- `events_sha256`: hash of the original event file.

The original event-file hash remains a source identifier after retiming; it is not the hash of the transformed inline event list. Retiming retains `original_events` and records timestamp changes separately. The bundle also contains the reference annotations and their source hash for provenance.

## Visibility audit

An object with an `episodes` list. Each entry has `episode_id`, `objects` (ordered integer object IDs), `counts` (frame-by-object pixel matrix), `fps`, `frames`, and `manifest_sha256`. `frames` must equal the matrix row count; every row must have one nonnegative count per object. `round(time * fps)` indexes a frame. Coverage must include every object referenced by the episode's events and every observation used to generate a question.

## Physical endpoint review

A JSON list of review records. Each record has:

- `episode_id`, `object_id`, `name`.
- `approved`: boolean; only approved records generate known physical-state questions.
- `sheet_index`: review-source identifier retained in provenance.
- `examples`: exactly two records, each containing `time`, `frame`, and `state`.

Example times use the integer 1 FPS clock, both targets must meet the 128-pixel threshold, and states must be visually reviewed. Supported state labels are `closed`/`open`, `off`/`on`, and `held by a person`/`resting on a surface`, according to the object's action domain. For televisions, the question describes screen appearance rather than inferring electrical power. The paired examples should represent distinct states for sequence questions.

Review annotations are external dataset inputs. The package cannot infer their approval from simulator state. The default profile requires known physical-state candidates, so an empty review is not sufficient for the full default release.

## Reference annotations and quota profile

The AI2THOR reference is a JSONL of full question records containing `id`, `episode_id`, `query_time`, `question_type`, `question_subtype`, `question`, `options`, `answer_index`, `evidence_spans`, `video_evidence`, `diagnostics.epistemic_status`, `diagnostics.uncertainty_sources`, and `change_intensity`. Additional fields are retained in field-presence distribution reports.

`qa290.json` contains two mappings: `types` gives total counts for each of the six question types; `uncertain` gives the uncertain subset for each type. The reference file supplies comparison statistics; an explicit quota file supplies the selection targets. Replacing the reference file alone does not recompute quotas.
