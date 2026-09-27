# STORM Real-Video Pipeline

A compact pipeline for constructing same-region revisit episodes and temporally grounded, four-choice questions from long egocentric videos. Five English workflows share the same implementation while retaining their original domain-specific prompts, region definitions, activities, and generation settings.

| `--domain` | Content | Functional regions include |
| --- | --- | --- |
| `cook` (default) | Kitchen activity | Sink, food preparation, stove, refrigerator, cabinets |
| `bike` | Bicycle repair | Front/rear wheels, tire/tube, chain, rear derailleur, tools |
| `health` | Health procedure demonstrations | Patient/mannequin, sample collection, test kit, instructions/records, supplies/PPE |
| `music` | Music performance | Instrument playing, score reference, controls/accessories, ensemble scene |
| `sports` | Basketball, soccer, climbing/bouldering | Separate discipline-specific court/field/route, ball/hold, goal/hoop, drill and rest regions |

Each domain has **three dedicated English prompt files**: segmentation, revisit selection, and QA. Sports preserves separate basketball, soccer, and climbing region families. Select the domain explicitly; a run does not classify a mixed input directory automatically.

```text
Source videos
    -> Windowing and timestamped frame sampling
    -> Region-aware semantic segmentation
    -> Same-region visit selection and chronological video assembly
    -> Direct visual QA generation and validation
    -> Per-episode JSON + question-level JSONL
```

## Installation

Use Python 3.10 or newer and an FFmpeg installation with the `libx264` encoder. From the STORM-Bench repository root:

```bash
cd storm-real
python -m venv .venv
source .venv/bin/activate
pip install -e .
ffmpeg -version
```

Copy the environment template and fill in the credentials for a vision-capable API supporting the Chat Completions message format and base64 images:

```bash
cp .env.example .env
# Edit .env, then load it into your shell:
set -a
. ./.env
set +a
```

The package reads environment variables; it does not automatically load `.env`. No credentials or provider account identifiers are distributed. The API must accept the configured token budget and temperature 1. Set `STORM_MAX_TOKENS` if the provider requires a different output limit. `STORM_FFMPEG` can point to FFmpeg when it is not available on `PATH`.

## Run the complete pipeline

```bash
storm-real run data/input.mp4 --domain cook --output-dir outputs
# Or recursively process a directory:
storm-real run data/music --domain music --output-dir outputs
storm-real run data/sports --domain sports --output-dir outputs
```

`python -m storm_real_pipeline` is equivalent to `storm-real`. The command runs the stages sequentially, continues after individual source/episode failures, writes a run summary, and exits nonzero if any stage fails. It requires API credentials for segmentation and QA. `--no-llm-selection` disables only the optional model-based revisit proposal.

```bash
storm-real run data/bike --domain bike --output-dir outputs --no-llm-selection
storm-real run data/music --domain music --output-dir outputs --region instrument_playing_area
```

The optional `--thinking disabled` flag sends the provider-specific `thinking` extension for segmentation and QA. It is omitted by default. Use it only with a compatible provider.

Each source receives a domain-prefixed pseudonymous ID derived from its relative input path. The original path is not written to segmentation or QA records. Keep the input root consistent between runs to retain IDs. Content hashes identify changed inputs without distributing the videos themselves. Full-pipeline and segmentation commands always create a domain subdirectory under `--output-dir`, including for Cook.

```text
outputs/
  music/
    segments/
      music_video_<hash>_window001.json
    episodes/
      music_video_<hash>_window001_revisit.mp4
      music_video_<hash>_window001_revisit.plan.json
    qa/
      music_video_<hash>_window001_revisit.json
    questions.jsonl
    run_summary.json
  sports/
    ...
```

`questions.jsonl` contains the successfully completed questions from the current invocation, including valid reused outputs. If every episode fails, an existing merged export is preserved. Artifacts already present from unrelated inputs are not silently included. To explicitly combine all QA files in a directory, use the merge command below.

## Run individual stages

```bash
storm-real segment data/input.mp4 --domain music --output-dir outputs

storm-real assemble outputs/music/segments/music_video_0123456789abcdef_window001.json \
  --video data/input.mp4 --domain music --output-dir outputs/music/episodes

storm-real qa outputs/music/episodes/music_video_0123456789abcdef_window001_revisit.mp4 \
  --domain music --output-dir outputs/music/qa

storm-real merge outputs/music/qa outputs/sports/qa --output outputs/questions.jsonl
```

Replace the example ID with an actual generated ID. Use the same `--domain` at every stage. Assembly and QA write directly into their explicit `--output-dir`; their defaults are `outputs/<domain>/episodes` and `outputs/<domain>/qa`. Stage commands use explicit source paths; assembly does not infer a private dataset layout. QA automatically finds the adjacent `.plan.json` and checks its domain. Bike, Health, Music, and Sports require a valid plan. Cook also supports an independent episode with `--episode-id`; without a plan, its compatibility fallback sets `change_intensity` to one, which is not a measured visit count. The full pipeline always supplies a plan.

Merge accepts one or more QA directories, validates each document against its domain, and includes `domain` in every JSONL row. Domain-prefixed IDs prevent the same input name in different domains from colliding.

Use `--overwrite` to regenerate artifacts intentionally. Segmentation and assembly reject stale or invalid cache records and ask for this flag. QA validates its cached inputs/configuration and regenerates stale artifacts. Changes to the source content, prompts, or relevant parameters must not silently reuse unrelated outputs. Run `storm-real <command> --help` for options.

## Core behavior

| Stage | Default behavior |
| --- | --- |
| Segmentation | 300 s processing windows; a tail shorter than 45 s is merged into the preceding window; 1 FPS; batches of 64 frames; Gaussian blur and JPEG quality 75 |
| Boundaries | Functional region, dominant activity, or a definite visible outcome; complete, contiguous, non-overlapping intervals |
| Visit candidates | Observation or interaction in one named region; navigation, unusable content, and `other_area` are excluded |
| Selection | Validate both LLM and deterministic proposals, then choose the valid plan closest to the target duration |
| Assembly | Cook: 3–10 visits; other domains: 2–10 visits. Each source clip is 3–20 s; 1.5x playback; 0.2 s black gaps and fades; target 45 s, maximum 60 s |
| QA | Directly sample the assembled episode at 1 FPS; JPEG quality 80; at least 8 Known and 2 Uncertain questions, at most 2 per semantic type |

Segmentation allows three schema attempts for Cook/Bike and five for Health/Music/Sports. All domains use strict schema validation; malformed model segments are retried rather than repaired by guessing missing semantic fields. Revisit selection preserves the original domain region priorities. QA allows five generation rounds for Cook/Bike and twelve for Health/Music/Sports; the latter request extra uncertainty candidates when needed. The four non-Cook domains retain deterministic balancing of correct-option positions.

A short tail can make a processing window longer than 300 s. Visits may contain interactions and visible changes; they need not be static. Black gaps mark omitted source intervals and do not assert that a definite change occurred in every gap. Fade frames lie inside visit clips and do not add another interval to the duration. Actual encoded frame counts determine output visit boundaries.

The QA generator receives sampled frames from the **complete assembled episode**. Every query and evidence endpoint is aligned to a sampled timestamp, and every evidence span must end at or before its query. This is an evidence-annotation constraint, not a claim that future frames were physically withheld from the annotation model. Segmentation records and revisit plans are not supplied as answer evidence.

### Question types and fields

The six semantic types are `factual_retrieval`, `current_state`, `state_change`, `object_tracking`, `temporal_reasoning`, and `history_aggregation`. Each question has four distinct options and a zero-based `answer_index`.

The core fields are:

- `query_time`, `question_type`, `question_subtype`, and `question`;
- `options`, `answer_index`, `video_evidence`, and minimal `evidence_spans`;
- `diagnostics.epistemic_status`: `known` or `uncertain`;
- `diagnostics.uncertainty_sources` and explanatory diagnostic rationales;
- `change_intensity`: the visit count, or Cook's no-plan fallback of one.

Known means the visible evidence supports one answer. Uncertain means it does not; the correct option must explicitly express the relevant inability to determine the answer, and at least one allowed uncertainty source must be present. The six question types and the two epistemic labels are separate dimensions.

The four non-Cook prompts require the uncertain correct option to be exactly `Cannot be determined`. Health, Music, and Sports retain normalization of recognized English uncertainty wording and source labels before final validation. Their prompts also retain domain-specific evidence and object-identity restrictions; changing `--domain` changes the complete prompts, not just a domain name in a shared template.

QA `change_intensity` preserves the original definition: the cumulative number of visits that have started by the query time, with a minimum of one. The plan's count of annotated changes in omitted intervals is named **`hidden_change_segment_count`** to distinguish it from this QA field.

## Repository layout

```text
src/storm_real_pipeline/
  cli.py                  # Whole-pipeline and stage commands
  client.py               # Shared environment-configured VLM client
  domains.py              # Five domain schemas, enums, and original settings
  segmentation.py         # Former llm_seg.py core
  revisit.py              # Former llm_gene_video.py core
  qa.py                   # Former qa_gen.py core
  io_utils.py             # Atomic JSON writes and hashing
  prompts/
    cook/                 # segmentation.txt, revisit.txt, qa.txt
    bike/                 # Same three stages, bicycle-specific instructions
    health/               # Health procedure-specific instructions
    music/                # Performance-specific instructions
    sports/               # Basketball, soccer, and climbing instructions
tests/                    # Offline contract and integration checks
```

Only English prompts are packaged. Personal paths, account information, original video assets, generated annotations, logs, historical backups, and dataset-specific balancing/repair scripts are excluded. Pseudonymous filenames do not anonymize the visual content of a video or text a model may observe; input media remain outside this code release. Human review is not implemented by these scripts.

## Offline checks

```bash
python -m unittest discover -s tests -v
```

Tests cover all five prompt sets, domain schemas, visit thresholds, output timestamp mapping, QA constraints, cross-domain rejection, cache consistency, and orchestration. The integration test runs all five domains using a synthetic video and deterministic mock model responses, then resumes and merges them: no network or API credits are needed. It verifies the implementation, not the semantic quality of a real model's annotations. Real codec checks require FFmpeg and OpenCV.
