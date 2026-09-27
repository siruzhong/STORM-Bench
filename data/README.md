# Data release

This directory holds the STORM-Bench data release: question annotations, splits, per-video ID/timestamp manifests, and a
few example clips. Full video media is not redistributed.

```
data/
  summary.json                         per-subset counts and label distributions
  annotations/<subset>.questions.jsonl released multiple-choice items
  manifests/<subset>.videos.jsonl      episode IDs, relative video paths, query timestamps
  splits/storm_sim.evaluation_splits.json    official simulation Known/Uncertain splits
  splits/storm_sim.evaluation_protocol.json  simulation evaluation protocol
  examples/<subset>/<clip>.mp4         one small clip per shipped subset
  examples/<subset>/<clip>.questions.jsonl  the QA rows that reference that clip
```

## Subsets

| Subset key | Paper name | Track | Source | QA | Episodes |
|---|---|---|---|---|---|
| `storm_cook` | Cook | Real | HD-EPIC | 2,640 | 314 |
| `storm_bike` | Bike Repair | Real | Ego-Exo4D | 786 | 61 |
| `storm_healthy` | Health | Real | Ego-Exo4D | 800 | 83 |
| `storm_music` | Music | Real | Ego-Exo4D | 120 | 16 |
| `storm_sports` | Sports | Real | Ego-Exo4D | 800 | 99 |
| `storm_sim` | THOR | Sim | AI2-THOR | 300 | 28 |
| `storm_vhome` | VHome | Sim | VirtualHome | 290 | 29 |

All seven subsets ship in this snapshot, totaling 5,736 questions over 630
episodes (4,620 Known / 1,116 Uncertain): STORM-Real spans five egocentric
domains (6.66 h) and STORM-Sim spans two controlled simulators (0.47 h THOR,
0.54 h VHome). VHome is `annotations/storm_vhome.questions.jsonl` and
`manifests/storm_vhome.videos.jsonl`, with its split files under
`splits/storm_vhome.*.json`.

## Annotation schema

`annotations/<subset>.questions.jsonl` is one JSON object per line:

| Field | Meaning |
|---|---|
| `id` | Globally unique item id |
| `episode_id` | Episode/video identifier; determines the relative video path |
| `query_time` | Query timestamp `t_q` in seconds; only frames `<= t_q` are visible |
| `question_type` | One of the six semantic types (current_state, factual_retrieval, history_aggregation, object_tracking, state_change, temporal_reasoning) |
| `question_subtype` | Finer-grained construction label |
| `question` | Question text (without options) |
| `options` | Four answer options in stored order |
| `answer_index` | Index into `options` of the correct answer |
| `evidence_spans` | `[start, end]` second intervals supporting the answer, inside `[0, t_q]` |
| `video_evidence` | Short free-text description of the supporting evidence |
| `change_intensity` | Ordinal accumulated change-intensity score `c_i` in 1-10; Low `[1,3]`, Medium `[4,6]`, High `[7,10]` |
| `diagnostics.epistemic_status` | `known` or `uncertain` |
| `diagnostics.uncertainty_sources` | Subset of the six controlled uncertainty causes |
| `diagnostic_rationale` | `volatility` and `uncertainty` rationale strings |

The six uncertainty causes, with the paper's root-cause names, are
`missing_observation` (missing observation), `partial_observation` (partial
observation or occlusion), `ambiguous_evidence` (ambiguous evidence),
`low_visual_quality` (low visual quality), `multiple_candidates` (multiple
plausible candidates), and `ambiguous_attribute` (ambiguous entity attribute).

At evaluation time the loader applies a deterministic, sample-specific
permutation to the options (`sha256(id:index)`); `answer_index` is interpreted
against the stored order, and the shuffled order is reproducible from `id`.

## Video manifests and media layout

`manifests/<subset>.videos.jsonl` lists one row per episode:

```
{"subset", "episode_id", "participant", "video_path", "num_questions", "query_times"}
```

`video_path` is the path relative to the subset `video/` root expected by the
evaluator, e.g. `P02/P02-..._revisit.mp4` for Cook participants,
`FloorPlan406/STORM_...mp4` for THOR, `Scene0_Room11/<episode>.mp4` for VHome,
and flat `<episode>.mp4` for the Ego-Exo4D domains. To evaluate subset `S`, place the downloaded clips under one
root so that `<root>/<video_path>` resolves, then pass that root as
`--video-root` together with `--gt-file data/annotations/S.questions.jsonl`.

Source media:

- **Cook** uses HD-EPIC videos (`https://hd-epic.github.io/`).
- **Bike Repair, Health, Music, Sports** use Ego-Exo4D videos
  (`https://ego-exo4d-data.org/`).
- **THOR** episodes are rendered with AI2-THOR (`https://ai2thor.allenai.org/`).
- **VHome** episodes are rendered with VirtualHome.

## Splits and protocol

`data/splits/{storm_sim,storm_vhome}.evaluation_splits.json` define the official
simulation Known/Uncertain splits and the headline metric; the matching
`.evaluation_protocol.json` files record the frame rule (`timestamp <= query_time`),
the required modality controls, and the matched-vs-shuffled gain thresholds. The
real-track items carry their labels inline and do not need a separate split file.

## Examples

`data/examples/<subset>/` contains one small clip per shipped subset, chosen to
be representative and under ~35 MB. Each clip is paired with
`<episode_id>.questions.jsonl`, the exact annotation rows that reference it (same
schema as `data/annotations/`), so the example is self-contained; `data/examples/README.md`
tables the clip-to-episode-to-QA mapping. They illustrate the episode format; they are not the full evaluation set.

## Construction summary

The STORM-Real track mines revisit episodes from HD-EPIC and Ego-Exo4D by
segmenting 1 FPS streams into region-activity timelines and assembling 3-10
chronologically ordered clips, sped up 1.5x with 0.2-second transition buffers,
into compact episodes targeting 45 s (mean durations 39.00-42.86 s);
evidence-grounded QA pairs are then generated across the six semantic types with
evidence spans and diagnostic labels. The STORM-Sim track renders scripted
rollouts in AI2-THOR and VirtualHome across four room categories (Kitchen,
Living Room, Bedroom, Bathroom) and four construction stages, then instantiates
deterministic question builders that emit balanced options and diagnostic labels
(VHome adds an abstention option on every item). Construction-time provenance
and internal tool paths are removed from the released annotations; only fields
required for evaluation and scoring are kept.

## Licenses and attribution

The question annotations and example clips are derived from HD-EPIC and
Ego-Exo4D and remain subject to the licenses and terms of those datasets; the
THOR and VHome annotations and clips are generated from the AI2-THOR and
VirtualHome simulators. Please cite the corresponding source datasets when
using each subset, and follow their non-commercial and attribution terms.
