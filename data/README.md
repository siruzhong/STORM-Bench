# Data (code-repo side)

This repository includes example clips under `data/examples/`. Full JSONL and videos are on Hugging Face: [siruzhong/RevisitQA](https://huggingface.co/datasets/siruzhong/RevisitQA).

```text
data/
  examples/<subset>/                 one demo mp4 + sibling questions.jsonl per subset
```

Release size: **2,577 questions / 622 episodes** (1,902 Known, 675 Uncertain; median duration **44.8 s**; ~7.56 h).

## Subsets

| Subset key | Paper name | Track | Source | QA | Episodes |
|---|---|---|---|---|---|
| `revisitqa_cook` | Cook | Real | HD-EPIC | 990 | 307 |
| `revisitqa_bike` | Bike Repair | Real | Ego-Exo4D | 389 | 61 |
| `revisitqa_healthy` | Health | Real | Ego-Exo4D | 297 | 82 |
| `revisitqa_music` | Music | Real | Ego-Exo4D | 76 | 16 |
| `revisitqa_sports` | Sports | Real | Ego-Exo4D | 306 | 99 |
| `revisitqa_thor` | THOR | Sim | AI2-THOR | 309 | 28 |
| `revisitqa_vhome` | VHome | Sim | VirtualHome | 210 | 29 |

## Hugging Face layout (evaluation)

Each subset folder on HF is:

```text
revisitqa_<name>/
  <name>_question_answer.jsonl
  video/RevisitQA_<...>.mp4
```

Point the runners at a local checkout (default path below, or set `REVISITQA_ROOT`):

```bash
# example: clone/copy HF contents so this resolves
export REVISITQA_ROOT=data/eval_video/RevisitQA
# $REVISITQA_ROOT/revisitqa_cook/cook_question_answer.jsonl
# $REVISITQA_ROOT/revisitqa_cook/video/...
```

`data/eval_video/` is gitignored (local HF checkout cache).

## Annotation schema

Each JSONL line is one item:

| Field | Meaning |
|---|---|
| `id` | Globally unique item id (`RevisitQA_<domain>_...`) |
| `episode_id` | Episode identifier (`RevisitQA_<domain>_...` in this release) |
| `query_time` | Query timestamp \(t_q\) in seconds; only frames \(\le t_q\) are visible |
| `question_type` | One of six semantic types: `factual_retrieval`, `current_state`, `state_change`, `object_tracking`, `temporal_reasoning`, `history_aggregation` |
| `question_subtype` | Construction label |
| `question` | Question stem |
| `options` | Four options; last slot is the shared abstention text when `abstention_index=3` |
| `answer_index` | Index of the gold option in stored `options` |
| `abstention_index` | Index of the shared abstention option (fixed wording) |
| `evidence_spans` | `[start, end]` intervals inside `[0, t_q]` |
| `change_intensity` | Ordinal \(c_i \in [1,10]\); bins Low `[1,3]`, Medium `[4,6]`, High `[7,10]` |
| `diagnostics.epistemic_status` | `known` or `uncertain` (Known iff the prefix supports a unique substantive answer) |
| `diagnostics.uncertainty_sources` | Subset of: missing observation; partial observation/occlusion; ambiguous evidence; low visual quality; multiple plausible candidates; ambiguous attribute |

Abstention wording (identical on every item):

`Cannot be determined from the video up to this point.`

At evaluation time, `scripts/eval/data_io.normalize_mcq_sample` shuffles only the three substantive options and keeps the abstention slot fixed, exposing `abstention_choice` for scoring.

Source media remains under HD-EPIC, Ego-Exo4D, AI2-THOR, and VirtualHome licenses.

## Examples

See [examples/README.md](examples/README.md).
