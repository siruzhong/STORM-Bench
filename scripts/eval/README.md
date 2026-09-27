# Evaluation reference

[Back to the project](../../README.md)

## Installation

The suite covers several model families that pin conflicting `transformers`
versions, so `run_storm_models.py` launches one worker per environment. Create
the four environments from a base interpreter that already provides a
CUDA-matched PyTorch build:

```bash
BASE_PYTHON=/path/to/torch/python bash scripts/setup_storm_envs.sh
```

This creates `.venv-storm-vlm`, `.venv-storm-modern`, `.venv-storm-legacy`, and
`.venv-storm-videollama3`. The runtime-to-family assignment is:

| Environment | Runtime | Families |
|---|---|---|
| `.venv-storm-vlm` | `vlm` | Qwen2.5-VL, Qwen3-VL, InternVL3.5, Molmo2, MiniCPM-V, Eagle2.5, LLaVA-NeXT-Video |
| `.venv-storm-modern` | `modern` | Qwen3.5, GLM-4.1V |
| `.venv-storm-legacy` | `legacy` | InternVideo2.5 |
| `.venv-storm-videollama3` | `videollama3` | VideoLLaMA3 |

For pure scoring (no model inference) a single environment is enough:
install a CUDA-matched PyTorch build, then `pip install -r requirements-eval.txt`.
VideoLLaMA3 and Eagle2.5 require FlashAttention-2 for the official inference
path, and the loader raises if it is missing; `setup_storm_envs.sh` installs a
CUDA-matched wheel, and the other families fall back to SDPA.

## Checkpoints

`scripts/eval/storm_model_registry.py` declares 14 open-weight checkpoints over
11 architectures. Place each under `ckpt/<local_name>` (or point
`--checkpoint-root` elsewhere). No weights are vendored.

```bash
python - <<'PY'
from scripts.eval.storm_model_registry import MODEL_SPECS
for key, spec in MODEL_SPECS.items():
    print(f"{key:22s} {spec.repo_id:42s} ckpt/{spec.local_name}")
PY
```

## Running the evaluation

Evaluate the full suite on one subset, both protocol conditions, with the
decoupled status probe that STORM-BR requires:

```bash
python scripts/eval/run_storm_models.py --models all --protocols offline,online \
  --checkpoint-root ckpt \
  --gt-file data/annotations/storm_cook.questions.jsonl \
  --video-root /path/to/storm_cook/video \
  --cuda-devices 0,1,2,3 --with-status
```

Run this once per subset, swapping `--gt-file` and `--video-root`. Useful flags:

- `--models all` or a comma-separated list of registry keys.
- `--protocols offline,online` (`online` is the causal protocol; `offline` is
  the whole-video control).
- `--with-status` enables the Known/Uncertain and uncertainty-source probes.
- `--text-only` and `--shuffle-frames` run the modality controls.
- `--fps 1.0`, `--max-pixels 200704`, `--seed 20260817`, `--bootstrap-replicates 1000`
  mirror the reported protocol (frames are resized to a 200,704-pixel budget,
  retaining the latest frames).
- Standard models decode greedily up to 8 tokens; GLM-4.1V uses a 512-token
  thinking budget followed by a separate answer completion.
- `--sample-limit`, `--episode-limit`, `--dry-run` for smoke tests.

Outputs go to `--output-root` (default `outputs/storm_bench/models`) as
`<protocol>/<model_key>/pred.jsonl` plus `result.json`. The runner records the
ground-truth manifest hash and rejects mismatched or incomplete predictions.

## Scoring existing predictions

To recompute STORM-BR and diagnostics from a `pred.jsonl` without running any
model:

```bash
python scripts/eval/score_storm_diagnostics.py \
  --gt-file data/annotations/storm_cook.questions.jsonl \
  --pred-file outputs/storm_bench/models/online/qwen3_vl_8b/pred.jsonl
```

## Online protocol and metrics

At 1 FPS, `storm_streaming.py` builds the visible prefix per query by taking
frames with timestamp `<= t_q` (with the configured transition buffer). The
`offline` condition exposes the whole stream. The two decoupled probes run
sequentially after the primary answer: a sufficiency probe with options
(A) directly observed, (B) inferred via temporal reasoning, (C) observations
missing, (D) observations ambiguous, where A/B map to Known and C/D to
Uncertain; and, when the status is Uncertain, a multi-select attribution probe
over the six uncertainty causes.

Metrics follow the paper. Each item has joint correctness
`J_i = 1[(answer == ground truth) and (status == ground-truth status)]`. The
intensity--answerability grid crosses the three change-intensity buckets
(Low `[1,3]`, Medium `[4,6]`, High `[7,10]`) with the two epistemic states
(Known, Uncertain). Each occupied cell is Laplace add-1 smoothed,
`(sum J_i + 1) / (n + 2)`, and STORM-BR is the harmonic mean of the smoothed
occupied cells. STORM-BR-Attr replaces joint correctness on Uncertain items with
`J_i * F_i` (where `F_i` is the F1 of the predicted uncertainty causes) and takes
the harmonic mean of the unsmoothed occupied cells, zero if any is zero. Cook,
Bike Repair, Health, Sports, THOR, and VHome populate all six cells; Music lacks
high-intensity events and is scored over its four occupied cells. Confidence
intervals use a 1,000-replicate episode-clustered bootstrap, and overconfidence
(OC) is the fraction of Uncertain items predicted as Known.

## Annotation schema

Each line of `data/annotations/<subset>.questions.jsonl` is one item with the
fields used by the evaluator:

```
id, episode_id, query_time, question_type, question_subtype,
question, options (4), answer_index, evidence_spans,
video_evidence, change_intensity,
diagnostics: { epistemic_status: known|uncertain,
               uncertainty_sources: [missing_observation, partial_observation,
                 low_visual_quality, ambiguous_evidence, ambiguous_attribute,
                 multiple_candidates] },
diagnostic_rationale: { volatility, uncertainty }
```

`question_type` is one of six semantic types (current_state, factual_retrieval,
history_aggregation, object_tracking, state_change, temporal_reasoning), and
`change_intensity` is the ordinal score `c_i` in `1`--`10` bucketed into Low
`[1,3]`, Medium `[4,6]`, and High `[7,10]`. The six `uncertainty_sources` labels
map to the paper's root causes: `missing_observation` (missing observation),
`partial_observation` (partial observation or occlusion), `ambiguous_evidence`
(ambiguous evidence), `low_visual_quality` (low visual quality),
`multiple_candidates` (multiple plausible candidates), and `ambiguous_attribute`
(ambiguous entity attribute).

Option order is fixed at load time by a deterministic sample-specific
permutation (`sha256(id:index)`), so `answer_index` refers to the stored option
order, and the shuffled order is reproducible from the item `id`.
