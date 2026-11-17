# Evaluation reference

[Back to the project](../../README.md)

## Installation

Several model families pin conflicting `transformers` versions. Create the four environments from a base interpreter that already provides a CUDA-matched PyTorch build:

```bash
BASE_PYTHON=/path/to/torch/python bash scripts/setup_revisitqa_envs.sh
```

| Environment | Runtime | Families |
|---|---|---|
| `.venv-revisitqa-vlm` | `vlm` | Qwen2.5-VL, Qwen3-VL, InternVL3.5, Molmo2, MiniCPM-V, Eagle2.5, LLaVA-NeXT-Video |
| `.venv-revisitqa-modern` | `modern` | Qwen3.5, GLM-4.1V |
| `.venv-revisitqa-internvideo` | `internvideo` | InternVideo2.5 |
| `.venv-revisitqa-videollama3` | `videollama3` | VideoLLaMA3 |

For pure scoring (no model inference):

```bash
pip install -r requirements-eval.txt
```

## Checkpoints

`revisitqa_model_registry.py` declares the open-weight **offline video LLM** suite from the paper (15 models). Place each under `ckpt/<local_name>` (or pass `--checkpoint-root`). No weights are vendored. The paper's 7 streaming models use the streaming runners below.

```bash
python - <<'PY'
from scripts.eval.revisitqa_model_registry import MODEL_SPECS
for key, spec in MODEL_SPECS.items():
    print(f"{key:22s} {spec.repo_id:42s} ckpt/{spec.local_name}")
PY
```

Streaming models use their native interfaces via `run_revisitqa_streaming_native.py` / `eval_revisitqa_streaming.py`.

## Evaluation protocol

Primary protocol (paper): **1 FPS causal prefix** \(V_{\le t_q}\) (up to 200,704 pixels per frame). `revisitqa_streaming.py` keeps frames with timestamp \(\le t_q\). Offline video LLMs re-encode that prefix per query; streaming models update once per frame without writing answers back.

| CLI `--protocols` | Paper meaning |
| --- | --- |
| `online` | Causal-prefix evaluation \(V_{\le t_q}\) |
| `offline` | Full-video control \(V\) (includes frames after \(t_q\)) |

Every item presents three substantive options plus a shared abstention option with fixed wording and position (*Cannot be determined from the video up to this point.*).

Metrics (`scripts/eval/revisitqa_reliability.py`), matching the paper:

- **KA** — Known accuracy
- **AR** — abstention recall on Uncertain items
- **H** — harmonic mean of KA and AR (primary ranking); chance baseline \(H=26.8\)
- **FAR** — false abstention on Known items
- Effective reliability / plain accuracy are reference-only

Confidence intervals use a 1,000-replicate episode-clustered bootstrap; headline tables use unweighted means over the seven subsets.

## Running evaluation

```bash
python scripts/eval/run_revisitqa_models.py \
  --models all --protocols online \
  --checkpoint-root ckpt \
  --subsets cook,bike,healthy,music,sports,thor,vhome
```

Useful flags:

- `--protocols online,offline`
- `--text-only`, `--shuffle-frames`, forced-choice controls (see `eval_revisitqa_model.py`)
- `--fps 1.0 --max-pixels 200704 --seed 20260817 --bootstrap-replicates 1000`
- `--sample-limit` / `--episode-limit` / `--dry-run` for smoke tests

Default output root: `outputs/revisitqa/released/<subset>/models/<protocol>/<model>/`.

## Scoring existing predictions

```bash
python scripts/eval/score_revisitqa_predictions.py \
  --gt-file data/eval_video/RevisitQA/revisitqa_cook/cook_question_answer.jsonl \
  --pred-file outputs/revisitqa/released/cook/models/online/qwen3_vl_8b/pred.jsonl
```

To remap a full previous suite onto a later release JSONL (kept items only):

```bash
python scripts/eval/rescore_revisitqa_subset.py \
  --model qwen3_vl_8b \
  --source outputs/revisitqa/previous \
  --target outputs/revisitqa/released
```


## Annotation fields used by the evaluator

See [data/README.md](../../data/README.md). Required: `id`, `episode_id`, `query_time`, `options`, `answer_index`, `abstention_index`, `diagnostics.epistemic_status`, plus evidence/intensity fields for stratified tables.
