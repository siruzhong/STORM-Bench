# STORM-Bench

STORM-Bench is an online video QA evaluation harness for dynamic state tracking and abstention. This repository contains only the public evaluation code. Dataset media and annotations are loaded from paths supplied by the user and are not bundled here.

## Quick start

```bash
python -m pip install -e .
python -m storm_bench validate data/questions.jsonl
python -m storm_bench evaluate data/questions.jsonl predictions.jsonl --output results.json
```

The evaluator accepts one JSON object per line in both files. A question record has `id`, `episode_id`, `query_time`, `question_type`, `question`, `options`, `answer_index`, `diagnostics`, and `change_intensity`. A prediction record has `id`, `task_answer` (option index or A-D), `status` (`known` or `uncertain`), and optional `uncertainty_sources`.

The online protocol exposes only frames whose timestamp is at most `query_time`. Candidate order must be permuted with `storm_bench.protocol.permute_options` using the sample id before prompting a model, and restored before scoring. See `examples/predictions.jsonl` and `docs/schema.md` for complete examples.

This code is released for anonymous review. It contains no author names, affiliations, absolute paths, private checkpoints, generated result tables, or construction artifacts.

## License

MIT. Dataset and source-video licenses remain those of their original providers.
