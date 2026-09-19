# STORM-Bench

This repository is the code release for the STORM-Bench paper. It contains the public evaluation path for the six reported subsets: Cook (`storm_real`), Bike, Health, Music, Sports, and Sim (`storm_sim`). It excludes training, method development, paper figures, and frozen result directories. Dataset media, annotations, and model checkpoints are supplied separately.

Run the 14-checkpoint evaluation with:

```bash
python scripts/eval/run_storm_models.py --models all --protocols online \
  --checkpoint-root /path/to/checkpoints \
  --gt-file /path/to/questions.jsonl --video-root /path/to/video \
  --cuda-devices 0,1,2,3 --with-status
```

The registry in `scripts/eval/storm_model_registry.py` contains the 14 checkpoint names. `scripts/eval/storm_model_adapters.py` provides the family-specific loaders and `scripts/eval/storm_streaming.py` enforces the 1 FPS causal frame protocol. GPU runtimes and model weights are intentionally not vendored.
