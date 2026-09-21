# STORM-Bench

This repository is the code release for the STORM-Bench paper. It contains the real-video data construction pipeline and the public evaluation path for the six reported subsets: Cook (`storm_real`), Bike, Health, Music, Sports, and Sim (`storm_sim`). It excludes training, method development, paper figures, and frozen result directories. Dataset media, annotations, and model checkpoints are supplied separately.

The [real-video construction pipeline](storm-real/README.md) in `storm-real/` converts long egocentric videos into same-region revisit episodes and temporally grounded QA. It includes dedicated English prompts for Cook, Bike, Health, Music, and Sports, along with installation instructions and offline tests.

Run the 14-checkpoint evaluation with:

```bash
python scripts/eval/run_storm_models.py --models all --protocols online \
  --checkpoint-root /path/to/checkpoints \
  --gt-file /path/to/questions.jsonl --video-root /path/to/video \
  --cuda-devices 0,1,2,3 --with-status
```

The registry in `scripts/eval/storm_model_registry.py` contains the 14 checkpoint names. `scripts/eval/storm_model_adapters.py` provides the family-specific loaders and `scripts/eval/storm_streaming.py` enforces the 1 FPS causal frame protocol. GPU runtimes and model weights are intentionally not vendored.

Install dependencies with `pip install -r requirements-eval.txt` after installing a CUDA-compatible PyTorch build. Place checkpoints below `ckpt/` using the names in the registry. Supply one released STORM manifest and its matching video directory through `--gt-file` and `--video-root`; the runner records manifest hashes and rejects mismatched predictions. Use `--protocols offline,online` to reproduce both reported protocol conditions. Outputs are written to `outputs/`, which is ignored by Git.
