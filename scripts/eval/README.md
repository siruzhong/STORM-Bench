# Model evaluation

`run_storm_models.py` is the single entry point for the 14 registered public checkpoints. It launches one worker per visible GPU, evaluates each model independently, merges chunked predictions, and writes `pred.jsonl` plus `result.json`.

The default protocol is causal online evaluation at 1 FPS. Use `--protocols offline,online` for the full control comparison. `--with-status` enables the decoupled Known/Uncertain probe required for STORM-BR. Checkpoint directories use the names declared in `storm_model_registry.py`; no checkpoint or dataset is included in this repository.
