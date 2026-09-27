# Verified moving-observer episode

Episode `patrol_06` was recorded on Linux on 2026-09-18. The original capture is stored externally and is not included in this code release.

The blue observer walks between observation points while the green injector handles the bowl. The hidden phase follows a route inside the living room to keep the kitchen operation out of view. A final short turn confirms the returned bowl. Checkpoint holds last 0.15 seconds.

| Check | Result |
| --- | --- |
| Video | 1,340 frames, 20 fps, 640 × 480, 67.0 seconds |
| Moving frames | 80.75% |
| Total stationary time | 12.9 seconds, including turns and checkpoints |
| Longest stationary interval | 1.75 seconds |
| Hidden injection | 19.90–41.00 seconds, 422 frames |
| Target, injector, and original support in hidden interval | Zero visible pixels |
| Container in hidden interval | At most one visible pixel; container invisibility is not an acceptance condition |
| Absence confirmed | 48.20 seconds |
| Target first reobserved | 52.10 seconds |
| Final return confirmed | 67.00 seconds |
| Injector visible during return | 243 frames |
| Camera | 3,162 samples, 1,091 retractions, zero collider overlaps |
| Questions | Eight across six categories, including two uncertain answers |
| Tests | 18 passed |

Movement uses horizontal hip displacement over a 0.5-second window and a 0.15 m/s threshold. This is a mostly moving patrol, with brief navigation, turning, and inspection pauses. The camera result applies to the recorded route and the simulator's collision geometry.

Run another capture with:

```bash
cd /path/to/STORM-Bench/storm-sim/vhome
.venv/bin/python scripts/rollout.py \
  --executable .runtime/simulator-observer-v4/linux_exec.v2.3.0.x86_64 \
  --config configs/observer_pair.yaml --xorg-root .runtime/xorg --gpu-index 3 \
  --output outputs/patrol_new
.venv/bin/python scripts/validate.py outputs/patrol_new
.venv/bin/python scripts/export_model_inputs.py outputs/patrol_new \
  --output outputs/patrol_new_evaluation
```

Navigation timing varies between simulator runs. Each capture must pass the frame-level visibility and motion checks independently. Trials `patrol_01` through `patrol_05` are rejected debugging records, not dataset episodes.

The source capture and its full RGB/mask frames remain external. Run the complete episode validator against your own capture directory.
