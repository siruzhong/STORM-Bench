# Long ten-event baseline

This older 286.4-second capture does not meet the current 60–70-second requirement. Use `configs/observer.yaml` for the short preset.

`patrol10_01` was recorded on Linux on 2026-09-18 in one continuous VirtualHome scene. Five removal-and-return cycles produce ten events: five disappearances and five reappearances. The blue observer only walks and turns; the green injector performs every manipulation.

| Event | Type | Sequence start (s) | Gap from previous start (s) |
| --- | --- | ---: | ---: |
| 1 | Disappear | 20.05 | — |
| 2 | Appear | 47.50 | 27.45 |
| 3 | Disappear | 77.45 | 29.95 |
| 4 | Appear | 103.25 | 25.80 |
| 5 | Disappear | 132.25 | 29.00 |
| 6 | Appear | 159.15 | 26.90 |
| 7 | Disappear | 188.20 | 29.05 |
| 8 | Appear | 213.75 | 25.55 |
| 9 | Disappear | 242.25 | 28.50 |
| 10 | Appear | 267.80 | 25.55 |

The nine gaps average 27.53 seconds and range from 25.55 to 29.95 seconds. These are starts of recorded manipulation sequences, not estimates of exact physical transition frames. The original capture contains 5,728 frames at 20 fps, lasting 286.4 seconds. It is not retimed, repeated, or stitched from reset episodes.

The final audit passed all five cycles. Across 1,949 offscreen-injection frames, the target, injector, and original support had zero visible pixels. The camera audit checked 13,362 samples, retracted the camera in 4,544 samples, and found zero collider overlaps. The observer moved in 83.90% of frames; total stationary time was 46.1 seconds and the longest individual interval was 1.9 seconds. Movement uses the same 0.15 m/s threshold over a 0.5-second window as the earlier patrol.

The debug video has the same 5,728 frames as the clean video, with visible-target highlighting in 1,046 frames. All 22 code tests passed, including later-cycle visibility rejection, event count and spacing checks, and cumulative history answers.

The long recipe targets a 28-second start gap, with a six-second tolerance for navigation. Observer routes pace the operations while the characters act concurrently. A capture with missing events, an out-of-range gap, a visibility failure, or excessive observer pauses fails acceptance. Changing the interval target alone does not shorten a route; adjust the patrol waypoints when changing the recipe.

The full capture is at `outputs/patrol10_01` on the tested host. Full action frames and masks remain there. The local `media/virtualhome/patrol10_01` folder holds videos, checkpoints, QA, and reports; the complete evaluation export remains at `outputs/patrol10_01_evaluation` on the tested host.

Reproduce a new capture with:

```bash
python scripts/rollout.py \
  --executable .runtime/simulator-observer-v4/linux_exec.v2.3.0.x86_64 \
  --config configs/observer_long.yaml --xorg-root .runtime/xorg --gpu-index 3 \
  --output outputs/patrol10_new
python scripts/validate.py outputs/patrol10_new
python scripts/export_model_inputs.py outputs/patrol10_new \
  --output outputs/patrol10_new_evaluation
```

`event_schedule.json` contains ten flat transition records and all nine intervals. `events.json` keeps five paired records for compatibility, each referencing its own checkpoint images through `stage_keys`. Forty questions cover the six existing categories; history answers accumulate from one through five observed returns, and ten questions preserve uncertainty about offscreen changes. Debug frames display the current event number and highlight only visible target pixels.
