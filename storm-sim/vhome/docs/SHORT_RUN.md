# Short observer rollout

`framing_04` was recorded on Linux on 2026-09-18. It contains 1,200 frames at 20 fps: **60.00 seconds**, without retiming or duplicated video segments. Five disappearance-and-reappearance cycles produce ten events. The blue observer only walks and turns; the green injector performs all object manipulation.

## Camera framing

The camera offset is `(0.55, 2.4, -2.4)`, with a 30-degree downward pitch and a 75-degree vertical field of view. Compared with the preceding capture, the requested camera position is 0.7 metres farther behind the observer and the pitch is reduced from 40 to 30 degrees. Collision checks can shorten that offset in confined areas.

The observer silhouette was present in all 1,200 frames and never touched the image's upper edge. Its smallest top margin was 13 pixels; that frame was also inspected visually. This boundary measurement checks cropping, not scene occlusion. The debug header occupies a separate 64-pixel band above the image, preserving the full camera view. Clean video is 640 × 480; debug video is 640 × 544.

The injector starts with the bowl in hand, puts it into an open upper kitchen cabinet outside the camera view, and retrieves it after the observer returns to inspect the absence. Preparation actions occur before the episode. The final event is observed by 51.15 seconds; the observer then continues walking until the video reaches the duration requirement. The event route and target are unchanged from `short10_12`.

| Event | Type | Sequence start (s) | Gap from previous start (s) |
| --- | --- | ---: | ---: |
| 1 | disappear | 3.95 | — |
| 2 | appear | 9.65 | 5.70 |
| 3 | disappear | 13.85 | 4.20 |
| 4 | appear | 19.60 | 5.75 |
| 5 | disappear | 23.90 | 4.30 |
| 6 | appear | 29.40 | 5.50 |
| 7 | disappear | 33.80 | 4.40 |
| 8 | appear | 39.20 | 5.40 |
| 9 | disappear | 43.50 | 4.30 |
| 10 | appear | 49.00 | 5.50 |

The nine start gaps average **5.01 seconds**, ranging from **4.20 to 5.75 seconds**. The default acceptance range is 5.0 ± 1.5 seconds. These are starts of recorded manipulation sequences, not exact physical transition frames. Videos outside 60–70 seconds are rejected.

## Validation

- All five cycles passed visibility and scene-state checks. The bowl was visible before each removal and after each retrieval, and absent at each return inspection.
- All 150 offscreen-injection frames contained zero target, injector, original-counter, and container pixels.
- The observer performed zero object manipulations.
- All 3,453 camera samples required some retraction; none reported a collider overlap.
- Translation occupied 77.75% of frames. Translation or active turning occupied 87.83%. Total inactive time was 7.30 seconds; the longest inactive interval was 1.10 seconds. Turning is reported separately from walking.
- Clean and debug videos both contain 1,200 frames. Yellow highlighting appears in 425 frames and uses only visible target pixels.
- Forty questions cover all six existing categories. Ten retain uncertainty about offscreen changes. Evidence spans end within their question's video prefix; cumulative history counts run from one through five.
- All 27 tests passed, including duration bounds, event counts and spacing, role restrictions, later-cycle visibility, cumulative QA, camera overlap rejection, black RGB frame rejection, and preservation of the camera image below the debug header.

The full capture, source frames, masks, setup recordings, and runtime logs remain at `outputs/framing_04` on the tested host. The local `media/virtualhome/framing_04` directory contains the clean and debug videos, QA, checkpoints, configuration, and validation reports. The evaluation export contains 15 clean prefix videos and 40 model-input records, with answer labels stored separately. The preceding 60.05-second capture remains at `outputs/short10_12` and its local media directory for comparison.

## Reproduce

```bash
python scripts/rollout.py \
  --executable .runtime/simulator-observer/linux_exec.v2.3.0.x86_64 \
  --config configs/observer.yaml --xorg-root .runtime/xorg --gpu-index 0 \
  --output outputs/observer_short
python scripts/export_model_inputs.py outputs/observer_short \
  --output outputs/observer_short_evaluation
```

Select a GPU with enough available resources. The tested host used GPU 3. The rollout performs the full audit and debug rendering before successful exit. This is a fixed apartment-0 recipe; changing the scene or camera requires a new accepted capture. Failed trials are retained with rejection notes and are not evaluation episodes.
