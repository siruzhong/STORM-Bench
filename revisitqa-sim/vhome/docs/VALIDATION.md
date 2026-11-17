# Validation

Validated on 2026-09-18 with VirtualHome 2.3.0, Python 3.11.15, an NVIDIA H20, driver 535.161.08, and a private Xorg server. Unity used OpenGL. The tested dependency versions are in `requirements-tested.txt`.

The current 60-second, ten-event capture is documented in [SHORT_RUN.md](SHORT_RUN.md). The measurements below describe earlier single-character baselines. See [OBSERVER_RUN.md](OBSERVER_RUN.md) for the two-character stationary capture and [PATROL_RUN.md](PATROL_RUN.md) for the moving observer.

## Completed baseline episode

`outputs/pilot_04` uses apartment 0 and its kitchen. The camera follows the acting character from behind and above.

| Item | Result |
| --- | --- |
| Video | 640 × 480, 20 fps |
| Frames / duration | 1,140 / 57.0 seconds |
| Accepted event pairs | 2 |
| Events | 2 disappearances, 2 reappearances |
| QA | 12; 2 in each of the 6 categories |
| First target | Bell pepper: 73 → 0 → 179 visible pixels |
| Second target | Bowl: 300 → 0 → 739 visible pixels |

The validator reread the MP4, instance masks, saved color palettes, graph relations, event times, and QA evidence spans. Both targets had unambiguous mask colors. The four exported video prefixes contain exactly 345, 610, 875, and 1,140 frames, ending at 17.25, 30.50, 43.75, and 57.00 seconds. Model input records contain no answer indices, evidence text, or scene graphs.

The QA file was regenerated from the accepted event records after wording and category balancing were finalized. The simulator recording and event labels were unchanged.

## Tests

Seven tests passed. They cover all six QA categories, deterministic choice ordering, prefix bounds, containment states, instance-color matching and collisions, palette refresh order, and a local HTTP integration test for the VLM adapter. That integration test confirmed that later blue frames were not sent for a prefix containing only red frames and that private labels were preserved. No external VLM API was called.

## Scope

The demonstrated route is the kitchen in apartment 0. Other apartment layouts and rooms have not been validated. That legacy self-interaction configuration requests two pairs. A five-pair trial completed three valid pairs, then stopped because a returned plate was not visible from the follow camera. That trial was not exported as a completed dataset.

Simulator instance colors can vary between runs and can collide, so a fixed configuration does not guarantee the same accepted object list or successful count. The pipeline records the actual scene, targets, masks, and rejected candidates and fails if its requested count cannot be verified. For larger collections, retain successful complete episodes and inspect failures before widening the selection rules.

Vulkan with Xvfb was not a working capture path in this environment. GPU OpenGL with a private Xorg server was verified. Xvfb with software OpenGL produced frames but was substantially slower.

## Camera and highlight update

The updated runtime was tested with episode `outputs/debug_04`: 547 frames at 20 fps (27.35 seconds), one bowl put-away-and-return cycle, and six QA records. The clean RGB and debug videos have the same frame count. Yellow target highlighting appears in 383 frames; occluded target geometry is not drawn.

The Unity camera audit contains 1,396 render samples. It shortened the requested offset in 426 samples and reported zero camera-sphere overlaps. A separate test requested an eight-metre rear offset: all 81 samples retracted before an obstruction, with zero overlaps. These measurements cover the tested kitchen route and the simulator's collision geometry.

Eleven tests passed after this update, including a regression that rejects a single camera overlap and a check that an invisible target leaves image pixels below the debug header unchanged. The separate camera-runtime build also completed successfully; the original simulator assembly hash matches the build manifest.
