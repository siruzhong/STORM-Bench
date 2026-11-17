# Multi-character routine

The current damped-camera controller and earlier accepted two-offscreen-event `randomized_09` capture are described in [CAMERA_RULES.md](CAMERA_RULES.md). Historical results at the end of this document predate those checks.

The observer walks through the apartment while two other characters carry out a short household routine. The woman handles kitchen objects. The second man controls the television. The observer never grabs, places, opens, closes, or switches an object.

The planner uses a seed to choose the cabinet, the order of the two portable objects, and an eligible event at each step. Independent opening events may occur before or after a completed pickup/placement pair. The complete event program is frozen before action rendering; see [PLANNING.md](PLANNING.md). Dependencies keep opening before closing and picking up before putting down. One kitchen object is handled at a time. The final portable object is carried to the dining table. If a second portable object is selected, its earlier placement uses the counter, so placements do not compete for one native slot. The television stays on through the placements. Its operator switches it off while the woman walks back from the table; the observer checks the television again after the kitchen tasks. Actions run through VirtualHome's character scripts; scene graph edits do not cause events.

The scene-0 candidate pool contains a television, microwave, two upper cabinets, a bowl, and dishwashing liquid. Each completed episode selects five objects and records ten state or support transitions: either three open/close pairs and one pickup/placement pair, or two of each, plus one television on/off pair. Cabinet selection and task ordering are randomized; this is not an arbitrary-scene activity planner. The event families are `state_change` and `movement`. These events do not claim to reproduce spawning, deletion, or identity replacement.

An event stores its acting character, object, action interval, previous visible inspection, and subsequent inspection. Inspections can be delayed: the final television check refers back to the earlier visible on-state, while its switch-off overlaps the other character’s return walk. Every counted manipulation must succeed in the simulator and change the scene graph. Its target needs a unique instance color and at least 40 visible pixels in both inspections. The default plan requires five full manipulation intervals with zero target and acting-character pixels. One additional unintentionally offscreen event is allowed; a larger deviation is rejected. The audit classifies other intervals from recorded masks instead of assuming their visibility from the plan.

Six questions per event cover the same QA categories as the AI2-THOR pipeline. An offscreen current-state question remains uncertain until the observer sees the object again. Graph states are private verification evidence. They are not used to answer what an observer could not see, and they are not sent to the VLM wording endpoint.

The debug video highlights only visible pixels of the current target. Its action descriptions and event labels are private debugging information, not model inputs. Use `export_model_inputs.py` to generate clean, exact-frame prefixes and separate answer labels.

## Run

Use the runtime and environment setup from the project README, then run:

```bash
python scripts/rollout_mixed.py \
  --executable .runtime/simulator-observer/linux_exec.v2.3.0.x86_64 \
  --config configs/randomized.yaml --seed 42 \
  --output outputs/mixed_001 --xorg-root .runtime/xorg --gpu-index 0
python scripts/validate.py outputs/mixed_001
python scripts/export_model_inputs.py outputs/mixed_001 --output outputs/mixed_001_eval
```

The observer must be moving or turning in at least 80% of recorded frames, with no inactive interval above two seconds. Event starts must be 6.0 ± 2.5 seconds apart, and the complete video must last 60–70 seconds.

Both `--output` directories must be new. A seed fixes task selection, but Unity navigation and animation can introduce small timing differences. The completed capture must still pass duration, spacing, visibility, motion, framing, and collision checks. Failed captures remain available for inspection and must not enter a dataset.

## Historical capture before continuity checks

`outputs/multi_16` was recorded on Linux with seed 42 and the v9 camera runtime. It contains 1,342 frames at 20 fps (67.1 seconds), ten events, five objects, and sixty questions. Event starts are 4.2–7.2 seconds apart. The two planned offscreen intervals contain no target or acting-character pixels. All frames retain the observer, with a minimum upper margin of 141 pixels. All 3,467 camera telemetry samples have zero overlaps.

The observer translates in 68.2% of frames and translates or turns in 80.1%. Its longest interval without either activity is 1.75 seconds. These are separate measurements; turning is not counted as walking. The simulator records normal-speed frames without retiming or still-frame padding.

| Event | Operator | Action | Object | Start (s) | Inspected after (s) |
| --- | --- | --- | --- | ---: | ---: |
| 1 | char2 | SwitchOn | tv | 0.65 | 3.85 |
| 2 | char1 | Open | microwave | 6.95 | 11.70 |
| 3 | char1 | Grab | dishbowl | 13.90 | 21.05 |
| 4 | char1 | PutBack | dishbowl | 21.10 | 27.00 |
| 5 | char1 | Open | kitchencabinet | 27.05 | 31.20 |
| 6 | char1 | Grab | dishwashingliquid | 31.25 | 35.35 |
| 7 | char1 | PutBack | dishwashingliquid | 37.45 | 48.05 |
| 8 | char2 | SwitchOff | tv | 42.10 | 60.00 |
| 9 | char1 | Close | kitchencabinet | 49.10 | 53.25 |
| 10 | char1 | Close | microwave | 53.30 | 57.60 |

The runtime source hash is stored in `runtime_manifest.json`. `validation.json` and `debug_validation.json` contain the full checks. The default seed was validated in the simulator; tests also check dependency ordering and candidate diversity for 100 seeds. Other seeds still require per-capture validation because native navigation can change visibility and timing.
