# STORM VHome Pipeline


This directory contains rollout and QA construction code. Install from this
checkout with `pip install -e '.[qa,test]'`. Keep captures, reviewed evidence,
generated QA, and evaluation outputs outside the repository.

The current 29-episode / 290-question QA workflow is provided by the
`storm-virtualhome-qa` command. See [QA_GENERATION.md](docs/QA_GENERATION.md)
for installation and complete generation commands, and [INPUTS.md](docs/INPUTS.md)
for the external evidence contract. It includes collection, mask auditing,
selection, deterministic wording revision, and evaluator export. The
scene-specific and older QA recipes below remain available as separate tools.

Repository layout follows the construction-code pattern used by `storm-real`:
`src/storm_virtualhome/` contains simulator and planning code,
`src/storm_virtualhome_qa/` contains the current QA pipeline, `tests/` contains
offline checks, and `scripts/` contains capture and maintenance entry points.
`configs/` and `unity_debug/` contain the scene recipes and camera patch source.

For the current multi-room pipeline, see [ROOM_DATASET.md](docs/ROOM_DATASET.md).
New native room profiles use first-person capture with a moving observer and
separate operating characters. Each accepted video includes ten raw QA records;
wording polish and batch curation follow capture. The scene-specific presets
below retain their third-person camera recipes.

Three characters have separate roles. The blue-shirted observer walks and looks around. A woman handles kitchen objects while another man switches on the television, steps back to watch it, and switches it off while the woman returns from the table. All changes come from native character actions. The observer never manipulates objects.

The default randomized preset records ten events involving five objects. It samples either three open/close pairs and one pickup/placement pair, or two of each, plus a television on/off pair. A seed selects the portable objects, cabinets, eligible event order, and timing jitter. Hand occupancy and state dependencies keep the program executable. Five events are planned outside the observer view: the first pickup, final placement, and three state changes. The final portable object goes to the dining table; earlier placements use the counter.

The supported event families are `state_change` and `movement`, with native opening, closing, grabbing, placing, and switching actions. These do not cover every AI2-THOR event family. `configs/mixed.yaml` retains fixed proportions with randomized eligible ordering for camera regression work.

Each capture must last 60–70 seconds, with starts spaced approximately six seconds apart. The planner reserves seven seconds for a hidden pickup-to-placement transition. Five manipulations are planned entirely outside the observer camera and discovered after returning. Full-interval checks reject partially visible manipulations; the measured total must be five or six. Captures save ten raw questions and a larger natural-choice candidate pool. The generator supports the six STORM categories and reports which are supported by the recorded events. Unseen current states remain uncertain until reobservation. See [MULTI_AGENT.md](docs/MULTI_AGENT.md) for the routine, validation rules, and capture results.

The validated damped-camera sample `offscreen5_smooth_transfer_v2_attempt_02` lasts 66.45 seconds, with ten events, five fully offscreen manipulations, 60 QA records, and no camera overlaps. Event gaps are 4.9–7.9 seconds; the longest inactive observer interval is 1.25 seconds. See [CAMERA_RULES.md](docs/CAMERA_RULES.md) for measured continuity, movement, and remaining visual limitations.

## Run

Use Linux, Python 3.10+, and the separately distributed [VirtualHome 2.3 simulator](http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.zip). GPU rendering needs the NVIDIA driver and Xorg.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . pytest
python scripts/download_simulator.py
bash scripts/setup_xorg.sh
bash scripts/setup_mono.sh
python scripts/build_camera_guard.py --output .runtime/simulator-observer
python scripts/rollout.py \
  --executable .runtime/simulator-observer/linux_exec.v2.3.0.x86_64 \
  --config configs/randomized.yaml \
  --output outputs/mixed_001 --xorg-root .runtime/xorg --gpu-index 0
python scripts/validate.py outputs/mixed_001
python scripts/export_model_inputs.py outputs/mixed_001 --output outputs/mixed_001_eval
pytest
```

Native navigation and animation duration can vary between attempts. For bounded retries of one frozen program, use:

```bash
python scripts/rollout_retry.py --executable .runtime/simulator-observer/linux_exec.v2.3.0.x86_64 \
  --config configs/randomized.yaml --plan plans/seed42.json \
  --output outputs/retry_001 --max-attempts 5 --xorg-root .runtime/xorg --gpu-index 0
```

The runner snapshots its inputs, retains each attempt and log, and records the accepted directory in `attempts.json`. It does not resample events, relax checks, or retry indefinitely. Exhausting the limit exits with an error and no accepted episode.

Every output directory must be new. The setup scripts extract runtime dependencies locally without sudo. `--display :N` uses an existing display. Otherwise `--xorg-root` starts a private GPU Xorg server; omitting both starts the slower Xvfb fallback. Only processes started by this run are stopped.

The default configuration uses scene 0, fixed character spawn positions, and explicit object and navigation IDs. It is a fixed scene recipe, not a planner for arbitrary rooms. Configure these IDs and validate a new episode when changing scenes. Entirely black RGB frames, palette collisions, blocked paths, camera overlap, or failed visibility checks stop the run and leave its partial output for inspection.

For other rooms and layouts, use the graph catalog, room-profile planner and native validation runner described in [MULTISCENE.md](docs/MULTISCENE.md). Generated profiles remain unvalidated until their captures pass the full audits; capability gaps and rejected rooms are recorded explicitly.

The bowl-only observer preset remains available through `scripts/rollout.py --config configs/observer.yaml`. Its accepted 60-second capture is described in [SHORT_RUN.md](docs/SHORT_RUN.md). The previous self-interaction baseline remains available as `scripts/rollout_self_interaction.py`, using `configs/rollout.yaml` or `configs/debug.yaml`. It is not the default observer protocol.

## Batch capture

Use `scripts/rollout_batch.py` to freeze a candidate pool, resume bounded attempts, and publish accepted videos with ten selected QA records each. A 30-video batch contains 300 questions, with 50 per category. See [BATCH.md](docs/BATCH.md) for commands, acceptance rules, and output layout.

## Plan before rendering

Use `scripts/plan_episode.py` with a saved `planning_graph.json` to create a standalone program, then pass it to `scripts/rollout.py --plan`. For a new scene, `rollout.py --plan-only` saves the graph and program without recording action frames. A regular run without `--plan` still freezes the complete event program before setup action rendering. The renderer never resamples events. See [PLANNING.md](docs/PLANNING.md) for commands, randomization scope, and replay checks.

## Camera, motion, and evidence

`char0` only walks and turns. `char1` handles kitchen objects, and `char2` controls the television. Scene-specific presets use the observer's rear-overhead camera; native room profiles use first-person capture. The recorder retains only stream 0, including during actions performed by the other characters.

Experimental observer walk pacing is disabled in default builds. Use `build_camera_guard.py --observer-walk-pacing` for isolated trials; initial trials did not pass the full capture audit. Such runtimes declare `observer_joint_walk_pacing` and can slow the observer's native walking animation during a joint action. The requested pace uses the verified route length and planned operation duration, with a scale between 0.45 and 1.0. The coroutine restores the original animation speed when walking ends. Other characters, the simulation clock, event windows, and encoded frames retain their normal timing. Each capture must still pass the motion, visibility, clearance, and event timing audits.

The camera retains its world pose between actions, follows a stable observation region, and moves around obstacles continuously. It uses a fixed 75-degree field of view. A damped position spring and interpolated heading soften movement and turns. The randomized preset caps translation at 3.5 m/s and rotation at 110 degrees/s, including action boundaries and snapshots. Walls, people, and ceiling-lamp keepout volumes constrain its path. Every action frame records matching RGB and segmentation camera poses. See [CAMERA_RULES.md](docs/CAMERA_RULES.md) for the rules and acceptance checks.

Visible instance masks create yellow highlights in the debug video; no hidden geometry is drawn. A separate header sits above the camera image. The audit also checks collision logs and observer headroom. See [DEBUG.md](docs/DEBUG.md) for the runtime patch. The previous `multi_16` clip passed the older checks but has visible camera jumps; it is not an accepted example of the new continuity rules.

Every event must correspond to one successful native manipulation and a verified graph transition. Both inspection checkpoints must show at least 40 target pixels with a unique instance color. Offscreen intervals require zero target and acting-character pixels throughout. Graphs verify what happened but do not supply hidden facts to observer-facing answers.

Observer movement runs concurrently with manipulations. Checkpoint holds last one frame at 20 fps. Skeleton recordings measure translation and turning separately. Acceptance requires at least 80% active frames and no inactive interval longer than two seconds. The report also retains the raw walking fraction. The closing route continues walking toward a 66-second target after the final inspection; no video retiming or static padding is used.

The mixed recipe has ten flat event records. The older bowl-only recipe retains five paired records and its separate `presence_change` protocol. [PIPELINE.md](docs/PIPELINE.md) describes both paths.

## Outputs

- `video.mp4`: clean observer-camera video.
- `debug.mp4`: matching frames with visible target highlights, actions, and timestamps.
- `event_program.json`, `planning_graph.json`: immutable planned events and their source scene.
- `plan_execution.json`: planned versus actual event timing and replay acceptance.
- `events.json`: private action intervals, acting characters, object identities, and observation times.
- `event_schedule.json`: numbered transitions, event count, and actual injection gaps.
- `stages.json`, `evidence/`: checkpoint times, images, masks, and private scene graphs.
- `qa.json`, `questions.jsonl`: questions, choices, private answers, uncertainty labels, and prefix-bounded evidence.
- `actions.json`, `actions/`, `frame_manifest.jsonl`: action records and synchronized source frames.
- `visibility.jsonl`, `observer_validation.json`: per-frame visibility and role/geometry checks.
- `motion_validation.json`, `motion_trace.jsonl`: measured travel and pause intervals.
- `camera_continuity.json`: exact exported-frame pose and lens checks.
- `debug_validation.json`, `logs/camera_collision.csv`: highlight and camera audits.

`export_model_inputs.py` creates exact-frame prefix videos and `model_inputs.jsonl`, with answers in a separate `evaluation_labels.jsonl`. Give answering models only the clean prefix video, question, and choices. Debug overlays, event logs, graphs, answers, and future frames are private evaluation data.

## Optional VLM wording pass

An optional [Jev review adapter](docs/JEV.md) checks rewrite meaning and reviews
planning or failure records. Full video capture takes priority; polishing is a
separate later pass.

Supply an endpoint and model. The adapter expects a multimodal chat-completions request with a JSON question in the response; change `make_request` and response parsing for a different API.

The script loads [the wording prompt](src/storm_virtualhome/prompts/qa_polish.txt) from the package. It asks for clear prose while preserving the original object references, time interval, observation unit, and distinction between visible and unresolved information. When the frames do not support a proposed clarification, the original question is retained. This API pass accepts raw episode `qa.json`; the `storm-virtualhome-qa polish` command instead applies deterministic wording rules to an exported release and makes no model request.

```bash
export VLM_API_KEY='your-key'
python scripts/polish_qa.py \
  --input outputs/mixed_001/qa.json --output outputs/mixed_001_polished \
  --endpoint https://your-provider.example/v1/chat/completions --model your-vlm
```

Only question wording can change. Choices, answers, and uncertainty labels are retained. Requests contain prefix frames, the question, and choices, without private evidence. `--dry-run` writes requests without network calls. Review rewrites for semantic changes before evaluation.

The simulator protocol follows the [official VirtualHome client](https://github.com/xavierpuigf/virtualhome/blob/master/virtualhome/simulation/unity_simulator/comm_unity.py). This project has no AI2-THOR runtime dependency.

## Natural-choice QA

For a four-choice reference distribution, use the [evidence-based alignment workflow](docs/ALIGNED_QA.md). This rebuilds question semantics and annotations after capture; it is separate from the wording-only API pass.

Use [the natural-choice QA workflow](docs/NATURAL_QA.md) to rebuild captured batches with two, three, or four valid options and an explicit per-question chance baseline. New batch runs expose this development candidate through `recommended_qa`; the legacy four-choice QA remains available for reproducing earlier results.
