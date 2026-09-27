# Plan events before rendering

`plan_episode.py` reads a saved scene graph and creates an immutable event program. It does not launch Unity or render frames. The program records a seed, ten ordered events, target IDs, native operations, operating characters, destinations, dependencies, visibility intentions, and start-time windows. The renderer accepts it with `--plan` and writes a copy to the output before rendering any setup actions.

```bash
python scripts/plan_episode.py \
  --graph outputs/previous_capture/planning_graph.json \
  --config configs/randomized.yaml --seed 42 --output plans/seed42.json
python scripts/rollout.py \
  --executable .runtime/simulator-observer/linux_exec.v2.3.0.x86_64 \
  --config configs/randomized.yaml --plan plans/seed42.json \
  --output outputs/seed42 --xorg-root .runtime/xorg --gpu-index 0
```

For a new scene, use `rollout.py --plan-only` with the normal runtime arguments to save a graph and program without recording action frames. A regular scene capture can also omit `--plan`: after scene initialization it saves `planning_graph.json`, creates and validates `event_program.json`, then starts action rendering. Later plans can use that saved graph. Output files must not already exist. A plan made with a different seed carries that seed into the renderer; an explicit conflicting `--seed` is rejected by configuration validation.

## Random sampling and constraints

The default scene recipe samples portable objects and openable objects from configured candidate pools. It chooses an eligible next event using the seed instead of following a fixed kitchen-event chain. Open events can exchange positions with complete pickup/placement pairs. A pickup requires a free hand; placement requires the same actor to hold the object. Opening and switching require the corresponding initial state. Closing follows opening. Independent closing events can exchange order. All ten events are selected before rendering.

Both presets retain two operating roles and a television on/off pair. `mixed.yaml` keeps two portable-object round trips and two open/close pairs for camera regression checks. `randomized.yaml` also samples the event proportions: either one portable-object pair and three open/close pairs, or two of each. Both choices yield ten events involving five objects. This is constrained scene sampling, not unconstrained sampling of arbitrary activities. The current candidate pool remains small. The default `offscreen_event_count: 5` freezes five offscreen intentions: the first pickup, final placement, and three state changes. Closing tasks are preferred because they follow visible open-state inspections; if the mixture has only two closing tasks, the last opening task supplies the fifth offscreen event. Objects and eligible ordering remain seeded. Every planned offscreen action must pass full-interval pixel checks, and the measured total must be five or six. The legacy mixed preset retains two offscreen intentions. The observer never operates an object. When two cabinets are selected, questions use their positions relative to the sink and nearby appliances. Same-class targets without distinct visible descriptions are rejected during planning.

Target start times are spaced six seconds apart with independent jitter of at most 0.3 seconds, then aligned to the configured video-frame grid. Each has a 2.5-second execution tolerance. The next preferred execution time follows the preceding actual start plus six seconds, clamped to the next immutable window. Native animation and navigation determine actual times; clips that miss their windows are rejected rather than retimed. Timing constraints are checked together with the 60–70 second duration, actual event gaps, visibility, observer movement, and camera continuity.

Native room plans sample their five offscreen events before rendering and spread them across the episode when dependencies allow it. The sampler avoids ending with an offscreen event and avoids consecutive hidden changes to the same object when another seeded allocation satisfies the same quota. It prefers completed pairs, such as closing a cabinet after a visible opening, because the restored state is quicker to rediscover. This reserves a later viewpoint change for discovering every hidden event within the duration limit.

In a two-operator room, the initially present operator performs up to the first four events. The delayed helper then enters outside the recorded view and completes both operations on its first object before later events resume. This prevents a new character from appearing in view and avoids postponing the helper's entry until the final event windows.

Native room events use a 4.75-second start-window tolerance for discrete Unity navigation steps. The independent 60–70 second duration gate remains unchanged, so this tolerance accepts local step quantization without allowing the whole episode to drift beyond the video contract.

## Replay checks

A content hash detects program modification. A configuration signature binds the renderer settings. A scene signature covers object identities, capabilities, initial states, and support/container relations; it excludes characters and transient animation transforms. This is a symbolic precondition check, not proof that navigation, visibility, or timing will succeed.

Rendering must preserve the planned task, operation, object, destination, actor, and visibility intention. Native failures and failed evidence checks reject the capture. Runtime checks may add observation movements, and operating characters may walk closer to a target before its scheduled manipulation. These preparation walks do not count as events. They may not replace planned events or silently resample targets. For an offscreen close, the preceding recorded open-state inspection supplies the before-state evidence; the observer does not revisit the same unchanged state immediately before every manipulation. Afterward, a new visible inspection must confirm the closed state. Camera turns finish before triggering an offscreen action, and the complete action interval is still checked frame by frame. The original program remains unchanged; actual timings belong to `events.json`.

## Relationship to AI2-THOR

The source AI2-THOR benchmark separates event planning from replay and seeds its event decisions. This implementation follows that separation. Its family names `movement` and `state_change` and action names `move`, `open`, `close`, `toggle_on`, and `toggle_off` match STORM terminology. Its six QA categories remain unchanged.

The AI2-THOR taxonomy also includes presence, occlusion, identity, quantity, and no-change families. They are not all implemented by this native-agent VirtualHome recipe. In particular, taking an object out of view is not relabeled as simulator object deletion, and moving an object is not an identity swap. Adding a family requires a native action plan plus visual and graph-effect checks before enabling it for sampling.

Related offscreen closures can share a later kitchen visit. Their native actions retain separate planned windows, complete hidden intervals, and individual before/after evidence. Until that return visit, their current-state questions remain uncertain. The observer does not reverse the camera after every closure merely to record a checkpoint.

The default schedule reserves seven seconds from a pickup to its matching placement when either event is offscreen. Other nominal gaps remain six seconds. This extra observation-and-departure time is included in the frozen program before rendering; it does not expand replay tolerances or the 60–70 second duration limit.
