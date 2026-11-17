# Observer pipeline

## Mixed routine (default)

1. Load scene 0 and add an observer plus two operating characters. Register the observer camera once.
2. Sample event proportions, eligible objects, dependency-respecting order, and start-time windows. Validate and save the complete ten-event program before action rendering. A supplied `--plan` is validated and replayed without resampling.
3. Position the characters using native walks before the episode. The second man starts near the television; the woman starts at the kitchen counter.
4. Open the episode with the television switch-on. The man steps back to watch while the observer moves toward the kitchen.
5. Execute kitchen tasks alongside observer walking legs. After a grab, let the worker step back from the counter so the held item can be inspected. Record previous and subsequent views of each target.
6. For planned offscreen events, walk away before the manipulation, continue moving during it, and return to inspect the changed object. Require zero target and acting-character pixels throughout the manipulation.
7. While the woman returns from the dining table, let the second man switch off the television. Finish the kitchen tasks, then inspect the television again. Continue walking toward the 66-second target.
8. Check ten real transitions, two operating characters, five objects, six operations, 60–70 seconds, configured event spacing, observer motion, headroom, camera collisions, continuous camera poses, and agreement with the frozen program.
9. Generate six questions per event. Export exact-frame prefixes and keep private graphs, answers, and debug videos out of model inputs.

See [MULTI_AGENT.md](MULTI_AGENT.md) for candidate limits and the tested run. `rollout.py` selects this path by default; an explicit `configs/observer.yaml` selects the earlier bowl-only protocol below.

## Bowl-only observer baseline

1. Start a private Xorg server on the selected GPU and launch Unity with OpenGL.
2. Load the apartment, register the rear-overhead camera, and add the observer and injector. Resolve the observer camera index once and use it for every action.
3. Prepare the scene through recorded agent actions outside the episode: open the upper cabinet, let the injector pick up the bowl, and position the observer. Preparation frames remain in `setup/` and are not part of model inputs.
4. Begin the episode by turning toward the injector. Confirm a unique target instance color and at least 40 visible pixels.
5. Walk the observer away. Pair the injector's `PutIn` with an observer walking leg. Require zero target and injector pixels throughout this manipulation.
6. Walk back and turn toward the injector. Confirm the bowl is out of view. The cabinet remains open, but the stored bowl must still have zero visible pixels.
7. Pair the injector's `Grab` with an observer walk, then turn toward the injector to see the bowl in the other person's hand again.
8. Repeat steps 5–7 for five cycles without resetting the scene. Ten event starts must be spaced 5.0 ± 1.5 seconds apart. If needed, continue walking after the final event to reach 60 seconds. Reject videos outside 60–70 seconds.
9. Audit every cycle's visibility, actor roles, camera collisions, event spacing, duration, and observer motion. Generate eight questions per cycle across the six existing categories after the audit passes.
10. Export clean prefix videos and model inputs separately from answers and private scene evidence. Optionally polish question wording through a user-supplied VLM API.

## Shared camera and movement

The mixed preset uses a 28-degree downward pitch; the earlier bowl-only preset uses 30 degrees. The bowl-only short preset uses camera offset `(0.55, 2.4, -2.4)`, Euler rotation `(30, 0, 0)`, and a 75-degree field of view. The runtime retracts the camera before scene colliders. Debug highlights use visible instance pixels and do not reveal hidden geometry.

Observer walks and injector actions share script lines. Checkpoint holds last 0.05 seconds. Skeleton positions measure translation over a 0.5-second window at a 0.15 m/s threshold; hip orientation measures turning at 30 degrees/s. The short preset requires at least 80% active frames and no inactive interval above two seconds. Reports retain raw translation statistics separately. The older long and stationary presets remain available.

## Bowl-only evidence and QA

The offscreen interval must contain zero pixels for the target, injector, and original support. Graph snapshots separately establish that the target was stored inside the cabinet and later held by the injector. The absence inspection must show the injector without the bowl. Hidden graph facts never answer observer-facing questions. Two questions per cycle retain uncertainty about the object's location and the removal method.

Times come from written video frames. Action intervals bound the recorded sequence; they do not claim the exact physical transition frame. Observation checkpoints and first reobservation times are stored separately. History counts accumulate across all five cycles.

The VLM adapter sends only frames preceding each question's prefix boundary, together with its text and choices. It cannot edit choices, answers, evidence spans, or event identities. Rewrites preserve the original text for review.

## Runtime compatibility

The client omits the obsolete `clear` command and allows the HTTP listener time to reopen between requests. It does not retry mutations automatically. A zero-distance walk can produce no animation frames; the recorder then writes the current frame. Tail walks reject zero-progress actions.

VirtualHome 2.3's `save_scene_states` path raised a `SceneStateSequence.SetFrameNum` exception on the tested host. It remains disabled. Graph snapshots use `environment_graph`; skeleton recordings use `save_pose_data`.

The original single-character implementation is retained as `rollout_self_interaction.py`. Its event selection and six known-answer questions per pair are a separate baseline. The compact recipe is a fixed scene configuration, not a general navigation planner.
