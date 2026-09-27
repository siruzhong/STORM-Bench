# Native room capture

The room pipeline targets each catalogued room instance separately. Rooms with
the same name retain different scene and room IDs. A planned room or a partial
video is never counted as an accepted sample.

The stages are:

1. `preflight_rooms.py` loads all seven native scenes and records instance-color
   collisions, room membership and candidate operations.
2. `probe_room_actions.py` tests approaches and reversible operations in a
   separate calibration session. Probe actors do not produce dataset events.
   Failed opening tests do not exclude independent switch or pickup tests.
3. `rollout_rooms_native.py` consumes those probes, freezes a seeded program,
   snapshots its source, and calls `rollout_native_room.py`. Each accepted
   capture is linked into `dataset/episode_*`; rejections retain their logs.
4. Each completed capture writes `qa.json` and `questions.jsonl` with ten
   unpolished, evidence-grounded questions, one per recorded event. The full
   natural-choice pool is saved as `qa_candidates.jsonl`; `qa_status.json` records
   the event-file hash, supported types, and pending polishing status. No model
   request is required. Missing question types are reported, never fabricated.
   The compatibility flag `--defer-qa` defers batch curation and prefix export,
   not these per-video annotations. `backfill_raw_qa.py` attaches the same draft
   artifacts to accepted captures from workers started before this change.
5. `rebuild_batch_qa.py` selects natural-choice questions from accepted episodes
   and exports their causal prefixes. Rendering success is separate from QA
   selection and semantic review.

The new room planner uses three to five distinct, natively verified targets.
Ten events may revisit a target, at most four events per target. Event order and
five offscreen intentions are sampled before recording, with dependencies and
occupied hands respected. It does not require every room to contain a television
or kitchen furniture. Room captures require at least four distinct native
operations; the legacy kitchen preset retains its six-operation requirement.
Room-level operation diversity and dataset-level event-family coverage must be
reported separately.

Native probes reserve slots for switch, open/close, and pickup/placement
families before filling the remaining target budget. The planner searches
three-to-five-target subsets jointly, preferring compact groups that still
provide at least two reversible operation pairs. Small duplicate props may use
stable room references such as a bathtub or washing machine when their distance
ordering is visually separated; ambiguous middle instances remain excluded.
Probe completion requires three uniquely describable targets, not only three
successful object instances. Repeated cabinets count only when stable room
geometry gives each selected instance an unambiguous spatial description.
Full-animation switch fallback is limited to short controls such as televisions,
computers, light switches, radios, and faucets. Long appliance cycles are not
rendered during calibration when their direct graph transition fails; the probe
continues to another target instead.

One or two operating characters cause events; the main character only moves and observes.
Operator spawn proposals use measured approach endpoints. Repeated object names
are admitted only when a unique reference object provides clearly separated
nearest/farthest descriptions. Shared segmentation colors remain disallowed.

New room profiles use a first-person camera at 1.6 m above the observer's
root. The observer's own mesh is hidden from capture; native locomotion and
other characters remain active. Camera position and yaw/pitch stay damped and
bounded. Each frame records the observer position beside the camera pose.
First-person validation requires a vertical offset of 1.35–1.8 m and at most
0.8 m of horizontal follow lag. Third-person captures retain their silhouette
and headroom checks. Camera collision and continuity checks apply to both modes.

The duration (60–70 seconds), cadence, offscreen, role, and motion checks still apply. An offscreen
manipulation must hide both its object and its operator throughout the action.
State confirmation requires a later visible observation and a checked graph
transition. No action frames are sped up or retimed to meet a duration limit.
The readiness decision uses the final encoded RGB/segmentation frame from the
preceding observer action when it is available. A separate live screenshot
cannot override what the accepted video actually shows.
The observer follows the frozen waypoint order. Runtime clearance checks may
skip a blocked waypoint, but they cannot invent a new destination. Consecutive
walks below 0.10 m may spend at most 0.5 seconds at one anchor. This lets one
short turn smooth the camera without creating a repeated furniture-edge loop.
Once a target has a verified visible frame, later events for that target try the
same observation anchor first. This avoids repeating a full room search for a
small switch or a partly occluded cabinet handle.
Visible evidence normally requires 40 target pixels at 640 by 480. A native
light switch uses a 12-pixel threshold because its stable rendered footprint is
smaller; the threshold is recorded in the episode configuration and reused by
capture and final audit.

Native probing can establish reachability and graph transitions, but does not
establish that a person can perceive the rendered state change. Accepted clips
still need semantic frame review before a formal benchmark release.

## Draft progress snapshots

`index_raw_dataset.py --ledger PATH --output PATH --watch-seconds 7200` publishes
immutable snapshots of accepted captures. Its `current` symlink points to the
latest snapshot, containing episode links, `manifest.json`, and
`raw_questions.jsonl`. It checks event hashes, question identifiers, natural
option domains, and evidence/query boundaries. Snapshots retain original video,
annotation, program, and runtime hashes. First-person and third-person counts
are kept separate. A full raw snapshot still requires semantic review and QA
polishing; it is not a formal benchmark release.

Run `audit_actor_separation.py CAPTURE --output REPORT.json` for supplementary
frame-level checks. Older runtimes provide animated hip recordings with explicit
clock gaps; the audit never fills those gaps with guessed poses. New runtimes
record all actor root positions with each RGB frame. Their capture gate requires
complete synchronized root evidence and the configured minimum separation for
every frame. Planar separation does not establish mesh-contact or swept-volume
collision freedom. NPC walking reservations follow native NavMesh corners so
observers avoid bends around furniture as well as straight crossing routes.

During concurrent manipulation, patrol scoring prioritizes a walking duration
that matches the NPC action. Geometric clearance and offscreen-path checks still
apply before scoring. This avoids choosing a short route solely because its
endpoint lies farther from the target, then standing still for the remaining
operation.

Visibility refresh keeps the original navigation stations and rechecks their
ray visibility when an operator or a held prop changes position. A fresh request
identifier prevents reuse of a response from an earlier world state. Small props
prefer closer observation stations; rotation of a held prop does not erase that
preference. Clearance, camera bounds, and rendered-pixel thresholds remain fixed.
Approach completion, discovery, and pre-event evidence use the final recorded
frame when one is available, matching the later event-readiness check. Live
screenshots cannot discard an observation already present in the video.

For captures that passed earlier checks, `replay_native_captures.py
--preserve-source` verifies the original source hashes and copies that snapshot
into the new attempt. The event program and implementation are preserved while
the new runtime records synchronized actor roots. The supervisor applies the
current separation audit even if the old implementation does not know that
report format. The replay still must pass the current capture gate. This mode
cannot be combined with viewpoint conversion.

## Optional scene enrichment

`enrich_rooms.py` creates a separate catalog with room-appropriate small props.
Existing object positions must remain within 5 cm of their original positions.
Placement requests retain their original IDs; the catalog records the native IDs
assigned by Unity using unique class-and-position matches. Every probe and capture
replays the placement requests and verifies that the same IDs are recovered.
Missing props, ambiguous matches and unplanned extra objects are rejected.
Prototypes that clone additional attached objects are removed during calibration,
and their rejected proposals remain in the placement ledger.

The camera runtime built by `build_camera_guard.py` includes a resource fallback
for scenes whose object prefabs are unavailable through `Resources.Load`.
Use that runtime for both enrichment and subsequent replay. X display leases
prevent parallel workers from attaching to another worker's GPU display.

The native planner prefers alternating operators where dependencies permit it.
An idle operator may approach its next target while the other performs an event.
This reduces travel between event slots without changing timestamps or playback
speed. The frozen timing windows and full-frame visibility checks still apply.
In sparse single-operator rooms, a target-to-target operator walk is paired with
the observer's visible approach. The observer leaves both the operator and
target out of frame only after that walk finishes, avoiding a second concealment
pass and a stationary camera gap. An unused helper is omitted in these rooms so
it cannot block a narrow patrol route.
Offscreen concealment uses one complete native walk to a frozen hidden anchor.
This prevents repeated partial walks from consuming an event window while the
camera still contains the changed object.

## Recovery and monitoring

Use `run_room_probe_queue.py` to isolate room probes in separate Unity sessions.
Each room has a configured attempt limit and a wall-clock timeout; a lost simulator
connection interrupts a trial instead of rejecting every remaining object.
Later attempts can skip a target that interrupted an earlier session while
retaining its infrastructure failure. Worker locks prevent duplicate queues.

Capture workers reject black setup frames before recording dataset events and
stop the batch on renderer initialization failure. They expose process IDs and
separate progress timestamps from status-write timestamps. Use
`rollout_status.py OUTPUT_ROOT` with process and native-log inspection; a status
file alone is not proof that a worker is alive or that a clip passed validation.

Native room plans reserve 5.5 seconds before the first event and distribute the
remaining nine gaps across a 54.5-second event span. Offscreen phases receive
more planning time for observer travel and camera rotation; all actual starts
must still satisfy the frozen absolute windows and interval bounds. Full-animation
probes run at the same 2.4 time scale as capture. This setting alone does not
establish that an episode fits the duration budget. Native action annotations,
encoded video duration, and the complete capture gates decide acceptance.
Operator spawn heights use the validated floor height rather than a navigation
endpoint slightly below the floor. Episodes receive distinct scene/room/seed
IDs before QA generation.

The detailed route, collision, concurrency, interpolation, and timing rules are
defined in `MOTION_CONTRACT.md`.

Validate a full pilot after changing a runtime, GPU, or worker configuration.
Nonblack smoke-test frames do not establish full event, camera, or semantic
acceptance.

`STORM_UNITY_JOB_WORKERS` controls Unity's per-instance CPU worker count and
defaults to eight. `STORM_FFMPEG_THREADS` controls video encoding threads and
defaults to four. Size these values across concurrent instances rather than per
process in isolation; the combined worker budget should leave CPU capacity for
Xorg, Python validation, and filesystem work.

Simulator workers lease X displays from 190 through 511 by default. Override
the half-open bounds with `STORM_XDISPLAY_MIN` and `STORM_XDISPLAY_STOP` when a
shared host reserves part of that range. Every display still uses an exclusive
file lock and rejects an existing X socket before Xorg starts.

Scale Unity sessions before raising the worker count of a single session. A
high-core host can run two capture sessions per GPU while room probes are still
active, provided GPU utilization and native RPC timeouts are monitored. CPU-only
plan generation can use a wider process pool because it does not create Unity,
Xorg, or encoder contention.

Use `scripts/probe_targets_parallel.py` when a few slow targets hold up a room
queue. It accepts the preflight catalog, existing probe reports through
`--probes`, and repeated `--room-index` arguments. Each target gets a separate
Unity instance and output directory. The queue interleaves rooms, prefers
untested targets, and retries previous infrastructure failures once. Semantic
failures are retained rather than repeated without a changed scene or policy.
Its merged `status.json` can be passed directly to `rollout_rooms_native.py`.

`--workers 24 --unity-workers 12 --gpu-indices 0 1 2 3` allows up to 24 target
sessions across four GPUs. `--reserve-memory-gib 128` pauses new admissions when
host available memory drops below 128 GiB; it does not allocate memory or impose
a per-process memory limit. The queue sets numerical-library threads to one;
Unity keeps its separate worker pool. Measure completed probes per minute,
native RPC timeouts, and accepted captures before raising concurrency further.
Pending work for a room is normally skipped once its merged native evidence
satisfies the planner's target and operation requirements. Use `--complete-pool`
to continue expanding the candidate pool, and `--target-classes` to prioritize
objects that are easier to inspect in the final view. Each task has a wall-clock
timeout and only its own simulator and Xorg processes are cleaned up.

## Native navigation and verified scene recipes

Pass `--require-animated-probes --native-navigation` to the room worker to use
full-animation evidence and the patched Unity NavMesh interface. The observer
uses complete native `Walk` actions to frozen destination coordinates. Short
`WalkTowards` actions do not reliably use that coordinate override. Each capture
writes requested and reached coordinates to `navigation_execution.jsonl` and
rejects endpoint errors above 0.40 m. Native animations remain active.

The navigation graph is compiled before recording. It includes static complete
paths and approximate visibility from candidate rear-overhead camera positions.
Physics rays can reject stations behind walls; runtime segmentation still decides
whether an observation is valid. Acquisition searches across several graph edges
instead of repeatedly choosing a locally closer point on the wrong side of a wall.
Paths reserve future operator spawn positions and simultaneous operator routes.
The runtime camera preserves its verified setup pose when recording begins.

`enrich_catalog_room.py` creates a new catalog for a targeted prop addition and
requires exact replay in an independent Unity process. `verify_room_catalog.py`
combines selected scene recipes and rebuilds every scene graph from native replay.
Both preserve the input catalogs. A rejected clone with extra attached objects is
not accepted as the requested prop.

Native character recorders have separate frame counters. The patched runtime
writes their shared Unity frame clock, and the recorder uses it to align action
annotations with observer video frames. This alignment is required for actors
that join later. An action that fails after generating frames invalidates the
attempt; those frames cannot be silently removed and followed by a retry.

Use `recovery_status.py --ledger /absolute/path/active_recovery.json` to inspect
owned job identities, latest room attempts, and accepted videos. Final QA counts
must be checked in the merged release rather than inferred from the 60 candidate
questions generated inside each accepted capture.


Native command scheduling separates the measured preparation time from the
manipulation itself. The preparation estimate advances the preferred command time, but the command
never precedes the earliest legal event time: measured preparation may already
have been completed by a prior native walk. Commands use the early part of the
legal window to leave room for variable native preparation. Timing patrols score
short edges against the remaining wait, avoiding a long final walk that consumes
the NPC's preparation window. Actual native action timestamps remain authoritative.

For third-person diagnostics, the navigation probe reports camera-feasible
stations and tests candidate routes against occlusion from the current camera.
These scores guide route selection; they do not replace per-frame image audits.
First-person capture skips that observer-visibility probe because the observer
is intentionally outside the camera image.
