# Room-specific rollout validation

The existing `randomized.yaml` is a validated kitchen recipe for scene 0. New
room profiles are proposals derived from native scene graphs. They are not
validated merely because event planning succeeds.

## Catalog and planning

```bash
python scripts/catalog_scenes.py \
  --executable "$VIRTUALHOME_EXECUTABLE" --xorg-root "$XORG_ROOT" \
  --gpu-index 3 --output outputs/scene_catalog
python scripts/plan_rooms.py \
  --catalog outputs/scene_catalog/catalog.json \
  --output outputs/room_plans --seed 42
```

The catalog probes scene IDs 0 through 6 by default; `--scenes` selects an explicit
list. A room is identified by `(scene, room_id)`, because one layout can contain
multiple rooms named `bedroom`. Catalog creation saves native graphs without
rendering actions or changing object states.

Each profile selects room-local manipulation targets, surfaces and patrol
anchors. It derives camera bounds and the hidden viewing direction from the
room geometry. Character spawn proposals avoid furniture bounding boxes and
one another. The simulator must still verify navigation and visibility.

The current recipe requires two distinct portable targets, at least two
distinct opening targets, a suitable switchable object and two support
surfaces. It excludes ambiguous same-class targets, windows, curtains and
doors from opening events. Rooms without these capabilities remain explicitly
unsupported by this recipe. They are not replaced with kitchen samples.

All ten events, their operators, visibility intentions and time windows are
frozen before rendering. Retries replay the same configuration and program.
No objects are moved or created to make a room qualify.

Room recipes use an explicit `room_recipe` scene-signature policy. It preserves
all non-character object identities, properties, states and containment edges,
plus support edges for every manipulation, placement and navigation target.
Native startup may recompute unrelated decoration support edges; these do not
invalidate a room recipe. Existing configurations retain the original full
scene-signature policy. The saved graph remains available in both cases.

## Native validation

Choose profile names marked `planned` in `profiles.json`:

```bash
python scripts/rollout_rooms.py \
  --inputs outputs/room_plans \
  --profiles scene_0_livingroom_335 scene_1_bedroom_50 \
  --output outputs/room_validation \
  --executable "$VIRTUALHOME_EXECUTABLE" --xorg-root "$XORG_ROOT" \
  --gpu-index 3 --max-attempts 3
```

The runner snapshots release source, saves source hashes, executes profiles in
the requested order, and records success or rejection for each room. A failed
room never consumes another room's quota. Use a new output directory for a new
code revision; existing captures and frozen inputs remain available.

Acceptance uses the existing duration, cadence, role, visibility, camera and
motion audits. Each accepted capture has ten events, a private pool of sixty
QA items, a clean video and a highlighted debug video. This validation runner
does not automatically publish a ten-question evaluation packet or count a
partial recording as a dataset sample. Use the existing selection and prefix
export stage after acceptance.

The observer remains character 0. Characters 1 and 2 perform all changes. QA
location choices use observed locations and room-independent distractors,
without assuming every room has a kitchen counter. Blind queries remain
uncertain until the observer sees the changed target again.

## Current validation boundary

Automatic profiles do not yet provide a measured navigation-time model or
guaranteed viewing poses. In native trials, crowded interaction positions,
occluded computer bodies and long routes between opening targets have caused
rejections. Closet bodies and their contents can also share an instance color;
these ambiguous masks are rejected rather than treated as object visibility.
The kitchen renderer still contains a specialized choreography;
room-local IDs and adaptive camera targets do not make that choreography
universally feasible. Treat this workflow as a room-validation pipeline until
the requested room has an accepted capture.
