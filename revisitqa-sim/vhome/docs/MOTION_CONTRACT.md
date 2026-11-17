# Motion contract

Every room episode freezes its motion contract before Unity starts recording.
The event program stores the route with the event, so retries cannot choose a
more convenient camera path after seeing rendered evidence.

## Character roles

- Character 0 is the observer. It may walk and turn, but it never changes an
  object state or carries an object.
- Characters 1 and 2 are operators. Only the operator named by an event may
  approach or manipulate that event target.
- Sparse rooms with at most seven usable patrol anchors use one active operator
  when all selected work zones are within 3.0 m. Distant zones may use a second
  operator so one person does not repeatedly cross the room.
- Operator walking is serialized. A second operator is never moved in the same
  native script as the current event.
- When the active operator must approach a new target, that walk runs alongside
  the observer's visible approach. For an offscreen event, the observer conceals
  the target only after the operator reaches it. This avoids revealing the
  target again with a late operator walk. The manipulation still starts in a
  later command after both endpoints pass the runtime clearance checks.
- If the two direct approach segments intersect, the observer first moves to a
  frozen room waypoint on the side farthest from the operator's future path and
  retries from there. The operator remains still during this staging step. The
  subsequent concurrent paths must keep the 1.10 m runtime margin along both
  complete segments.
- Setup walks are also serialized and occur before the accepted video begins.
- An operator whose first event occurs later in the episode enters only after
  the observer has moved both its target and workspace fully out of frame. The
  planner forces that first event to be offscreen, and the runtime records a
  post-entry patrol step before the event can start. The next paired event on
  that object is kept in view, so the observer discovers the hidden change
  without making an immediate second concealment pass. This keeps a waiting
  actor from blocking narrow early routes without creating a visible appearance
  jump or a late cluster of long walks.

## Route planning

- Each event contains ordered approach and hidden waypoint lists plus an
  operator approach target. All waypoints are native room objects.
- Room routes may use doors, cabinets, counters, beds, tables, seating,
  bookshelves, wall shelves, closets, sinks, toilets, and washing machines.
  These anchors let small bathrooms and sparse living rooms use the same
  planner without inventing a room-specific path.
- The current event target and its operator approach object are excluded from
  that event's observer route. A later target assigned to the same operator may
  remain a patrol anchor when it is outside the active workspace. A workspace
  assigned to an inactive operator is excluded at runtime. Native walking ends
  beside an anchor rather than at its object center, so this rule prevents an
  apparently clear route from ending beside the parked operator while retaining
  enough routes in sparse rooms.
- A container selected as a fallback anchor is checked against the live scene
  graph. Once opened, it is skipped until it is closed again because the native
  navigator cannot reliably approach an open door or cabinet panel.
- A waypoint must move the observer by at least 0.75 m.
- If every regular waypoint is blocked, the runtime may use a 0.50--0.75 m
  relocation that satisfies the same path, endpoint, and measured collision
  checks. This keeps the observer moving away from a nearby character without
  forcing a route through that character's workspace.
- A route segment used while an operator is stationary must stay at least
  1.00 m from the manipulation target during static planning.
- Character endpoints must remain at least 0.85 m apart after setup, walking,
  and manipulation.
- Runtime route selection rejects predicted path or endpoint clearance below
  1.10 m, leaving 0.25 m for native navigation endpoint error. It prefers
  1.20 m of path and endpoint clearance. The observer
  may leave a starting position inside either margin only when every step
  preserves the 0.85 m absolute clearance and increases separation. A waypoint that produces
  less than 0.10 m of travel is timed at each anchor. That anchor is excluded
  after 0.5 seconds of consecutive low-motion footage. This gives the camera
  time to finish a damped turn while staying well below the two-second inactivity
  limit.
- A concurrent manipulation starts only when the observer is at least 1.45 m
  from both operators and their reserved work positions. Its observer route
  requires 1.45 m of path clearance from the active operator and every reserved
  work position. An inactive helper retains the normal 1.20 m runtime margin because it
  has no root motion during the action. These margins cover the
  forward root motion produced by native open, close, and switch animations.
- A route records its planned turn angle and target clearance. Native walking
  handles body rotation; the camera separately limits heading motion to 80
  degrees per second so a necessary room-scale reversal does not become a
  single-frame visual jump.
- Native walking may return after only part of a route segment. The observer
  keeps the same frozen destination across those partial walks and advances to
  the next anchor only after reaching it or exhausting the low-motion budget.
  Event preparation prefers destinations that already meet the manipulation
  start clearance, avoiding last-second shuttling between nearby furniture.
- A new target begins with the first waypoint in its frozen approach order.
  The renderer does not rotate that route from the prior event's anchor. If
  VirtualHome rejects an observer-only walk at runtime, that anchor is skipped
  for the current phase and the next frozen alternative is tried without
  changing the event or its visibility intention.
- When a complete approach reaches an anchor but a new target or small target
  remains outside the frame, the observer makes one native turn toward it
  before starting another room-scale walk. A rejected turn is discarded and
  the frozen route continues. Large revisited targets proceed directly to the
  next waypoint and may use the later exact-anchor turn fallback from a known
  visible anchor.
- For an offscreen event, the first complete concealment walk may be followed
  by one native turn while the camera keeps its frozen offscreen heading. This
  gives the smoothed camera time to finish rotating without adding a second
  room-scale detour. The final recorded frame must still contain zero target
  pixels before the event can be injected.
- When a delayed operator enters for its first offscreen event, the next patrol
  action records a post-spawn verification frame. The event cannot start while
  that frame still shows the target or operator.
- The observer uses an in-place native turn during manipulation. Object-state
  changes therefore remain atomic with respect to observer navigation: a
  rejected walk cannot accompany a successful open, close, switch, grab, or
  placement action. The target-free 30-degree left or right turn cannot fail
  because a small object or another character is outside the native selection
  cone. Approach, concealment, timing, and discovery paths retain translational
  observer motion between manipulations.
- If every collision-safe timing path is reserved, the observer may make one
  native turn in place. Offscreen timing still requires the turn's final frame
  to contain zero target and operator pixels; otherwise the episode is rejected.
- Initial target acquisition uses the complete native `Walk` path to a frozen
  anchor. Offscreen concealment uses bounded `WalkTowards` segments and checks
  the target and operator mask after every segment. This avoids a room-scale
  detour when one short step and a camera turn already hide the event. Timing
  patrols, manipulation, and final inspection use the same shorter segments so
  the observer keeps moving without crossing several room zones in one event
  slot. Complete walks keep 1.45 m from every operator and reserved work
  position to absorb native navigation endpoint error; shorter segments use
  their phase-specific margins.
- A repeated target first returns along its previously verified visible anchor
  with `WalkTowards`. It does not repeat the larger first-acquisition margin
  unless its operator must also change position. Its path keeps the normal
  1.20 m runtime margin from an inactive helper and still enforces the 0.85 m
  measured clearance after every native action.
- A repeated target with an undiscovered offscreen change first attempts one
  in-place turn toward the target. If the turn restores visibility, the runtime
  records the discovery and avoids a redundant room-scale walk. If it does not,
  the frozen approach route continues normally.
- When a target has an undiscovered offscreen change, the return to its verified
  visible anchor uses a complete `Walk`. This avoids a chain of partial steps
  while the camera is reacquiring an opened cabinet or another changed object.
  If the normal 1.20 m recovery route is unavailable, the same frozen route may
  use the 1.10 m hard margin and must still pass the 0.85 m measured check.
- Target acquisition, offscreen concealment, and a visible event's
  pre-operation waiting step may use the same 1.10 m fallback when the normal
  route is unavailable. Each move completes and passes measured clearance
  before the other agent starts the manipulation. Operator motion and
  manipulation root motion are never combined with this tighter route.
- If that verified anchor is the only collision-safe visible position, the
  observer may use one native `TurnTo` action there. The turn supplies smooth
  body and camera motion, is limited to one attempt, and is followed by the
  next planned walk rather than repeated stationary turns.
- If manipulation has no second collision-safe walking anchor, the observer
  turns once while the other agent performs the operation. Visible events turn
  toward that agent, which is already in the recorded view. Hidden events turn
  toward the closed target while retaining their offscreen camera direction.
  The runtime still checks character clearance and recorded visibility.
- The observer carries its current anchor across event boundaries. Concurrent
  manipulation must choose a different anchor, so an already reached furniture
  target cannot masquerade as continued locomotion.
- Target choice prefers the next operation near the observer's most recently
  viewed target. A short penalty prevents mechanical immediate reversals when a
  nearby operation is available, while longer cross-room walks remain grouped.
  Seeded jitter preserves variation without making the camera shuttle between
  distant operator zones.

## Concurrent motion

The observer may walk while one stationary operator manipulates an object only
when the remaining observer step is at most 1.5 m. Longer observer routes and
all operator approaches are serialized. This removes the earlier three-agent
script that could make two paths cross while an event was being executed.

## Timing

- The first event is planned near 4 seconds.
- In-view events normally reserve 6.8 seconds from the preceding event start.
- Offscreen events normally reserve 7.1 seconds because the observer must first
  establish the visible state and then leave the target and operator outside
  the frame.
- Every native-room event has a 4.75-second timing tolerance. This absorbs native animation
  length without changing the frozen order or retiming recorded frames.
- Runtime preparation aims to finish 0.5 seconds before the preferred event
  center. It may use the rest of the frozen tolerance only when native walking
  or manipulation takes longer, preventing avoidable 3-second event gaps.
- Pickup and replacement dependencies retain their dedicated transfer slot.
- The event-specific gap is saved in the event program. Ten events must still
  fit the 60–70 second episode contract.

## Camera interpolation

The follow camera uses bounded interpolation rather than copying the character
pose directly. The room profile limits camera speed to 2.6 m/s, turn rate to 80
degrees/s, and acceleration to 6 m/s². Position, rotation, and heading use
separate smoothing windows. Post-render validation rejects excessive 95th
percentile linear or angular acceleration and action-boundary direction jumps.

## Runtime evidence

`motion_ledger.json` records the time, character endpoints, route context, and
minimum clearance after every movement or manipulation. The final skeleton
audit still requires at least 80 percent active observer frames and no inactive
run longer than 2 seconds. Successful native actions do not waive collision,
visibility, camera, timing, or state-transition checks.
