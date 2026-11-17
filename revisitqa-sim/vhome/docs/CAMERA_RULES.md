# Camera continuity rules

The camera follows one observer for the entire episode. A clear endpoint alone is not sufficient: its movement to that endpoint must also be continuous and collision-free.

## Hard limits

- Keep the same camera and a 75-degree field of view throughout the clip. Verify the actual rendered lens, not only the configuration. Do not cut between shoulder views, zoom, skip frames, blend frames, or retime the video to hide a transition.
- Limit translation to 3.5 m/s. The randomized preset limits rotation to 110 degrees/s; at 20 fps its adjacent frames may differ by at most 0.175 m and 5.5 degrees. The legacy mixed preset retains its 120-degree/s limit. Apply these limits across action boundaries and inspection snapshots as well as inside actions.
- Initialize the camera once after scene setup and before the first clip frame. Preserve camera position, orientation, and velocity between all subsequent simulator commands. A new action must not reset the camera to the character-relative offset.
- Advance the controller once per recorded observer frame. Wall-clock delays, HTTP inspections, duplicate character streams, and segmentation renders must not advance it again.
- Sweep the camera's 18 cm collision sphere along each movement segment, with 4 cm clearance. Include the observer and other characters in endpoint and path checks. People can temporarily occlude the line of sight, but the camera must not enter their bodies. Ceiling lamps also get bounds-based keepout volumes because their native collision geometry is incomplete.
- Never teleport to escape an infeasible position. Hold the last pose if a safe move is unavailable. Reject the capture if collisions or loss of observer framing remain.

## Tracking behavior

Choose among clear rear and shoulder positions by their distance from the preferred rear view and the previous camera position. Avoid a fixed left/right priority that changes sides as soon as one probe becomes blocked. Approach the selected position with a critically damped position spring and an acceleration-limited velocity. Orbit selection redirects motion around the observer; sweep the resulting movement segment against obstacles. Avoidance and collision braking can change velocity faster than ordinary tracking.

Keep a stable viewing direction toward the current observation region during short walks and backtracking. Change the region only at a planned transition. Interpolate its heading with a frame-rate-independent exponential Slerp; the physical camera still follows the same speed limits. Region anchors are static scene positions and do not track hidden object states. During offscreen kitchen manipulations, the randomized preset uses a fixed 180-degree world heading. A native room capture first looks away from the target. If the recorded target is already hidden but the operator remains visible, it selects one of the two 120-degree rear-side directions and keeps the one closest to the observer's travel direction. The successful direction is retained during the manipulation. Entering and leaving either heading uses the same interpolation and motion limits.

Maintain an orbital clearance around the observer instead of taking a shortcut across the head. Slow translation when the view cannot keep up with the observer. Use one continuous look-at target near the observer's upper body. The randomized preset sets `focus_height_m` to 1.5 m above the character origin to retain headroom during close overhead turns. Do not switch rotation formulas when entering or leaving avoidance. Damp yaw and pitch separately, keep roll at zero, and enforce the angular-speed cap on the resulting quaternion. Preserve spring and angular velocities across action boundaries. Use continuous framing correction rather than repeatedly halving a frame step. Position and rotation retain their state when the observer turns quickly.

The scene-0 recipe also constrains camera positions to a configured interior region, leaving extra clearance near doorway panels and tall furniture. This region is scene-specific and is checked on exported poses.

The controller settings come from `camera_continuity` in the chosen preset. The randomized preset uses position, rotation, and heading smoothing times of 0.30, 0.18, and 0.12 seconds. These parameters change the real camera trajectory; no frame blending or optical-flow interpolation is applied to the video. The launcher writes those same values to `logs/camera_rules.json` for the Unity component. Acceleration shapes normal tracking; collision braking may stop faster. Translation and rotation step limits remain hard constraints.

## Evidence and acceptance

Each saved RGB frame has a camera-pose sidecar. The Python recorder copies its position, quaternion, and field of view into `frame_manifest.jsonl`. It also compares the normal and segmentation camera-pose sidecars for each action frame. Inspection and idle frames retain the frozen camera pose. `camera_continuity.json` checks the exact exported frame sequence, including every action boundary. Missing or invalid poses, lens changes, and excessive steps fail acceptance.

These checks supplement the existing collision, headroom, event visibility, timing, and observer-motion checks. They do not replace them. Failed clips must not be exported as accepted datasets. Image-motion diagnostics and visual review can locate residual scene or animation discontinuities that a camera-pose check cannot detect.

The previous `multi_16` capture predates these rules. Its original collision and framing checks passed, but it contains abrupt avoidance transitions, including the adjacent frames at 45.50 and 45.55 seconds. It is retained as the comparison clip and is not evidence of continuity compliance.

## Accepted damped-camera capture

`offscreen5_smooth_transfer_v2_attempt_02` is a validated seed-42 capture with the damped controller. It contains 1,329 frames at 20 fps (66.45 seconds), ten events, five objects, two operating characters, and 60 questions. Five manipulations have zero target and operator pixels in every injection frame. All five have later visible inspections. Actual start gaps are 4.9–7.9 seconds.

All 51 action boundaries pass the camera continuity check. Maximum translation and rotation per frame are 0.175000145 m and 5.293979 degrees. The field of view stays at 75 degrees. The 95th-percentile camera speed, linear acceleration, turn rate, and angular acceleration are 2.4701 m/s, 12.0663 m/s², 97.4218 degrees/s, and 335.0793 degrees/s². These are descriptive measurements of this route, not a controlled comparison with the previous capture.

All 1,593 camera collision samples have zero overlaps. The minimum silhouette top margin is 36 pixels. The observer moves in 74.6% of frames and moves or turns in 83.8%; its longest inactive interval is 1.25 seconds. An approximate head-projection diagnostic flags four frames (0.2 seconds) near a hanging lamp. Visual review confirms brief lamp occlusion, rather than image-edge cropping. Fast-turn frame sequences remain continuous, although turns are still noticeable.

The frozen program has ID `7b692545c12796c422e97368900070ff401d88568d5cb431d9f593c4410a7302`. This batch accepted its second attempt. Earlier development batches also failed and are retained in external capture storage; the batch result is not an overall success-rate estimate. One accepted seed does not establish that every program or native navigation replay will succeed.

## Previous accepted capture

The continuous controller compiles against the VirtualHome 2.3.0 runtime. The Python suite has 56 passing tests, including action-boundary cuts, quaternion sign changes, missing poses, lens mismatches, frozen-program validation, and seeded planning diversity.

The earlier two-offscreen-event `randomized_09` capture passes its complete episode audit. It predates the five-offscreen-event quota and the damped controller. It contains 1,374 frames at 20 fps (68.7 seconds), ten events, five objects, two operating characters, and 60 questions across the six QA categories. Actual event gaps are 5.0–7.0 seconds. Its event identities and start times match the frozen seed-42 program; both planned offscreen manipulations have zero target and operating-character pixels throughout.

The maximum camera step is 0.17500064 m and 6.000019 degrees, within the numerical tolerance of the configured limits. All 49 action boundaries are checked; the largest boundary translation is 0.1042 m. The actual field of view remains 75 degrees. RGB and segmentation pose comparisons pass, and all 1,656 camera telemetry samples have zero overlaps.

The observer translates in 77.2% of frames and translates or turns in 83.0%. Its longest interval without either activity is 1.95 seconds. The minimum observer-silhouette upper margin is 42 pixels. This margin measures image cropping, not whether a lamp or another object briefly occludes the head.

This is one accepted capture, not a guarantee that every seed or replay will succeed. Native navigation can vary. Each capture must pass its own checks; failed attempts remain rejected. The offline 100-seed planning check establishes reproducible symbolic programs, not rendering success for all seeds.
