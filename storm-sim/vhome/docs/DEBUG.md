# Collision-aware camera and debug video

The debug runtime attaches a persistent controller to the observer's `REAR_OVERHEAD` camera. It advances once per recorded observer frame and restores the same pose for RGB, segmentation, and HTTP inspections. It scores clear rear and shoulder positions against its current position, then moves toward the chosen point under translation and rotation limits. A new simulator command does not reset the camera.

The controller sweeps an 18 cm sphere with 4 cm clearance along the actual movement path. Walls and character bodies constrain that path. Ceiling lamps also receive bounds-based keepout volumes because their native colliders are incomplete. People may occlude the view without shortening the observer-to-camera sightline. The camera follows an orbit around the observer instead of crossing over the head. It holds its position when no safe movement is available. The complete policy is in [CAMERA_RULES.md](CAMERA_RULES.md).

For the continuous preset, the runtime renders image stream 0 while retaining all character animation and skeleton recordings.

The original simulator stays intact. The build script copies it into a separate directory and patches camera creation to attach the component. It also gives every recorder a minimum recording budget of 1,200 frames, preventing the idle observer’s shorter estimate from cutting off another character’s recording. The patch applies to the VirtualHome 2.3.0 Linux binary. The patch also replaces the upstream nine-bit instance-color encoder with a fifteen-bit encoder, avoiding color reuse after 512 instances. Structural groups can still share a color and are checked separately. Its source is in `unity_debug/`; the build records assembly and source hashes in `camera_guard.json`.

## Run

From the project root on an Ubuntu/Debian machine:

```bash
bash scripts/setup_mono.sh
python scripts/build_camera_guard.py --output .runtime/simulator-observer
python scripts/rollout.py \
  --executable .runtime/simulator-observer/linux_exec.v2.3.0.x86_64 \
  --config configs/mixed.yaml \
  --xorg-root .runtime/xorg --gpu-index 0 \
  --output outputs/debug_001
python scripts/validate.py outputs/debug_001
```

`setup_mono.sh` extracts the C# compiler and assembly tools locally; it does not install system packages. `build_camera_guard.py` requires a new output directory. A pre-existing Xorg display can be selected with `--display` instead of `--xorg-root`.

The separate observer preset records a removal outside the observer camera, an inspection after turning back, and a visible return performed by the second character. The observer never manipulates objects. In the stationary preset, the guard holds its pose during the other character's actions to prevent VirtualHome from resetting the idle character's heading. The default mixed routine runs the observer and household actors concurrently and does not use the pose lock. `STORM_OBSERVER_POSE` points to the hold marker written by the rollout; the component captures its own Unity transform before an injector script starts.

## Output

- `video.mp4`: clean RGB video with the collision-aware camera.
- `debug.mp4`: the same frames with yellow target fill, outline, and bounding box; a separate 64-pixel header above the image shows time, object identity, current action, and visible pixel count. The header never covers the camera image. A 640 × 480 capture produces a 640 × 544 debug video.
- `frame_manifest.jsonl`: RGB/mask pairing, actual camera pose and lens, active target, palette entry, action, and output frame index.
- `camera_continuity.json`: translation, rotation, fixed-lens, and action-boundary checks on exported frames.
- `logs/camera_collision.csv`: requested and actual follow distance, camera position, detected obstruction, and overlap flag for each sampled Unity render frame.
- `debug_validation.json`: frame and highlight counts plus the camera audit.

Only accepted event targets are highlighted. Transit and rejected-candidate frames have no target overlay. Highlights use the synchronized instance mask, so they do not reveal an object behind a wall or inside a closed container. Zero target pixels are labeled `NOT VISIBLE`; that label alone does not claim that a disappearance event has occurred.

Keep `debug.mp4` and the frame manifest out of model evaluation inputs. The normal exporter continues to use `video.mp4`.

The audit rejects a debug export if any recorded camera sphere overlaps a solid collider. This tests the geometry represented by the simulator's collision meshes; it is not a proof for every untested room or asset. The camera may move closer to the character in narrow spaces, so the full body is not guaranteed to remain in view.

To rebuild an overlay from an existing successful capture, use `python scripts/render_debug.py outputs/debug_001`. This command requires synchronized mask frames and will not overwrite an existing `debug.mp4`.
