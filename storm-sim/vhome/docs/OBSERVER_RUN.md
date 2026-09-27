# Verified observer episode

Episode `observer_15` was recorded on Linux on 2026-09-18 with the stationary two-character preset. The original capture is stored externally and is not included in this code release.

The blue-shirted character only walks and turns. The green-shirted character takes the bowl from the kitchen counter, stores it in the refrigerator while the observer looks elsewhere, then retrieves it and puts it back on the counter.

| Check | Result |
| --- | --- |
| Video | 945 frames, 20 fps, 640 × 480, 47.25 seconds |
| Offscreen action interval | 14.50–25.70 seconds |
| Target and injector during that interval | Zero visible pixels in all 224 frames |
| Original support and container during that interval | Zero visible pixels in all 224 frames |
| Absence confirmed after returning to the viewing anchor | 31.40 seconds |
| First target reobservation after the absence check | 35.45 seconds |
| Injector visible during the return sequence | 205 frames |
| Injector distance from initial target / storage container | 0.74 m / 1.69 m in the horizontal plane |
| Camera checks | 2,061 samples; 544 retractions; zero collider overlaps |
| Questions | Eight questions spanning all six categories, including two uncertain answers |
| Evaluation export | Three prefix videos with answers stored separately |

These times refer to written video frames. The action interval bounds a sequence; the absence confirmation is an observation checkpoint, not an estimate of the first physical removal frame. The bowl becomes visible again before the final returned-state checkpoint at 47.25 seconds.

Reproduce a new capture on a configured Linux host with:

```bash
cd /path/to/STORM-Bench/storm-sim/vhome
.venv/bin/python scripts/rollout.py \
  --executable .runtime/simulator-observer-v4/linux_exec.v2.3.0.x86_64 \
  --config configs/observer_stationary.yaml --xorg-root .runtime/xorg --gpu-index 3 \
  --output outputs/observer_new
```

The patched runtime manifest is saved with the episode. Navigation timing can vary; observations, questions, and acceptance checks are generated from the actual capture. Earlier trial outputs are debugging records, not additional accepted episodes. In particular, `observer_11` was rejected after the injector reach check was added.

The source capture and its full RGB/mask frames remain external. Run the complete episode validator against your own capture directory.
