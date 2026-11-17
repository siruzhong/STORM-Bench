#!/usr/bin/env python
"""Extract, from the AI2-THOR rollout plans, when each event object's original spot is in camera view.

Writes revisitqa_sim_traces/thor_focus.json: {episode: {event_obj_type: [[start_s, end_s], ...]}} with intervals
where the object's target id is in the planned focus list of a frame whose phase is before/after/hidden/
intervention (approach and depart frames are excluded: the camera is still turning). Hidden-phase focus lists
include disabled (disappeared) objects, so an interval means the spot is in view whether or not the object is there.

Set REVISITQA_THOR_ROLLOUT_ROOT to the local directory that contains the relative ``rollout_dir`` values in
``thor_timeline.json`` (for example the parent of ``benchgen_four_scene_28_*/``).
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent / "revisitqa_sim_traces"
PHASES = {"before", "after", "hidden", "intervention"}


def main():
    root = Path(os.environ.get("REVISITQA_THOR_ROLLOUT_ROOT", "")).expanduser()
    if not root.is_dir():
        raise SystemExit("Set REVISITQA_THOR_ROLLOUT_ROOT to a directory containing the relative rollout paths")
    timeline = json.loads((HERE / "thor_timeline.json").read_text())
    out = {}
    for episode, info in sorted(timeline.items()):
        plan = json.loads((root / info["rollout_dir"] / "episode_plan.json").read_text())
        ids = {e["target_id"]: e["target_type"] for e in plan["event_program"]["events"]}
        dt = 1.0 / plan["trajectory"]["fps"]
        spans = {t: [] for t in set(ids.values())}
        for frame in plan["trajectory"]["frames"]:
            if frame["phase"] not in PHASES:
                continue
            t = frame["time_s"]
            for tid in set(frame["focus_ids"]) & set(ids):
                s = spans[ids[tid]]
                if s and t - s[-1][1] <= dt * 1.5:
                    s[-1][1] = round(t, 3)
                else:
                    s.append([round(t, 3), round(t, 3)])
        out[episode] = spans
    (HERE / "thor_focus.json").write_text(json.dumps(out, indent=0, sort_keys=True))
    print(len(out), "episodes")


if __name__ == "__main__":
    main()
