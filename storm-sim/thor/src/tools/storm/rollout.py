"""Rollout helpers for the two-pass hidden-change pipeline.

Pass A (dry run): walk the pose list, recording per-frame visibility only.
Pass B (real run): walk the SAME pose list, and at each event's scheduled trigger
frame fire its mutation -- which Pass A has proven happens while the target is
off-screen, so the change occurs behind the agent's back with no cut or fade.
"""
from __future__ import annotations

import numpy as np

from .path import (
    poses_along_reachable_perimeter,
    poses_along_skeleton_exploration,
    poses_along_skeleton_navigation_loop,
    poses_along_spline,
    poses_along_thor_waypoints,
)
from .visibility import VisibilityTimeline


def build_poses(reach, fps, speed, inset, smooth, lookahead=0.35,
                path_mode="skeleton_exploration", perimeter_points=24,
                controller=None, objects=None, waypoint_object_types=None,
                waypoint_count=6, shortest_path_allowed_error=0.25,
                waypoint_min_clearance=0.35,
                waypoint_min_open_neighbors=3,
                path_seed=0, skeleton_coverage=0.65,
                skeleton_prune_length=0.5,
                skeleton_smoothing=0.05,
                loop_clearance=0.5,
                functional_zone_max_distance=1.6,
                functional_zone_merge_distance=1.25,
                body_max_yaw_rate=60.0,
                turn_smoothing_distance=0.35,
                look_object_types=None,
                look_max_turn_degrees=125.0,
                look_center_max_turn_degrees=180.0,
                look_min_distance=0.55,
                look_max_distance=5.0,
                look_object_strength=0.9,
                look_center_strength=0.65,
                look_max_yaw_rate=40.0,
                look_smoothing_frames=4,
                look_deadband_degrees=0.0,
                look_target_hold_frames=60,
                look_release_grace_frames=8,
                look_switch_margin=0.15,
                look_filter="lowpass",
                look_kalman_process_noise=900.0,
                look_kalman_measurement_noise=64.0,
                look_kalman_initial_uncertainty=100.0):
    """Precompute the deterministic pose list for one loop (shared by both passes)."""
    if path_mode == "skeleton_navigation_loop":
        return list(poses_along_skeleton_navigation_loop(
            reach,
            objects or [],
            fps=fps,
            speed=speed,
            lookahead=lookahead,
            seed=path_seed,
            clearance=loop_clearance,
            smoothing=skeleton_smoothing,
            functional_zone_max_distance=functional_zone_max_distance,
            functional_zone_merge_distance=functional_zone_merge_distance,
            max_yaw_rate=body_max_yaw_rate,
            turn_smoothing_distance=turn_smoothing_distance,
        ))
    if path_mode == "skeleton_exploration":
        return list(poses_along_skeleton_exploration(
            reach, objects or [], fps=fps, speed=speed,
            lookahead=lookahead, seed=path_seed,
            coverage=skeleton_coverage,
            prune_length=skeleton_prune_length,
            smoothing=skeleton_smoothing,
            look_object_types=look_object_types,
            look_max_turn_degrees=look_max_turn_degrees,
            look_center_max_turn_degrees=look_center_max_turn_degrees,
            look_min_distance=look_min_distance,
            look_max_distance=look_max_distance,
            look_object_strength=look_object_strength,
            look_center_strength=look_center_strength,
            look_max_yaw_rate=look_max_yaw_rate,
            look_smoothing_frames=look_smoothing_frames,
            look_deadband_degrees=look_deadband_degrees,
            look_target_hold_frames=look_target_hold_frames,
            look_release_grace_frames=look_release_grace_frames,
            look_switch_margin=look_switch_margin,
            look_filter=look_filter,
            look_kalman_process_noise=look_kalman_process_noise,
            look_kalman_measurement_noise=look_kalman_measurement_noise,
            look_kalman_initial_uncertainty=look_kalman_initial_uncertainty,
        ))
    if path_mode == "thor_shortest_path_waypoints":
        if controller is None or objects is None:
            raise ValueError("thor_shortest_path_waypoints requires controller and objects")
        return list(poses_along_thor_waypoints(
            controller, reach, objects, fps=fps, speed=speed,
            lookahead=lookahead, object_types=waypoint_object_types,
            max_waypoints=waypoint_count,
            allowed_error=shortest_path_allowed_error,
            waypoint_min_clearance=waypoint_min_clearance,
            waypoint_min_open_neighbors=waypoint_min_open_neighbors,
            look_object_types=look_object_types,
            look_max_turn_degrees=look_max_turn_degrees,
            look_center_max_turn_degrees=look_center_max_turn_degrees,
            look_min_distance=look_min_distance,
            look_max_distance=look_max_distance,
            look_object_strength=look_object_strength,
            look_center_strength=look_center_strength,
            look_max_yaw_rate=look_max_yaw_rate,
            look_smoothing_frames=look_smoothing_frames,
            look_deadband_degrees=look_deadband_degrees,
            look_target_hold_frames=look_target_hold_frames,
            look_release_grace_frames=look_release_grace_frames,
            look_switch_margin=look_switch_margin,
            look_filter=look_filter,
            look_kalman_process_noise=look_kalman_process_noise,
            look_kalman_measurement_noise=look_kalman_measurement_noise,
            look_kalman_initial_uncertainty=look_kalman_initial_uncertainty,
        ))
    if path_mode == "reachable_perimeter":
        return list(poses_along_reachable_perimeter(
            reach, fps=fps, speed=speed, lookahead=lookahead,
            anchor_count=perimeter_points,
        ))
    if path_mode != "spline":
        raise ValueError(f"unknown path_mode: {path_mode}")
    return list(poses_along_spline(reach, fps=fps, speed=speed,
                                   inset=inset, smooth=smooth,
                                   lookahead=lookahead))


def object_snapshot(event, object_ids):
    """Capture current world state of the given objects for the change log."""
    by_id = {o["objectId"]: o for o in event.metadata["objects"]}
    snap = {}
    for oid in object_ids:
        o = by_id.get(oid)
        if o is None:
            snap[oid] = {"present": False}
            continue
        snap[oid] = {
            "present": True,
            "position": o["position"],
            "isSliced": o.get("isSliced"),
            "isDirty": o.get("isDirty"),
            "isCooked": o.get("isCooked"),
            "isBroken": o.get("isBroken"),
            "isToggled": o.get("isToggled"),
            "isOpen": o.get("isOpen"),
            "openness": o.get("openness"),
            "isFilledWithLiquid": o.get("isFilledWithLiquid"),
            "fillLiquid": o.get("fillLiquid"),
            "parentReceptacles": o.get("parentReceptacles"),
        }
    return snap


def _teleport(controller, x, z, agent_y, yaw, horizon, force_action=False):
    return controller.step(
        action="TeleportFull",
        position={"x": x, "y": float(agent_y), "z": z},
        rotation={"x": 0.0, "y": yaw, "z": 0.0},
        horizon=float(horizon), standing=True, forceAction=force_action,
    )


def _pose_values(pose, default_horizon):
    if hasattr(pose, "yaw"):
        return pose.x, pose.z, pose.yaw, pose.horizon
    if len(pose) == 4:
        return pose
    x, z, yaw = pose
    return x, z, yaw, default_horizon


def walk_dry(controller, poses, agent_y, horizon, target_ids, min_pixels,
             force_action=False, on_frame=None):
    """Pass A: walk poses, record visibility timeline. Returns VisibilityTimeline."""
    tl = VisibilityTimeline(min_pixels=min_pixels)
    for i, pose in enumerate(poses):
        x, z, yaw, pose_horizon = _pose_values(pose, horizon)
        e = _teleport(
            controller, x, z, agent_y, yaw, pose_horizon, force_action)
        if not e.metadata["lastActionSuccess"]:
            raise RuntimeError(f"TeleportFull failed during dry run at frame {i}: {e.metadata.get('errorMessage')}")
        tl.record(e)
        if on_frame:
            on_frame(i, e)
    return tl


def walk_capture(controller, poses, agent_y, horizon, target_ids, min_pixels,
                 triggers=None, rng=None, on_trigger=None, force_action=False,
                 frame_writer=None, highlighted_writer=None,
                 highlight_alpha=0.75, on_frame=None,
                 highlight_frame_fn=None):
    """Pass B: walk poses capturing RGB, firing scheduled mutations.

    `triggers` maps frame_index -> list of events to apply just before rendering
    that frame. `on_trigger(event, frame_index)` is called after each mutation so
    the caller can log it. Returns (frames, VisibilityTimeline).
    """
    triggers = triggers or {}
    tl = VisibilityTimeline(min_pixels=min_pixels)
    frames = []
    highlighted_ids = set()
    for i, pose in enumerate(poses):
        # Apply any mutations scheduled for this frame BEFORE we render it, so the
        # frame already shows the post-change world (change happened off-screen).
        for ev in triggers.get(i, []):
            ev.apply(controller, rng)
            highlighted_ids.update(ev.highlight_object_ids())
            if on_trigger:
                on_trigger(ev, i)
        x, z, yaw, pose_horizon = _pose_values(pose, horizon)
        e = _teleport(
            controller, x, z, agent_y, yaw, pose_horizon, force_action)
        if not e.metadata["lastActionSuccess"]:
            raise RuntimeError(f"TeleportFull failed during capture at frame {i}: {e.metadata.get('errorMessage')}")
        if frame_writer is None:
            frames.append(e.frame)
        else:
            frame_writer.append_data(e.frame)
        if highlighted_writer is not None:
            if highlight_frame_fn:
                highlighted_frame = highlight_frame_fn(
                    i, e, highlighted_ids, highlight_alpha)
            else:
                highlighted_frame = highlight_instance_masks(
                    e.frame,
                    e,
                    highlighted_ids,
                    alpha=highlight_alpha,
                )
            highlighted_writer.append_data(highlighted_frame)
        tl.record(e)
        if on_frame:
            on_frame(i, e)
    return frames, tl


def highlight_instance_masks(frame, event, object_ids, alpha=0.75):
    """Overlay visible target instance masks in red without changing geometry."""
    alpha = min(max(float(alpha), 0.0), 1.0)
    mask = np.zeros(frame.shape[:2], dtype=bool)
    instance_masks = event.instance_masks
    for object_id in object_ids:
        if object_id in instance_masks:
            mask |= np.asarray(instance_masks[object_id], dtype=bool)
    if not np.any(mask):
        return frame.copy()

    highlighted = frame.copy()
    red = np.array([255.0, 0.0, 0.0])
    blended = highlighted[mask].astype(float) * (1.0 - alpha) + red * alpha
    highlighted[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    # Keep a crisp one-pixel red boundary around the translucent fill.
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    highlighted[mask & ~interior] = np.array([255, 0, 0], dtype=np.uint8)

    ring = mask.copy()
    for _ in range(2):
        expanded = ring.copy()
        expanded[1:, :] |= ring[:-1, :]
        expanded[:-1, :] |= ring[1:, :]
        expanded[:, 1:] |= ring[:, :-1]
        expanded[:, :-1] |= ring[:, 1:]
        ring = expanded
    highlighted[ring & ~mask] = np.array([255, 24, 24], dtype=np.uint8)
    return highlighted


def highlight_bboxes(frame, boxes, color=(255, 24, 24), thickness=3):
    """Draw evidence rectangles for post-change regions with no live instance."""
    highlighted = frame.copy()
    height, width = highlighted.shape[:2]
    color = np.asarray(color, dtype=np.uint8)
    thickness = max(1, int(thickness))
    for box in boxes:
        if not box:
            continue
        x1, y1, x2, y2 = [int(value) for value in box]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        highlighted[y1:min(y1 + thickness, y2), x1:x2] = color
        highlighted[max(y2 - thickness, y1):y2, x1:x2] = color
        highlighted[y1:y2, x1:min(x1 + thickness, x2)] = color
        highlighted[y1:y2, max(x2 - thickness, x1):x2] = color
    return highlighted
