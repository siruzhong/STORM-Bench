"""Rendered evidence metrics used by egocentric event quality gates."""
from __future__ import annotations

import numpy as np


def union_instance_mask(event, object_ids):
    mask = np.zeros(event.frame.shape[:2], dtype=bool)
    for object_id in object_ids:
        if object_id and object_id in event.instance_masks:
            mask |= np.asarray(event.instance_masks[object_id], dtype=bool)
    return mask


def mask_bbox(mask):
    rows, cols = np.where(mask)
    if not len(rows):
        return None
    return [int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)]


def frame_evidence(event, object_ids):
    """Serializable size, location and edge-margin evidence for instances."""
    mask = union_instance_mask(event, object_ids)
    bbox = mask_bbox(mask)
    height, width = mask.shape
    if bbox is None:
        return {
            "visible": False,
            "mask_pixels": 0,
            "bbox": None,
            "bbox_width": 0,
            "bbox_height": 0,
            "edge_margin": 0,
            "center_distance": 1.0,
        }
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    center_distance = np.hypot(
        (center_x - width / 2.0) / max(width / 2.0, 1.0),
        (center_y - height / 2.0) / max(height / 2.0, 1.0),
    )
    return {
        "visible": True,
        "mask_pixels": int(mask.sum()),
        "bbox": bbox,
        "bbox_width": int(x2 - x1),
        "bbox_height": int(y2 - y1),
        "edge_margin": int(min(x1, y1, width - x2, height - y2)),
        "center_distance": float(center_distance),
    }


def bbox_iou(left, right):
    if not left or not right:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def visual_delta(before_frame, after_frame, before_mask, after_mask):
    """RGB change inside the union of before/after object masks."""
    mask = np.asarray(before_mask, dtype=bool) | np.asarray(after_mask, dtype=bool)
    if not np.any(mask):
        return {"mean_pixel_delta": 0.0, "changed_pixel_ratio": 0.0}
    delta = np.abs(
        before_frame.astype(np.int16) - after_frame.astype(np.int16)
    ).mean(axis=2)
    values = delta[mask]
    return {
        "mean_pixel_delta": float(values.mean()),
        "changed_pixel_ratio": float(np.mean(values >= 20.0)),
    }


def evidence_passes(evidence, min_pixels=3000, min_dimension=40,
                    min_edge_margin=8, max_center_distance=1.0):
    return bool(
        evidence.get("visible")
        and evidence.get("mask_pixels", 0) >= int(min_pixels)
        and min(evidence.get("bbox_width", 0),
                evidence.get("bbox_height", 0)) >= int(min_dimension)
        and evidence.get("edge_margin", 0) >= int(min_edge_margin)
        and evidence.get("center_distance", float("inf"))
        <= float(max_center_distance)
    )


def target_evidence_passes(evidence, quality, occluded=False):
    """Apply the shared target framing contract, including its edge waiver."""
    prefix = "min_occluded_target" if occluded else "min_target"
    minimum_edge_margin = int(quality.get("min_edge_margin", 8))
    base_quality = evidence_passes(
        evidence,
        min_pixels=quality.get(
            f"{prefix}_pixels", 1500 if occluded else 3000),
        min_dimension=quality.get(
            f"{prefix}_dimension", 30 if occluded else 40),
        min_edge_margin=0,
        max_center_distance=quality.get("max_center_distance", 1.0),
    )
    if not base_quality:
        return False
    if int(evidence.get("edge_margin", 0)) >= minimum_edge_margin:
        return True
    waiver_pixels = quality.get("edge_margin_waiver_min_pixels")
    if waiver_pixels is None:
        return False
    return bool(
        int(evidence.get("mask_pixels", 0)) >= int(waiver_pixels)
        and float(evidence.get("center_distance", float("inf")))
        <= float(quality.get(
            "edge_margin_waiver_max_center_distance", 0.25))
    )
