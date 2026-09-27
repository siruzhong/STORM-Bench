"""Visibility judgement for the No-Witness invariant.

The key distinction (verified empirically): AI2-THOR's metadata `visible` flag
means "in view AND within ~1.5m interaction range" -- that is NOT what we want.
For "is this object actually rendered on screen right now" we must use the
instance segmentation detection boxes (`event.instance_detections2D`), which
list every object drawn in the frame regardless of distance.
"""
from __future__ import annotations


def in_frame_ids(event, min_pixels: int = 1) -> set:
    """Object ids that are actually rendered in this frame.

    Requires the controller to run with renderInstanceSegmentation=True.
    `min_pixels` filters out boxes whose area is below a threshold, guarding
    against 1-2px sliver detections at the frame edge.
    """
    dets = event.instance_detections2D
    if not dets:
        return set()
    ids = set()
    for obj_id, box in dets.items():
        # box is (x1, y1, x2, y2) in pixels.
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) >= min_pixels:
            ids.add(obj_id)
    return ids


class VisibilityTimeline:
    """Records, per frame, which target objects were rendered on screen.

    Used to prove the No-Witness invariant (an object never changes while it is
    on screen) and to check Perceptibility (a change is observable if the object
    -- or its old location -- is on screen at least once in the relevant loop).
    """

    def __init__(self, min_pixels: int = 4):
        self.min_pixels = min_pixels
        self.frames: list[set] = []  # frames[i] = set of object ids in frame i

    def record(self, event) -> set:
        ids = in_frame_ids(event, self.min_pixels)
        self.frames.append(ids)
        return ids

    def ever_visible(self, object_id: str) -> bool:
        return any(object_id in f for f in self.frames)

    def visible_frame_count(self, object_id: str) -> int:
        return sum(1 for f in self.frames if object_id in f)

    def first_visible_frame(self, object_id: str) -> int:
        for i, f in enumerate(self.frames):
            if object_id in f:
                return i
        return -1

    def summary(self, object_ids: list[str]) -> dict:
        return {
            oid: {
                "ever_visible": self.ever_visible(oid),
                "visible_frames": self.visible_frame_count(oid),
                "first_frame": self.first_visible_frame(oid),
                "total_frames": len(self.frames),
            }
            for oid in object_ids
        }

    def offscreen_windows(self, object_id, margin=3, min_len=1):
        """Frame ranges [start, end) where object_id is continuously off-screen,
        eroded by `margin` on each side.

        The margin guarantees the object is off-screen for `margin` frames before
        and after the window too, so a mutation fired inside the window can never
        be witnessed at the window boundary (the agent isn't about to see it).
        """
        n = len(self.frames)
        offscreen = [object_id not in self.frames[i] for i in range(n)]
        windows = []
        i = 0
        while i < n:
            if not offscreen[i]:
                i += 1
                continue
            j = i
            while j < n and offscreen[j]:
                j += 1
            # [i, j) is off-screen; erode by margin
            s, e = i + margin, j - margin
            if e - s >= min_len:
                windows.append((s, e))
            i = j
        return windows
