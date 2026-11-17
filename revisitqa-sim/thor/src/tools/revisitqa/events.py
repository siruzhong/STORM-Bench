"""Event executors for the five hidden-change types, plus compatibility checks
and a structured change log.

Two-phase lifecycle (v1 "between-loops" mutation):
  - pre_setup(): runs BEFORE Loop 1. Only `appear` uses it (disable the object
    so it is absent in Loop 1, to be enabled during intervention).
  - apply():     runs during Intervention (camera off). Performs the mutation.

Truth source for "is the object rendered" is always instance segmentation
(see visibility.py) -- NOT the metadata objects list, because DisableObject
leaves the object in metadata while removing it from the render.
"""
from __future__ import annotations

import random

from .taxonomy import event_labels


def _find_object(event, spec):
    """Resolve an event's target object id from a config spec.

    spec may carry `object_id` (exact) or `object` (objectType; first match).
    """
    objs = event.metadata["objects"]
    if spec.get("object_id"):
        oid = spec["object_id"]
        return next((o for o in objs if o["objectId"] == oid), None)
    otype = spec["object"]
    return next((o for o in objs if o["objectType"] == otype), None)


def _place_on_receptacle(controller, object_id, rng, avoid_receptacle=None,
                         prefer_types=("CounterTop", "DiningTable", "Shelf", "Stool"),
                         allowed_receptacle_ids=None):
    """Physically place an object onto a receptacle surface (never floating).

    Uses GetSpawnCoordinatesAboveReceptacle + PlaceObjectAtPoint so the object
    rests on a real surface with a correct parentReceptacle, instead of being
    teleported to an arbitrary floating (x, 1.0, z). Tries multiple receptacles
    and multiple candidate points (some are occupied and fail).

    Returns (ok, receptacle_id, position) — position is the object's resting pos.
    """
    ev = controller.step(action="Pass")
    allowed = set(allowed_receptacle_ids or [])
    recs = [
        o for o in ev.metadata["objects"]
        if o.get("receptacle")
        and o["objectId"] != avoid_receptacle
        and (not allowed or o["objectId"] in allowed)
    ]
    # Prefer flat elevated surfaces; shuffle within each tier for variety.
    def tier(o):
        return prefer_types.index(o["objectType"]) if o["objectType"] in prefer_types else len(prefer_types)
    recs.sort(key=tier)
    tiers = {}
    for o in recs:
        tiers.setdefault(tier(o), []).append(o)
    ordered = []
    for k in sorted(tiers):
        bucket = tiers[k]
        rng.shuffle(bucket)
        ordered.extend(bucket)

    for rec in ordered:
        e = controller.step(action="GetSpawnCoordinatesAboveReceptacle",
                            objectId=rec["objectId"], anywhere=True)
        coords = e.metadata.get("actionReturn") or []
        rng.shuffle(coords)
        for coord in coords[:12]:  # sample a handful of candidate points
            e2 = controller.step(action="PlaceObjectAtPoint",
                                objectId=object_id, position=coord)
            if e2.metadata["lastActionSuccess"]:
                obj = next((o for o in controller.step(action="Pass").metadata["objects"]
                            if o["objectId"] == object_id), None)
                pos = obj["position"] if obj else coord
                return True, rec["objectId"], pos
    return False, None, None


# --- compatibility: which object property each event requires ---
def check_compatible(event, spec) -> tuple[bool, str]:
    o = _find_object(event, spec)
    if o is None:
        return False, f"object not found: {spec.get('object') or spec.get('object_id')}"
    t = spec["type"]
    if t == "move" and not o.get("pickupable"):
        return False, f"{o['objectType']} not pickupable (move)"
    if t == "appear" and not o.get("pickupable"):
        return False, f"{o['objectType']} not pickupable (appear)"
    if t == "state_change":
        action = spec.get("action")
        if action == "slice" and not o.get("sliceable"):
            return False, f"{o['objectType']} not sliceable"
        if action == "break" and not o.get("breakable"):
            return False, f"{o['objectType']} not breakable"
        if action in {"toggle", "toggle_on", "toggle_off"} and not o.get("toggleable"):
            return False, f"{o['objectType']} not toggleable"
        if action in {"dirty", "clean"} and not o.get("dirtyable"):
            return False, f"{o['objectType']} not dirtyable"
        if action == "cook" and not o.get("cookable"):
            return False, f"{o['objectType']} not cookable"
        if action in {"open", "close"} and not o.get("openable"):
            return False, f"{o['objectType']} not openable"
        if action in {"fill", "empty"} and not o.get("canFillWithLiquid"):
            return False, f"{o['objectType']} cannot contain liquid"
    if t in {"occlude", "reveal"}:
        cover = spec.get("cover")
        cover_id = spec.get("cover_id")
        c = next((
            x for x in event.metadata["objects"]
            if x["objectId"] == cover_id
        ), None) if cover_id else next((
            x for x in event.metadata["objects"] if x["objectType"] == cover
        ), None)
        if c is None or not c.get("pickupable"):
            return False, f"cover {cover} missing or not pickupable"
    if t == "identity_swap":
        other_id = spec.get("other_object_id")
        other = next((
            item for item in event.metadata["objects"]
            if item["objectId"] == other_id
        ), None)
        if not o.get("pickupable") or not other or not other.get("pickupable"):
            return False, "identity swap requires two pickupable objects"
    return True, "ok"


class Event:
    """One configured change event. Holds the resolved target id after setup."""

    def __init__(self, spec, index):
        self.spec = spec
        self.index = index
        self.type = spec["type"]
        self.event_family, self.event_action = event_labels(spec)
        self.target_id = None       # resolved at bind()
        self.cover_id = None
        self.secondary_target_id = None
        self.before = None
        self.after = None
        self.related_before = None
        self.related_after = None
        self.status = "pending"     # pending -> applied / invalid / failed
        self.note = ""
        self.trigger_frame = None   # scheduled Pass-B frame index
        self.no_witness = None      # set in validation: True if change unseen
        self.gaze_index = None      # index into the loop's gaze-point list
        self.allowed_receptacle_ids = None
        self.witness_before_frame = None
        self.witness_after_frame = None
        self.result_object_ids = []

    def bind(self, event):
        """Resolve object ids against the live scene (before any loop)."""
        o = _find_object(event, self.spec)
        self.target_id = o["objectId"] if o else None
        if self.type in {"occlude", "reveal"}:
            cover = self.spec.get("cover")
            cover_id = self.spec.get("cover_id")
            c = next((x for x in event.metadata["objects"]
                      if x["objectId"] == cover_id), None) if cover_id else next((
                          x for x in event.metadata["objects"]
                          if x["objectType"] == cover), None)
            self.cover_id = c["objectId"] if c else None
        if self.type == "identity_swap":
            self.secondary_target_id = self.spec.get("other_object_id")

    def affected_object_ids(self):
        if self.type == "identity_swap":
            return [
                object_id for object_id in
                (self.target_id, self.secondary_target_id) if object_id
            ]
        return [self.target_id] if self.target_id else []

    def _state_pre_setup(self, controller):
        action = self.spec.get("action")
        target = next((
            obj for obj in controller.last_event.metadata["objects"]
            if obj["objectId"] == self.target_id
        ), None)
        if target is None:
            return False
        command = None
        kwargs = {}
        if action == "open" and target.get("isOpen"):
            command = "CloseObject"
        elif action == "close" and not target.get("isOpen"):
            command, kwargs = "OpenObject", {"openness": 1.0}
        elif action == "dirty" and target.get("isDirty"):
            command = "CleanObject"
        elif action == "clean" and not target.get("isDirty"):
            command = "DirtyObject"
        elif action == "fill" and target.get("isFilledWithLiquid"):
            command = "EmptyLiquidFromObject"
        elif action == "empty" and not target.get("isFilledWithLiquid"):
            command, kwargs = "FillObjectWithLiquid", {"fillLiquid": "water"}
        elif action == "toggle_on" and target.get("isToggled"):
            command = "ToggleObjectOff"
        elif action == "toggle_off" and not target.get("isToggled"):
            command = "ToggleObjectOn"
        if command is None:
            return True
        result = controller.step(
            action=command, objectId=self.target_id,
            forceAction=True, **kwargs)
        return bool(result.metadata["lastActionSuccess"])

    def pre_setup(self, controller):
        """Construct the Visit-1 state before planning or capture."""
        if self.type == "appear" and self.target_id:
            controller.step(action="DisableObject", objectId=self.target_id)
            self.note = "disabled before loop1"
        elif self.type == "state_change" and self.target_id:
            if not self._state_pre_setup(controller):
                self.note = "state precondition failed"
        elif self.type == "reveal" and self.cover_id:
            cover_position = self.spec.get("cover_position")
            if cover_position:
                result = controller.step(
                    action="PlaceObjectAtPoint",
                    objectId=self.cover_id,
                    position=cover_position,
                )
                if not result.metadata["lastActionSuccess"]:
                    self.note = "reveal precondition placement failed"

    def apply(self, controller, rng=None, strict=False):
        """Runs during intervention (camera off). Performs the mutation."""
        t = self.type
        try:
            if t == "move":
                # Physically relocate onto a different receptacle surface (never
                # floating). Avoid the object's current receptacle so it visibly
                # moves elsewhere.
                destination = self.spec.get("destination_position")
                if destination:
                    e = controller.step(
                        action="PlaceObjectAtPoint",
                        objectId=self.target_id,
                        position=destination,
                    )
                    ok = e.metadata["lastActionSuccess"]
                    rec_id = self.spec.get("destination_receptacle_id")
                    pos = destination
                elif strict:
                    ok, rec_id, pos = False, None, None
                else:
                    cur = next((o for o in controller.last_event.metadata["objects"]
                                if o["objectId"] == self.target_id), None)
                    cur_rec = (cur.get("parentReceptacles") or [None])[0] if cur else None
                    ok, rec_id, pos = _place_on_receptacle(
                        controller, self.target_id, rng, avoid_receptacle=cur_rec,
                        allowed_receptacle_ids=self.allowed_receptacle_ids)
                self.note = f"moved onto {rec_id} @ {pos}" if ok else "no valid receptacle placement"
            elif t == "disappear":
                e = controller.step(action="DisableObject", objectId=self.target_id)
                ok = e.metadata["lastActionSuccess"]
                self.note = "disabled" if ok else "DisableObject failed"
            elif t == "appear":
                e = controller.step(action="EnableObject", objectId=self.target_id)
                ok = e.metadata["lastActionSuccess"]
                # Verify the enabled object actually re-registers under our id.
                live_ids = {o["objectId"] for o in e.metadata["objects"]}
                if self.target_id not in live_ids:
                    # id changed on enable; try to re-resolve by objectType
                    ot = self.spec.get("object")
                    match = next((o for o in e.metadata["objects"]
                                  if o["objectType"] == ot), None)
                    if match:
                        self.note = (f"enabled; id changed "
                                     f"{self.target_id} -> {match['objectId']}")
                        self.target_id = match["objectId"]
                        ok = True
                    else:
                        self.note = "enabled but object id not found after enable"
                        ok = False
                else:
                    self.note = "enabled" if ok else "EnableObject failed"
            elif t == "state_change":
                action = self.spec.get("action", "slice")
                action_map = {
                    "slice": "SliceObject",
                    "dirty": "DirtyObject",
                    "clean": "CleanObject",
                    "cook": "CookObject",
                    "break": "BreakObject",
                    "open": "OpenObject",
                    "close": "CloseObject",
                    "fill": "FillObjectWithLiquid",
                    "empty": "EmptyLiquidFromObject",
                }
                before_ids = {
                    obj["objectId"]
                    for obj in controller.last_event.metadata["objects"]
                }
                if action in {"toggle", "toggle_on", "toggle_off"}:
                    target = next((
                        obj for obj in controller.last_event.metadata["objects"]
                        if obj["objectId"] == self.target_id
                    ), None)
                    desired = self.spec.get("desired_toggle_state")
                    if action == "toggle_on":
                        desired = True
                    elif action == "toggle_off":
                        desired = False
                    if desired is None:
                        desired = not bool(target and target.get("isToggled"))
                    thor_action = "ToggleObjectOn" if desired else "ToggleObjectOff"
                    # Interventions are deliberately applied while the target is
                    # behind the camera, so visibility-based interaction checks
                    # must not make an otherwise valid state transition fail.
                    e = controller.step(
                        action=thor_action,
                        objectId=self.target_id,
                        forceAction=True,
                    )
                else:
                    thor_action = action_map.get(action)
                    if thor_action is None:
                        raise ValueError(f"unsupported state action: {action}")
                    kwargs = {}
                    if action == "open":
                        kwargs["openness"] = 1.0
                    elif action == "fill":
                        kwargs["fillLiquid"] = "water"
                    e = controller.step(
                        action=thor_action, objectId=self.target_id,
                        forceAction=True, **kwargs)
                ok = e.metadata["lastActionSuccess"]
                if ok and action in {"slice", "break"}:
                    self.result_object_ids = [
                        obj["objectId"] for obj in e.metadata["objects"]
                        if obj["objectId"] not in before_ids
                    ]
                self.note = (
                    f"{action}" if ok else
                    f"{thor_action} failed: "
                    f"{e.metadata.get('errorMessage', 'unknown simulator error')}"
                )
            elif t == "occlude":
                # Place the cover onto the target's own receptacle, at the
                # target's position, so it rests on the surface over the target.
                by_id = {o["objectId"]: o for o in controller.last_event.metadata["objects"]}
                tgt = by_id.get(self.target_id)
                ok = False
                cover_position = self.spec.get("cover_position")
                if tgt and (cover_position is not None or not strict):
                    e = controller.step(action="PlaceObjectAtPoint",
                                        objectId=self.cover_id,
                                        position=cover_position or tgt["position"])
                    ok = e.metadata["lastActionSuccess"]
                    if not ok and cover_position is None and not strict:
                        # fall back to any receptacle placement near the target
                        tgt_rec = (tgt.get("parentReceptacles") or [None])[0]
                        ok, rec_id, _ = _place_on_receptacle(
                            controller, self.cover_id, rng,
                            prefer_types=(tgt["objectType"],)) if tgt_rec else (False, None, None)
                self.note = "cover placed over target" if ok else "occlude placement failed"
            elif t == "reveal":
                destination = self.spec.get("reveal_destination_position")
                if destination and self.cover_id:
                    e = controller.step(
                        action="PlaceObjectAtPoint",
                        objectId=self.cover_id,
                        position=destination,
                    )
                    ok = e.metadata["lastActionSuccess"]
                else:
                    ok = False
                self.note = "cover removed from target" if ok else "reveal placement failed"
            elif t == "identity_swap":
                by_id = {
                    obj["objectId"]: obj
                    for obj in controller.last_event.metadata["objects"]
                }
                first = by_id.get(self.target_id)
                second = by_id.get(self.secondary_target_id)
                ok = bool(first and second)
                if ok:
                    first_position = (
                        self.spec.get("first_destination_position")
                        if strict else second["position"])
                    second_position = (
                        self.spec.get("second_destination_position")
                        if strict else first["position"])
                    first_rotation = (
                        self.spec.get("first_destination_rotation")
                        if strict else second["rotation"])
                    second_rotation = (
                        self.spec.get("second_destination_rotation")
                        if strict else first["rotation"])
                    if strict and any(value is None for value in (
                            first_position, second_position,
                            first_rotation, second_rotation)):
                        ok = False
                if ok:
                    first_result = controller.step(
                        action="TeleportObject",
                        objectId=self.target_id,
                        position=first_position,
                        rotation=first_rotation,
                        forceAction=True,
                    )
                    second_result = controller.step(
                        action="TeleportObject",
                        objectId=self.secondary_target_id,
                        position=second_position,
                        rotation=second_rotation,
                        forceAction=True,
                    )
                    ok = bool(
                        first_result.metadata["lastActionSuccess"]
                        and second_result.metadata["lastActionSuccess"])
                self.note = "swapped object positions" if ok else "identity swap failed"
            elif t == "no_change":
                ok = True
                self.note = "no intervention"
            else:
                ok, self.note = False, f"unknown event type {t}"
            self.status = "applied" if ok else "failed"
        except Exception as ex:  # noqa: BLE001
            self.status = "failed"
            self.note = f"exception: {ex}"
        return self.status == "applied"

    def highlight_object_ids(self):
        """Rendered instances that visualize this event's post-change state."""
        if (self.status != "applied"
                or self.type in {"disappear", "no_change"}):
            return []
        if self.type == "state_change" and self.result_object_ids:
            return list(self.result_object_ids)
        if self.type == "occlude":
            return [oid for oid in (self.target_id, self.cover_id) if oid]
        if self.type == "identity_swap":
            return self.affected_object_ids()
        return [self.target_id] if self.target_id else []

    def to_dict(self):
        if self.type == "identity_swap":
            targets = [
                {"object_id": self.target_id, "role": "first"},
                {"object_id": self.secondary_target_id, "role": "second"},
            ]
        else:
            targets = [{"object_id": self.target_id, "role": "target"}]
        conclusion = "unknown"
        if self.type == "no_change" and getattr(self, "final_state_seen", False):
            conclusion = "unchanged"
        elif self.status == "applied" and getattr(self, "final_state_seen", False):
            conclusion = "changed"
        return {
            "index": self.index,
            "type": self.type,
            "event_family": self.event_family,
            "event_action": self.event_action,
            "spec": self.spec,
            "targets": targets,
            "target_id": self.target_id,
            "cover_id": self.cover_id,
            "secondary_target_id": self.secondary_target_id,
            "status": self.status,
            "note": self.note,
            "trigger_frame": self.trigger_frame,
            "witness_before_frame": self.witness_before_frame,
            "witness_after_frame": self.witness_after_frame,
            "no_witness": self.no_witness,
            "final_state_seen": getattr(self, "final_state_seen", None),
            "result_object_ids": self.result_object_ids,
            "highlight_object_ids": self.highlight_object_ids(),
            "before": self.before,
            "after": self.after,
            "related_before": self.related_before,
            "related_after": self.related_after,
            "observation": {
                "witness_before_frame": self.witness_before_frame,
                "intervention_frame": self.trigger_frame,
                "witness_after_frame": self.witness_after_frame,
                "no_witness": self.no_witness,
                "perceptible": getattr(self, "final_state_seen", None),
                "conclusion": conclusion,
            },
        }
