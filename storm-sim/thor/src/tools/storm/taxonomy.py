"""Canonical STORM event families and backward-compatible action labels."""
from __future__ import annotations


EVENT_FAMILIES = (
    "movement",
    "presence_change",
    "occlusion",
    "state_change",
    "identity_swap",
    "quantity_change",
    "no_change",
)

STATE_ACTIONS = (
    "slice",
    "break",
    "dirty",
    "clean",
    "cook",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "fill",
    "empty",
)

FAMILY_ACTIONS = {
    "movement": ("move", "put_in", "take_out", "stack", "unstack"),
    "presence_change": ("appear", "disappear"),
    "occlusion": ("occlude", "reveal"),
    "state_change": STATE_ACTIONS,
    "identity_swap": ("swap_positions", "replace_with_similar"),
    "quantity_change": ("add_instance", "remove_instance"),
    "no_change": ("none",),
}


def event_labels(spec):
    """Return `(event_family, event_action)` for old or v2 event specs."""
    if spec.get("event_family") and spec.get("event_action"):
        return spec["event_family"], spec["event_action"]
    event_type = spec.get("type")
    if event_type == "move":
        return "movement", "move"
    if event_type in {"appear", "disappear"}:
        return "presence_change", event_type
    if event_type in {"occlude", "reveal"}:
        return "occlusion", event_type
    if event_type == "state_change":
        action = spec.get("action", "slice")
        if action == "toggle":
            desired = spec.get("desired_toggle_state")
            action = "toggle_on" if desired is not False else "toggle_off"
        return "state_change", action
    if event_type == "identity_swap":
        return "identity_swap", spec.get("action", "swap_positions")
    if event_type == "quantity_change":
        return "quantity_change", spec.get("action", "add_instance")
    if event_type == "no_change":
        return "no_change", "none"
    raise ValueError(f"unknown event type: {event_type}")


def event_request(event_family=None, event_action=None):
    """Translate a v2 family/action request into legacy executor filters."""
    if event_family is None:
        return None, None
    if event_family not in FAMILY_ACTIONS:
        raise ValueError(f"unknown event family: {event_family}")
    if event_action and event_action not in FAMILY_ACTIONS[event_family]:
        raise ValueError(
            f"event action {event_action} does not belong to {event_family}")
    action = event_action
    if event_family == "movement":
        if action not in {None, "move"}:
            raise ValueError(f"event action not implemented yet: {action}")
        return ["move"], None
    if event_family == "presence_change":
        return [action] if action else ["appear", "disappear"], None
    if event_family == "occlusion":
        return [action] if action else ["occlude", "reveal"], None
    if event_family == "state_change":
        return ["state_change"], [action] if action else list(STATE_ACTIONS)
    if event_family == "identity_swap":
        if action not in {None, "swap_positions"}:
            raise ValueError(f"event action not implemented yet: {action}")
        return ["identity_swap"], None
    if event_family == "no_change":
        return ["no_change"], None
    raise ValueError(f"event family not implemented yet: {event_family}")
