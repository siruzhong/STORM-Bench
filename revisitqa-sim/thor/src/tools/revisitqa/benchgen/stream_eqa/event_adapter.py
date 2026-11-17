"""Convert simulator-grounded Benchgen events into StreamEQA events."""
from __future__ import annotations

from typing import Any


OBJECT_LABELS = {
    "Apple": "apple",
    "Bowl": "bowl",
    "Bread": "bread",
    "Bottle": "bottle",
    "ButterKnife": "butter knife",
    "Cup": "cup",
    "Egg": "egg",
    "Fork": "fork",
    "Kettle": "kettle",
    "Knife": "knife",
    "Ladle": "ladle",
    "Lettuce": "lettuce",
    "Mug": "mug",
    "Pan": "pan",
    "PepperShaker": "pepper shaker",
    "Plate": "plate",
    "Potato": "potato",
    "SaltShaker": "salt shaker",
    "SoapBottle": "soap bottle",
    "Spatula": "spatula",
    "Spoon": "spoon",
    "Tomato": "tomato",
}

SURFACE_LABELS = {
    "CounterTop": "countertop",
    "DiningTable": "dining table",
    "StoveBurner": "stove burner",
    "CoffeeTable": "coffee table",
    "SideTable": "side table",
    "Chair": "chair",
    "Sofa": "sofa",
    "TVStand": "TV stand",
    "Bed": "bed",
    "Desk": "desk",
    "Dresser": "dresser",
    "Shelf": "shelf",
    "Bathtub": "bathtub",
    "BathtubBasin": "bathtub basin",
    "Floor": "floor",
    "HandTowelHolder": "hand-towel holder",
    "Sink": "sink",
    "SinkBasin": "sink basin",
    "TowelHolder": "towel holder",
}

EVENT_TYPE_MAP = {"appear": "appearance", "disappear": "disappearance"}


def object_label(object_type: str) -> str:
    return OBJECT_LABELS.get(object_type, object_type)


def _surface_label(station: dict[str, Any]) -> str:
    surface_type = str(station.get("surface_type") or "supporting surface")
    return SURFACE_LABELS.get(surface_type, surface_type)


def build_events(
    plan: dict[str, Any],
    trace: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build exact event spans and human-readable state changes."""
    if not trace.get("accepted"):
        raise ValueError("cannot export a rejected execution trace")
    opportunities = {
        int(item["index"]): item
        for item in plan["mobility"]["opportunities"]
    }
    executions = {int(item["index"]): item for item in trace["events"]}
    stations = {item["station_id"]: item for item in profile["stations"]}
    events = []
    for intent in sorted(
        plan["event_program"]["events"], key=lambda item: int(item["index"]),
    ):
        index = int(intent["index"])
        event_type = EVENT_TYPE_MAP.get(str(intent["event_type"]))
        if event_type is None:
            raise ValueError(f"unsupported Benchgen event: {intent['event_type']}")
        opportunity = opportunities[index]
        execution = executions[index]
        if not execution.get("applied") or not execution.get("observed"):
            raise ValueError(f"event {index} is not applied and observed")
        station = stations[str(intent["station_id"])]
        label = object_label(str(intent["target_type"]))
        surface = _surface_label(station)
        location = f"on the {surface} in {plan['recipe']['scene']}"
        if event_type == "disappearance":
            before_state = f"The {label} is clearly visible on the {surface}"
            after_state = f"The {label} is no longer visible at its original {surface} position"
            description = (
                f"After the camera briefly moves away and returns to the {surface}, "
                f"the {label} is no longer at its original position."
            )
        else:
            before_state = f"The {label} is temporarily absent from its original {surface} position"
            after_state = f"The {label} reappears at its original {surface} position"
            description = (
                f"When the camera returns to the {surface}, the missing {label} "
                "is visible again at its original position."
            )
        cues = [f"The object is a {label}", f"The location is the {surface}"]
        if event_type == "disappearance":
            cues.append(f"The {label} is clearly visible at its original {surface} position before disappearing")
        else:
            cues.append(f"The {label} reappears at the same position from which it disappeared")
        events.append({
            "start_sec": round(float(opportunity["before_time_s"]), 2),
            "end_sec": round(float(opportunity["after_time_s"]), 2),
            "event_type": event_type,
            "subject": label,
            "before_state": before_state,
            "after_state": after_state,
            "location": location,
            "identity_cues": cues,
            "certainty": "known",
            "description": description,
            "_target_id": str(intent["target_id"]),
            "_index": index,
            "_surface": surface,
        })
    return events


def public_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove simulator-only keys from the interchange payload."""
    return [
        {key: value for key, value in event.items() if not key.startswith("_")}
        for event in events
    ]
