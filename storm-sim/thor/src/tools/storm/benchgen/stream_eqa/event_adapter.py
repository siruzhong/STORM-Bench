"""Convert simulator-grounded Benchgen events into StreamEQA events."""
from __future__ import annotations

from typing import Any


OBJECT_LABELS = {
    "Apple": "苹果",
    "Bowl": "碗",
    "Bread": "面包",
    "Bottle": "瓶子",
    "ButterKnife": "黄油刀",
    "Cup": "杯子",
    "Egg": "鸡蛋",
    "Fork": "叉子",
    "Kettle": "水壶",
    "Knife": "刀",
    "Ladle": "汤勺",
    "Lettuce": "生菜",
    "Mug": "马克杯",
    "Pan": "平底锅",
    "PepperShaker": "胡椒瓶",
    "Plate": "盘子",
    "Potato": "土豆",
    "SaltShaker": "盐瓶",
    "SoapBottle": "洗洁精瓶",
    "Spatula": "锅铲",
    "Spoon": "勺子",
    "Tomato": "番茄",
}

SURFACE_LABELS = {
    "CounterTop": "操作台",
    "DiningTable": "餐桌",
    "StoveBurner": "灶台",
    "CoffeeTable": "茶几",
    "SideTable": "边桌",
    "Chair": "椅子",
    "Sofa": "沙发",
    "TVStand": "电视柜",
    "Bed": "床",
    "Desk": "书桌",
    "Dresser": "斗柜",
    "Shelf": "置物架",
    "Bathtub": "浴缸",
    "BathtubBasin": "浴缸内",
    "Floor": "地面",
    "HandTowelHolder": "擦手巾架",
    "Sink": "洗手池",
    "SinkBasin": "水槽内",
    "TowelHolder": "毛巾架",
}

EVENT_TYPE_MAP = {"appear": "appearance", "disappear": "disappearance"}


def object_label(object_type: str) -> str:
    return OBJECT_LABELS.get(object_type, object_type)


def _surface_label(station: dict[str, Any]) -> str:
    surface_type = str(station.get("surface_type") or "承载面")
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
        location = f"{plan['recipe']['scene']} 场景的{surface}上"
        if event_type == "disappearance":
            before_state = f"{label}位于{surface}上并清晰可见"
            after_state = f"原本放在{surface}上的{label}已经不在画面中"
            description = (
                f"镜头短暂移开后再次回到{surface}，原本放在那里的"
                f"{label}已经不见了。"
            )
        else:
            before_state = f"画面中暂时看不到原本位于{surface}上的{label}"
            after_state = f"{label}重新出现在{surface}的原位置"
            description = (
                f"镜头再次回到{surface}时，先前不见的{label}又出现在"
                "原来的位置。"
            )
        cues = [f"物体为{label}", f"所在位置为{surface}"]
        if event_type == "disappearance":
            cues.append(f"消失前可以在{surface}的原位置清楚看到{label}")
        else:
            cues.append(f"重新出现的位置与{label}消失前的位置一致")
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
