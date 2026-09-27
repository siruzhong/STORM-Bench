"""Domain-specific schemas, labels, and generation settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainSpec:
    name: str
    regions: frozenset[str]
    activities: frozenset[str]
    segmentation_schema: str
    qa_schema: str
    min_visits: int
    segmentation_retries: int
    region_priority: dict[str, float] = field(default_factory=dict)
    region_aliases: dict[str, str] = field(default_factory=dict)
    exact_uncertain_answer: bool = False
    qa_generation_attempts: int = 5


DOMAINS: dict[str, DomainSpec] = {
    "cook": DomainSpec(
        name="cook",
        regions=frozenset({
            "sink_area", "food_prep_area", "stove_area", "refrigerator_area",
            "cabinet_area", "other_area",
        }),
        activities=frozenset({
            "observe", "navigate", "retrieve", "place", "clean", "prepare",
            "cook", "open_close", "organize", "other_interaction", "unusable",
        }),
        segmentation_schema="stream_eqa_activity_segments_v6_en",
        qa_schema="storm_real_cook_qa_en_v1",
        min_visits=3,
        segmentation_retries=3,
    ),
    "bike": DomainSpec(
        name="bike",
        regions=frozenset({
            "front_wheel_area", "rear_wheel_area", "tire_tube_area", "chain_area",
            "rear_derailleur_area", "tools_supplies_area", "other_area",
        }),
        activities=frozenset({
            "observe", "navigate", "retrieve", "place", "inspect", "remove",
            "install", "loosen_tighten", "adjust", "clean", "lubricate",
            "inflate_deflate", "organize", "other_interaction", "unusable",
        }),
        segmentation_schema="ego_exo4d_bike_activity_segments_en_v1_no_entities",
        qa_schema="ego_exo4d_bike_qa_en_v1",
        min_visits=2,
        segmentation_retries=3,
        region_priority={
            "tire_tube_area": 6.0,
            "front_wheel_area": 5.0,
            "rear_wheel_area": 5.0,
            "chain_area": 5.0,
            "rear_derailleur_area": 5.0,
            "tools_supplies_area": 1.0,
        },
        exact_uncertain_answer=True,
    ),
    "health": DomainSpec(
        name="health",
        regions=frozenset({
            "patient_mannequin_area", "sample_collection_area", "test_kit_area",
            "instruction_record_area", "supplies_ppe_area", "other_area",
        }),
        activities=frozenset({
            "observe", "navigate", "prepare", "don_doff_ppe", "collect_sample",
            "transfer_sample", "add_reagent", "operate_test_kit",
            "perform_compressions", "ventilate", "check_response", "record_result",
            "clean_dispose", "retrieve", "place", "organize", "other_interaction",
            "unusable",
        }),
        segmentation_schema="ego_exo4d_health_activity_segments_en_v1_no_entities",
        qa_schema="ego_exo4d_health_qa_en_v1",
        min_visits=2,
        segmentation_retries=5,
        region_priority={
            "sample_collection_area": 7.0,
            "test_kit_area": 7.0,
            "patient_mannequin_area": 7.0,
            "supplies_ppe_area": 3.0,
            "instruction_record_area": 2.0,
        },
        exact_uncertain_answer=True,
        qa_generation_attempts=12,
    ),
    "music": DomainSpec(
        name="music",
        regions=frozenset({
            "instrument_playing_area", "score_reference_area",
            "controls_accessories_area", "ensemble_scene_area", "other_area",
        }),
        activities=frozenset({
            "observe", "navigate", "prepare", "play", "practice_phrase",
            "adjust_tune", "turn_page", "handle_instrument", "coordinate",
            "organize", "other_interaction", "unusable",
        }),
        segmentation_schema="ego_exo4d_music_activity_segments_en_v1_no_entities",
        qa_schema="ego_exo4d_music_qa_en_v1",
        min_visits=2,
        segmentation_retries=5,
        region_priority={
            "instrument_playing_area": 7.0,
            "score_reference_area": 5.0,
            "controls_accessories_area": 4.0,
            "ensemble_scene_area": 3.0,
        },
        exact_uncertain_answer=True,
        qa_generation_attempts=12,
    ),
    "sports": DomainSpec(
        name="sports",
        regions=frozenset({
            "basketball_court_area", "basketball_hoop_area",
            "basketball_ball_drill_area", "basketball_sideline_area",
            "soccer_field_area", "soccer_goal_area", "soccer_cone_drill_area",
            "soccer_ball_area", "soccer_sideline_area", "climbing_wall_route_area",
            "climbing_hold_sequence_area", "climbing_start_finish_area",
            "climbing_equipment_rest_area", "other_area",
        }),
        activities=frozenset({
            "observe", "navigate", "prepare", "dribble_control", "pass_receive",
            "shoot", "defend_mark", "run_position", "kick_control", "climb_move",
            "grip_foothold", "clip_manage_rope", "rest_reset", "retrieve", "place",
            "coordinate", "other_interaction", "unusable",
        }),
        segmentation_schema="ego_exo4d_sports_activity_segments_en_v1_no_entities",
        qa_schema="ego_exo4d_sports_qa_en_v1",
        min_visits=2,
        segmentation_retries=5,
        region_priority={
            "basketball_ball_drill_area": 7.0,
            "basketball_hoop_area": 6.0,
            "basketball_court_area": 5.0,
            "basketball_sideline_area": 2.0,
            "soccer_cone_drill_area": 7.0,
            "soccer_ball_area": 6.0,
            "soccer_goal_area": 5.0,
            "soccer_field_area": 4.0,
            "soccer_sideline_area": 2.0,
            "climbing_hold_sequence_area": 7.0,
            "climbing_wall_route_area": 6.0,
            "climbing_start_finish_area": 4.0,
            "climbing_equipment_rest_area": 2.0,
        },
        exact_uncertain_answer=True,
        qa_generation_attempts=12,
    ),
}

DOMAIN_NAMES = tuple(DOMAINS)


def get_domain(name: str = "cook") -> DomainSpec:
    """Resolve a domain name without importing video or API dependencies."""
    if not isinstance(name, str) or name.strip().casefold() not in DOMAINS:
        raise ValueError("Domain must be one of: " + ", ".join(DOMAIN_NAMES))
    return DOMAINS[name.strip().casefold()]
