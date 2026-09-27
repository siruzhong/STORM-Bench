"""Catch camera cuts, including cuts at action boundaries."""
import math

import pytest

from storm_virtualhome.continuity import camera_metrics, check_limits

RULES = {'max_speed_mps': 3.5, 'max_turn_rate_dps': 120}


def row(x, angle, action='walk'):
    return {'action': action, 'camera_pose': {'position': [x, 2.4, 0],
            'rotation': [0, math.sin(math.radians(angle) / 2), 0, math.cos(math.radians(angle) / 2)], 'fov': 75}}


def test_continuous_motion_and_quaternion_sign_are_valid():
    rows = [row(0, 0), row(.1, 4), row(.2, 8, 'turn')]
    rows[1]['camera_pose']['rotation'] = [-x for x in rows[1]['camera_pose']['rotation']]
    report = camera_metrics(rows, 20)
    check_limits(report, RULES)
    assert report['action_boundaries_checked'] == 1
    assert report['max_boundary_rotation_degrees'] == pytest.approx(4)
    assert report['p95_angular_acceleration_dps2'] == pytest.approx(0, abs=1e-8)
    assert report['p95_linear_acceleration_mps2'] == pytest.approx(0)


@pytest.mark.parametrize('second,reason', [(row(1, 0, 'new action'), 'translation'), (row(0, 30, 'new action'), 'rotation')])
def test_boundary_cuts_are_not_exempt(second, reason):
    with pytest.raises(ValueError, match=reason):
        check_limits(camera_metrics([row(0, 0), second], 20), RULES)


def test_lens_change_is_rejected():
    rows = [row(0, 0), row(0, 0)]
    rows[1]['camera_pose']['fov'] = 90
    with pytest.raises(ValueError, match='focal'):
        check_limits(camera_metrics(rows, 20), RULES)


def test_missing_or_invalid_pose_is_rejected():
    with pytest.raises(ValueError, match='synchronized'):
        camera_metrics([row(0, 0), {}], 20)
    with pytest.raises(ValueError, match='values'):
        camera_metrics([row(0, 0), row(float('nan'), 0)], 20)


def test_actual_lens_must_match_the_configured_lens():
    with pytest.raises(ValueError, match='configured lens'):
        check_limits(camera_metrics([row(0, 0), row(0, 0)], 20), dict(RULES, fov_degrees=90))


def test_camera_must_remain_inside_the_safe_region():
    with pytest.raises(ValueError, match='safe region'):
        check_limits(camera_metrics([row(0, 0), row(.1, 0)], 20),
                     dict(RULES, position_min=[-1, 1, -1], position_max=[1, 2, 1]))


def test_smoothness_metrics_detect_direction_reversals_at_constant_speed():
    report = camera_metrics([row(0, 0), row(.1, 4), row(0, 0)], 20)
    assert report['p95_linear_acceleration_mps2'] == pytest.approx(80)
    assert report['p95_angular_acceleration_dps2'] == pytest.approx(3200)


def test_acceleration_and_boundary_limits_are_enforced():
    from storm_virtualhome.continuity import check_limits
    report=dict(position_min=[0,0,0],position_max=[0,0,0],fov_degrees=75,
                max_camera_speed_mps=1,max_camera_turn_rate_dps=1,max_fov_step_degrees=0,
                p95_linear_acceleration_mps2=9,p95_angular_acceleration_dps2=100,
                max_boundary_rotation_degrees=1)
    with pytest.raises(ValueError,match='acceleration spike'):
        check_limits(report,dict(max_speed_mps=2,max_turn_rate_dps=2,
                                 max_p95_linear_acceleration_mps2=8))
