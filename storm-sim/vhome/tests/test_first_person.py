"""Check physical first-person attachment independently of third-person framing."""
import pytest
from storm_virtualhome.recording import validate_first_person_pose


def test_head_height_camera_follows_recorded_observer():
    offset=validate_first_person_pose({'position':[3.2,1.6,4.1], 'observer_position':[3,0,4]})
    assert offset==pytest.approx(.2236068)


@pytest.mark.parametrize('camera', [[3,2.2,4],[4,1.6,4],[3,1.2,4],[3,float('nan'),4]])
def test_detached_or_invalid_camera_is_rejected(camera):
    with pytest.raises(ValueError):
        validate_first_person_pose({'position':camera,'observer_position':[3,0,4]})
