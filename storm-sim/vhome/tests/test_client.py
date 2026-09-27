"""Check bounded process configuration without starting the simulator."""
import pytest

from storm_virtualhome.client import xdisplay_numbers


def test_xdisplay_pool_is_large_and_configurable():
    default = xdisplay_numbers({})
    assert (default.start, default.stop) == (190, 512)
    custom = xdisplay_numbers({'STORM_XDISPLAY_MIN': '300', 'STORM_XDISPLAY_STOP': '340'})
    assert list(custom) == list(range(300, 340))


@pytest.mark.parametrize('env', [
    {'STORM_XDISPLAY_MIN': '0'},
    {'STORM_XDISPLAY_MIN': '300', 'STORM_XDISPLAY_STOP': '300'},
    {'STORM_XDISPLAY_STOP': '70000'},
])
def test_xdisplay_pool_rejects_invalid_bounds(env):
    with pytest.raises(ValueError, match='invalid range'):
        xdisplay_numbers(env)
