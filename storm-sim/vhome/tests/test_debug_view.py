"""Check that debug overlays never expose an occluded target."""
import numpy as np
from storm_virtualhome.debug_view import highlight, target_mask


def test_hidden_target_does_not_draw_through_occluder():
    rgb = np.full((160, 200, 3), 70, dtype=np.uint8)
    mask = np.zeros((160, 200), dtype=bool)
    result = highlight(rgb, mask, 'bowl #326', 'Close fridge', 1.0)
    assert result.shape == (224, 200, 3)
    assert np.array_equal(result[64:], rgb)


def test_only_target_color_matches_and_gets_highlighted():
    rgb = np.full((160, 200, 3), 70, dtype=np.uint8)
    seg = np.zeros_like(rgb)
    seg[100:110, 80:90] = [255, 128, 0]
    mask = target_mask(seg, [1.0, 0.5, 0])
    assert mask.sum() == 100
    result = highlight(rgb, mask, 'bowl #326', 'Grab', 1.0)
    assert result[169, 85, 0] > rgb[105, 85, 0]
    assert np.array_equal(result[204, 140], rgb[140, 140])


def test_collision_audit_rejects_even_one_overlap(tmp_path):
    import pytest
    from storm_virtualhome.debug_view import camera_audit
    (tmp_path / 'logs').mkdir()
    trace = tmp_path / 'logs/camera_collision.csv'
    trace.write_text('frame,blocked,overlaps\n1,1,0\n2,1,1\n')
    with pytest.raises(RuntimeError):
        camera_audit(tmp_path)
    trace.write_text('frame,blocked,overlaps\n1,1,0\n2,0,0\n')
    assert camera_audit(tmp_path) == {'camera_samples': 2, 'camera_retracted_samples': 1, 'camera_overlap_samples': 0}
