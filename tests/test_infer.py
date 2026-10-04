import numpy as np

from emovision.infer import FaceSmoother, iou


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - 1 / 3) < 1e-9


def test_smoother_tracks_moving_face_and_resets_new_ones():
    s = FaceSmoother(alpha=0.5)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    first = s.update([(0, 0, 100, 100)], np.array([a]))
    np.testing.assert_allclose(first[0], a)

    # Same face moved slightly: blended with history
    second = s.update([(5, 5, 100, 100)], np.array([b]))
    np.testing.assert_allclose(second[0], [0.5, 0.5])

    # Far-away face is new: no blending
    third = s.update([(500, 500, 100, 100)], np.array([b]))
    np.testing.assert_allclose(third[0], b)
