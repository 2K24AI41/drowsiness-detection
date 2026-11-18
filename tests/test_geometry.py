import numpy as np
import pytest

from drowsiness.geometry import LEFT_EYE, MOUTH, RIGHT_EYE, eye_aspect_ratio, face_ratios, mouth_aspect_ratio


def make_eye(width=30.0, opening=9.0):
    """Synthetic eye ordered p1..p6 (corner, top, top, corner, bottom, bottom)."""
    return np.array([
        [0, 0],
        [width / 3, -opening / 2],
        [2 * width / 3, -opening / 2],
        [width, 0],
        [2 * width / 3, opening / 2],
        [width / 3, opening / 2],
    ])


def test_open_eye_ear():
    # vertical distances 9 + 9, horizontal 30 -> 18 / 60 = 0.3
    assert eye_aspect_ratio(make_eye()) == pytest.approx(0.3)


def test_closed_eye_ear_is_zero():
    assert eye_aspect_ratio(make_eye(opening=0)) == pytest.approx(0.0)


def test_ear_is_scale_invariant():
    assert eye_aspect_ratio(make_eye() * 5) == pytest.approx(eye_aspect_ratio(make_eye()))


def test_degenerate_eye_does_not_crash():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        eye_aspect_ratio(np.zeros((5, 2)))


def test_mar():
    mouth = np.array([[0, 0], [40, 0], [20, -10], [20, 10]])
    assert mouth_aspect_ratio(mouth) == pytest.approx(0.5)


def test_face_ratios_uses_correct_indices():
    pts = np.zeros((478, 2))
    pts[RIGHT_EYE] = make_eye(opening=9) + [100, 100]
    pts[LEFT_EYE] = make_eye(opening=6) + [200, 100]
    pts[MOUTH] = [[0, 0], [40, 0], [20, -10], [20, 10]]
    r = face_ratios(pts)
    assert r["ear_right"] == pytest.approx(0.3)
    assert r["ear_left"] == pytest.approx(0.2)
    assert r["ear"] == pytest.approx(0.25)
    assert r["mar"] == pytest.approx(0.5)
