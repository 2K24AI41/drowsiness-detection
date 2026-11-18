"""Eye / mouth geometry: Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

EAR (Soukupova & Cech, 2016) for six eye landmarks p1..p6:

        p2   p3
    p1          p4
        p6   p5

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

An open eye gives a roughly constant EAR (~0.25-0.35). When the eye closes,
EAR drops towards 0. Because it is a *ratio*, it does not depend on how far
the face is from the camera.
"""
from __future__ import annotations

import numpy as np

# MediaPipe Face Mesh (478-point) indices, ordered p1..p6 as in the diagram.
# "Right"/"left" are from the SUBJECT's point of view.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Inner-lip landmarks for MAR: left corner, right corner, top, bottom.
MOUTH = [78, 308, 13, 14]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Compute EAR from a (6, 2) array of points ordered p1..p6."""
    eye = np.asarray(eye, dtype=float)
    if eye.shape != (6, 2):
        raise ValueError(f"expected shape (6, 2), got {eye.shape}")
    p1, p2, p3, p4, p5, p6 = eye
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """Compute MAR from a (4, 2) array: left corner, right corner, top, bottom."""
    mouth = np.asarray(mouth, dtype=float)
    if mouth.shape != (4, 2):
        raise ValueError(f"expected shape (4, 2), got {mouth.shape}")
    left, right, top, bottom = mouth
    width = _dist(left, right)
    if width < 1e-6:
        return 0.0
    return _dist(top, bottom) / width


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """Convert MediaPipe normalised landmarks to an (N, 2) pixel array."""
    return np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=float)


def face_ratios(points: np.ndarray) -> dict:
    """Return EAR (left, right, mean) and MAR for a full (478, 2) landmark array."""
    left = eye_aspect_ratio(points[LEFT_EYE])
    right = eye_aspect_ratio(points[RIGHT_EYE])
    return {
        "ear_left": left,
        "ear_right": right,
        "ear": (left + right) / 2.0,
        "mar": mouth_aspect_ratio(points[MOUTH]),
    }
