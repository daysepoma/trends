from types import SimpleNamespace

import pytest

from main import POSE_LANDMARKS, POSE_MODEL_POINTS, estimate_head_pose, eye_aspect_ratio, is_looking

EAR, PITCH, YAW = 0.21, 18.0, 25.0


def landmarks(points: dict[int, tuple[float, float]], size: float = 100.0) -> list[SimpleNamespace]:
    """Landmark list indexable by id; `points` are pixel coords on a size x size image."""
    return [
        SimpleNamespace(x=points.get(i, (0.0, 0.0))[0] / size, y=points.get(i, (0.0, 0.0))[1] / size)
        for i in range(max(points) + 1)
    ]


EYE = (0, 1, 2, 3, 4, 5)


def test_eye_aspect_ratio_open_eye():
    # horizontal = 10, both verticals = 4 -> (4 + 4) / (2 * 10)
    lm = landmarks({0: (0, 5), 1: (3, 3), 2: (7, 3), 3: (10, 5), 4: (7, 7), 5: (3, 7)})
    assert eye_aspect_ratio(lm, EYE, 100, 100) == pytest.approx(0.4)


def test_eye_aspect_ratio_closed_eye_is_zero():
    lm = landmarks({0: (0, 5), 1: (3, 5), 2: (7, 5), 3: (10, 5), 4: (7, 5), 5: (3, 5)})
    assert eye_aspect_ratio(lm, EYE, 100, 100) == 0.0


def test_eye_aspect_ratio_degenerate_horizontal_returns_zero():
    lm = landmarks({i: (5, 5) for i in EYE})
    assert eye_aspect_ratio(lm, EYE, 100, 100) == 0.0


@pytest.mark.parametrize(
    "ear, pose, expected",
    [
        (0.30, (0.0, 0.0), True),
        (0.30, (PITCH, YAW), True),  # pose thresholds are inclusive
        (EAR, (0.0, 0.0), True),  # EAR exactly at threshold counts as open
        (0.10, (0.0, 0.0), False),  # eyes closed
        (0.30, (PITCH + 0.1, 0.0), False),  # head up/down
        (0.30, (-PITCH - 0.1, 0.0), False),
        (0.30, (0.0, YAW + 0.1), False),  # head turned sideways
        (0.30, (0.0, -YAW - 0.1), False),
        (None, (0.0, 0.0), False),  # no face detected
        (0.30, None, False),  # pose could not be solved
    ],
)
def test_is_looking(ear, pose, expected):
    assert is_looking(ear, pose, EAR, PITCH, YAW) is expected


def test_estimate_head_pose_frontal_face_is_near_zero():
    # Project the canonical model with identity rotation, 600 units from the camera.
    size, depth = 640, 600.0
    pixels = {
        idx: (size * x / (z + depth) + size / 2, size * y / (z + depth) + size / 2)
        for idx, (x, y, z) in zip(POSE_LANDMARKS, POSE_MODEL_POINTS)
    }
    pose = estimate_head_pose(landmarks(pixels, size), size, size)

    assert pose is not None
    pitch, yaw = pose
    assert abs(pitch) < 1.0
    assert abs(yaw) < 1.0
