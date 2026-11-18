"""Detector tests with synthetic EAR signals at 20 FPS (no camera needed)."""
import pytest

from drowsiness.detector import DetectorConfig, DrowsinessDetector, Status

FPS = 20
DT = 1 / FPS


def run(det, ear, seconds, t0, mar=0.1):
    t = t0
    for _ in range(int(round(seconds * FPS))):
        det.update(ear, mar, t)
        t += DT
    return t


def fixed(**kw):
    kw.setdefault("auto_calibrate", False)
    kw.setdefault("smoothing", 0.0)
    return DrowsinessDetector(DetectorConfig(**kw))


def test_open_eyes_stay_awake():
    det = fixed()
    run(det, 0.30, 5, 0)
    assert det.state.status == Status.AWAKE
    assert det.state.blinks == 0
    assert not det.state.alarm


def test_blink_is_counted_not_alarmed():
    det = fixed()
    t = run(det, 0.30, 1, 0)
    t = run(det, 0.10, 0.2, t)   # 200 ms blink
    run(det, 0.30, 1, t)
    assert det.state.blinks == 1
    assert det.state.alarm_events == 0


def test_long_closure_triggers_alarm_once():
    det = fixed(drowsy_seconds=1.5)
    t = run(det, 0.30, 1, 0)
    t = run(det, 0.10, 2.0, t)
    assert det.state.alarm and det.state.status == Status.DROWSY
    run(det, 0.30, 1, t)
    assert not det.state.alarm
    assert det.state.alarm_events == 1
    assert det.state.blinks == 0  # long closure is not a blink


def lose_face(det, seconds, t):
    for _ in range(int(round(seconds * FPS))):
        det.update(None, None, t)
        t += DT
    return t


def test_turning_away_with_open_eyes_does_not_alarm():
    det = fixed(drowsy_seconds=1.5)
    t = run(det, 0.30, 1.0, 0)
    t = lose_face(det, 3.0, t)
    assert det.state.status == Status.NO_FACE
    assert det.state.alarm_events == 0
    run(det, 0.10, 1.0, t)  # timer restarted, so 1 s of closure is not enough
    assert not det.state.alarm


def test_face_lost_while_eyes_closed_keeps_alarming():
    # Head drops forward: eyes were closed, then the face disappears.
    det = fixed(drowsy_seconds=1.5)
    t = run(det, 0.30, 1.0, 0)
    t = run(det, 0.10, 1.0, t)
    t = lose_face(det, 1.0, t)
    assert det.state.alarm and det.state.status == Status.DROWSY
    assert det.state.alarm_events == 1


def test_jitter_frame_is_not_a_blink():
    det = fixed()
    t = run(det, 0.30, 1, 0)
    det.update(0.10, 0.1, t)          # a single 50 ms frame below threshold
    run(det, 0.30, 1, t + DT)
    assert det.state.blinks == 0


def test_perclos_is_independent_of_frame_rate():
    for fps in (10, 30):
        det = fixed(perclos_window=20, perclos_alarm=0.9)
        t = 0.0
        for _ in range(10):
            for ear in (0.30, 0.10):
                for _ in range(fps):
                    det.update(ear, 0.1, t)
                    t += 1 / fps
        assert det.state.perclos == pytest.approx(0.5, abs=0.05), fps


def test_calibration_uses_median_not_mean():
    det = DrowsinessDetector(DetectorConfig(auto_calibrate=True, smoothing=0.0, calibration_seconds=3))
    t = run(det, 0.30, 2.4, 0)
    t = run(det, 0.16, 0.6, t)        # blink-ish frames during calibration
    run(det, 0.30, 0.2, t)
    assert det.state.threshold == pytest.approx(0.75 * 0.30)


def test_auto_calibration_sets_personal_threshold():
    det = DrowsinessDetector(DetectorConfig(auto_calibrate=True, smoothing=0.0, calibration_seconds=3))
    t = run(det, 0.24, 2, 0)
    assert det.state.status == Status.CALIBRATING
    run(det, 0.24, 1.5, t)
    assert det.state.status == Status.AWAKE
    assert det.state.threshold == pytest.approx(0.18)


def test_calibration_threshold_is_clamped():
    det = DrowsinessDetector(DetectorConfig(auto_calibrate=True, smoothing=0.0))
    run(det, 0.60, 4, 0)
    assert det.state.threshold == pytest.approx(0.30)


def test_yawn_detection():
    det = fixed(mar_threshold=0.6, yawn_seconds=1.0)
    t = run(det, 0.30, 1, 0, mar=0.2)
    t = run(det, 0.30, 2, t, mar=0.8)   # one long yawn counts once
    t = run(det, 0.30, 1, t, mar=0.2)
    run(det, 0.30, 0.5, t, mar=0.8)     # too short to be a yawn
    assert det.state.yawns == 1


def test_perclos_alarm_from_frequent_short_closures():
    # Eyes closed 1 s out of every 2 s: never long enough for the closure alarm,
    # but PERCLOS (~50%) should still catch the fatigue.
    det = fixed(drowsy_seconds=1.5, perclos_window=20, perclos_alarm=0.3)
    t = 0
    for _ in range(10):
        t = run(det, 0.30, 1, t)
        t = run(det, 0.10, 1, t)
    assert det.state.perclos == pytest.approx(0.5, abs=0.05)
    assert det.state.alarm_events >= 1
