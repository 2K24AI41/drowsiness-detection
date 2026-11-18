"""Drowsiness logic: turns a stream of EAR/MAR values into blinks, yawns and alarms.

The detector is deliberately pure Python (no OpenCV / MediaPipe) so it can be
unit-tested with synthetic signals. It is *time based*, not frame based, so it
behaves the same at 10 FPS on a laptop CPU and at 30 FPS on a faster machine.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    CALIBRATING = "CALIBRATING"
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    NO_FACE = "NO FACE"


@dataclass
class DetectorConfig:
    ear_threshold: float = 0.21          # eyes count as closed below this
    drowsy_seconds: float = 1.5          # eyes closed this long -> alarm
    mar_threshold: float = 0.6           # mouth open above this -> yawn candidate
    yawn_seconds: float = 1.0            # mouth open this long -> one yawn
    smoothing: float = 0.5               # EMA factor for EAR (0 = off, closer to 1 = smoother)
    perclos_window: float = 60.0         # seconds of history used for PERCLOS
    perclos_alarm: float = 0.30          # alarm if eyes closed >30% of the window
    auto_calibrate: bool = True          # learn the user's open-eye EAR first
    calibration_seconds: float = 3.0
    calibration_ratio: float = 0.75      # threshold = ratio * median open-eye EAR
    blink_min_seconds: float = 0.07      # shorter dips are landmark jitter, not blinks
    blink_max_seconds: float = 0.5       # longer closures are not blinks (but may not be alarms yet)


@dataclass
class DetectorState:
    status: Status = Status.CALIBRATING
    ear: float = 0.0
    mar: float = 0.0
    threshold: float = 0.21
    eyes_closed: bool = False
    closed_for: float = 0.0
    blinks: int = 0
    yawns: int = 0
    perclos: float = 0.0
    alarm: bool = False
    alarm_events: int = 0


@dataclass
class DrowsinessDetector:
    config: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        cfg = self.config
        self.state = DetectorState(
            status=Status.CALIBRATING if cfg.auto_calibrate else Status.AWAKE,
            threshold=cfg.ear_threshold,
        )
        self._ema: float | None = None
        self._closed_since: float | None = None
        self._mouth_open_since: float | None = None
        self._yawn_counted = False
        self._calib_start: float | None = None
        self._calib_values: list[float] = []
        self._calibrated = not cfg.auto_calibrate
        self._history: deque[tuple[float, bool]] = deque()

    def update(self, ear: float | None, mar: float | None, t: float) -> DetectorState:
        """Feed one frame. Pass ear=None when no face was found. `t` is in seconds."""
        s = self.state
        if ear is None:
            self._ema = None  # don't smooth across a gap with a stale value
            self._mouth_open_since = None
            if self._calibrated and self._closed_since is not None:
                # The face vanished while the eyes were closed (e.g. the head dropped
                # forward). That looks like sleep, so the closure timer keeps running.
                s.closed_for = t - self._closed_since
                s.eyes_closed = True
                self._update_perclos(t)
            else:
                # Eyes were open: the driver just turned away, so reset instead of alarming.
                self._closed_since = None
                s.eyes_closed, s.closed_for = False, 0.0
            self._set_alarm()
            if not s.alarm:
                s.status = Status.NO_FACE
            return s

        self._ema = ear if self._ema is None else (
            self.config.smoothing * self._ema + (1 - self.config.smoothing) * ear
        )
        s.ear = self._ema
        s.mar = mar or 0.0

        if not self._calibrated:
            self._calibrate(self._ema, t)
            return s

        self._update_eyes(t)
        self._update_mouth(t)
        self._update_perclos(t)
        self._set_alarm()
        return s

    # ------------------------------------------------------------ internals
    def _set_alarm(self) -> None:
        s, cfg = self.state, self.config
        was_alarm = s.alarm
        s.alarm = s.closed_for >= cfg.drowsy_seconds or s.perclos >= cfg.perclos_alarm
        if s.alarm and not was_alarm:
            s.alarm_events += 1
        s.status = Status.DROWSY if s.alarm else Status.AWAKE

    def _calibrate(self, ear: float, t: float) -> None:
        cfg, s = self.config, self.state
        if self._calib_start is None:
            self._calib_start = t
        # Ignore obvious blinks while calibrating.
        if ear > 0.15:
            self._calib_values.append(ear)
        if t - self._calib_start >= cfg.calibration_seconds and self._calib_values:
            # Median, so blinks / half-closed frames during calibration don't drag it down.
            typical_open = statistics.median(self._calib_values)
            # Clamp so a bad calibration can never produce a silly threshold.
            s.threshold = min(max(cfg.calibration_ratio * typical_open, 0.12), 0.30)
            self._calibrated = True
            s.status = Status.AWAKE
        else:
            s.status = Status.CALIBRATING

    def _update_eyes(self, t: float) -> None:
        cfg, s = self.config, self.state
        closed = s.ear < s.threshold
        if closed:
            if self._closed_since is None:
                self._closed_since = t
            s.closed_for = t - self._closed_since
        else:
            # A short closure that ends is a blink; a long one was a drowsy episode.
            if self._closed_since is not None:
                duration = t - self._closed_since
                if cfg.blink_min_seconds <= duration <= cfg.blink_max_seconds:
                    s.blinks += 1
            self._closed_since = None
            s.closed_for = 0.0
        s.eyes_closed = closed

    def _update_mouth(self, t: float) -> None:
        cfg, s = self.config, self.state
        if s.mar > cfg.mar_threshold:
            if self._mouth_open_since is None:
                self._mouth_open_since = t
                self._yawn_counted = False
            if not self._yawn_counted and t - self._mouth_open_since >= cfg.yawn_seconds:
                s.yawns += 1
                self._yawn_counted = True
        else:
            self._mouth_open_since = None

    def _update_perclos(self, t: float) -> None:
        """PERCLOS = fraction of recent time with eyes closed (a standard fatigue metric)."""
        cfg, s = self.config, self.state
        self._history.append((t, s.eyes_closed))
        while self._history and t - self._history[0][0] > cfg.perclos_window:
            self._history.popleft()
        # Only trust PERCLOS once we have at least a quarter of the window.
        span = t - self._history[0][0]
        if span < cfg.perclos_window * 0.25:
            s.perclos = 0.0
            return
        # Time-weighted: each sample's state lasts until the next sample, so the
        # result does not depend on the frame rate. A long gap (e.g. a stall) is capped.
        closed_time = total_time = 0.0
        prev_t, prev_closed = self._history[0]
        for ts, closed in list(self._history)[1:]:
            dt = min(ts - prev_t, 0.5)
            total_time += dt
            if prev_closed:
                closed_time += dt
            prev_t, prev_closed = ts, closed
        s.perclos = closed_time / total_time if total_time > 0 else 0.0
