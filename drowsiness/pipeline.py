"""Glue: frame -> landmarks -> ratios -> detector -> annotated frame.

Shared by the Streamlit app and the desktop script so both behave identically.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .detector import DetectorConfig, DetectorState, DrowsinessDetector, Status
from .geometry import LEFT_EYE, MOUTH, RIGHT_EYE, face_ratios
from .landmarks import FaceLandmarkExtractor

GREEN, RED, YELLOW, WHITE, GREY = (80, 200, 80), (40, 40, 230), (0, 200, 255), (255, 255, 255), (160, 160, 160)


class DrowsinessPipeline:
    def __init__(self, config: DetectorConfig | None = None, extractor: FaceLandmarkExtractor | None = None):
        self.detector = DrowsinessDetector(config or DetectorConfig())
        self.extractor = extractor or FaceLandmarkExtractor()
        self._fps = 0.0
        self._last = None

    @property
    def config(self) -> DetectorConfig:
        return self.detector.config

    def process(
        self, frame_bgr: np.ndarray, draw: bool = True, timestamp: float | None = None
    ) -> tuple[np.ndarray, DetectorState]:
        """Process one frame.

        `timestamp` is the frame's time in seconds. Pass it for recorded video so the
        detector's timers follow video time; leave it None for a live camera.
        """
        wall = time.monotonic()
        now = wall if timestamp is None else timestamp
        if self._last is not None:
            dt = wall - self._last  # FPS = processing speed, always wall-clock
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        self._last = wall

        points = self.extractor.process(frame_bgr, timestamp)
        if points is None:
            state = self.detector.update(None, None, now)
        else:
            r = face_ratios(points)
            state = self.detector.update(r["ear"], r["mar"], now)

        if draw:
            frame_bgr = annotate(frame_bgr, points, state, self._fps)
        return frame_bgr, state

    def close(self) -> None:
        self.extractor.close()


def annotate(frame: np.ndarray, points: np.ndarray | None, s: DetectorState, fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if points is not None:
        eye_color = RED if s.eyes_closed else GREEN
        for idx in (LEFT_EYE, RIGHT_EYE):
            pts = points[idx].astype(np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)
            for p in pts:
                cv2.circle(out, tuple(p), 2, eye_color, -1, cv2.LINE_AA)
        mouth = points[MOUTH].astype(np.int32)
        cv2.line(out, tuple(mouth[0]), tuple(mouth[1]), YELLOW, 1, cv2.LINE_AA)
        cv2.line(out, tuple(mouth[2]), tuple(mouth[3]), YELLOW, 1, cv2.LINE_AA)

    # Info panel
    panel = out.copy()
    cv2.rectangle(panel, (0, 0), (260, 150), (0, 0, 0), -1)
    out = cv2.addWeighted(panel, 0.45, out, 0.55, 0)
    lines = [
        (f"EAR: {s.ear:.3f}  (thr {s.threshold:.3f})", WHITE),
        (f"MAR: {s.mar:.3f}", WHITE),
        (f"Blinks: {s.blinks}   Yawns: {s.yawns}", WHITE),
        (f"Closed: {s.closed_for:.1f}s  PERCLOS: {s.perclos * 100:.0f}%", WHITE),
        (f"FPS: {fps:.1f}", GREY),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(out, text, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    status_color = {Status.AWAKE: GREEN, Status.DROWSY: RED, Status.CALIBRATING: YELLOW, Status.NO_FACE: GREY}[s.status]
    label = "CALIBRATING - keep eyes open" if s.status == Status.CALIBRATING else s.status.value
    cv2.putText(out, label, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

    if s.alarm:
        # Flashing red border + banner
        if int(time.monotonic() * 4) % 2 == 0:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), RED, 12)
        text = "DROWSINESS ALERT!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
        x, y = (w - tw) // 2, int(h * 0.78)  # below the face so the eyes stay visible
        cv2.rectangle(out, (x - 12, y - th - 12), (x + tw + 12, y + 12), RED, -1)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    return out
