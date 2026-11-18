"""Face landmark extraction with the MediaPipe Tasks API (FaceLandmarker).

Recent MediaPipe releases removed the old `mp.solutions.face_mesh` API, so this
project uses the Tasks API, which needs a small model file (~3.6 MB). It is
downloaded automatically into ./models on first run.
"""
from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .geometry import landmarks_to_pixels

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the FaceLandmarker model if it is not already on disk."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmark model to {path} ...")
    tmp = path.with_suffix(".part")
    try:
        # Explicit timeout so a blocked network can't hang the caller forever.
        with urllib.request.urlopen(MODEL_URL, timeout=30) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if tmp.stat().st_size < 1_000_000:
            raise RuntimeError("downloaded file is too small")
        tmp.replace(path)
    except Exception as exc:
        raise RuntimeError(
            f"Model download failed ({exc}). Download it manually from\n"
            f"  {MODEL_URL}\nand save it as {path}"
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)  # never leave a partial .part file behind
    return path


class FaceLandmarkExtractor:
    """Wraps FaceLandmarker in VIDEO mode (tracks between frames => faster on CPU)."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, min_confidence: float = 0.5):
        model_path = ensure_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> np.ndarray | None:
        """Return a (478, 2) array of pixel landmarks, or None if no face is found.

        `timestamp` (seconds) is the frame's time in its own stream (e.g. position in a
        video file). Without it the wall clock is used, which suits a live webcam.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        elapsed = (time.monotonic() - self._t0) if timestamp is None else timestamp
        ts = int(elapsed * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        return landmarks_to_pixels(result.face_landmarks[0], w, h)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
