"""Desktop version: OpenCV window + sound alarm. Also works on a recorded video.

Examples:
    python run_webcam.py                    # default webcam
    python run_webcam.py --camera 1         # another camera
    python run_webcam.py --video test.mp4   # a video file
    python run_webcam.py --save out.mp4     # record the annotated output (good for your README GIF)

Keys:  q = quit   r = reset & recalibrate
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import cv2

from drowsiness.detector import DetectorConfig
from drowsiness.pipeline import DrowsinessPipeline


_beeping = threading.Event()
BEEP_REPEAT_SECONDS = 1.0   # keep sounding while the alarm stays active


def beep() -> None:
    """Non-blocking alarm sound (winsound on Windows, terminal bell elsewhere).

    Ignored while a previous beep is still playing, so overlapping calls never stack.
    """
    if _beeping.is_set():
        return
    _beeping.set()

    def _play():
        try:
            if sys.platform.startswith("win"):
                import winsound
                for _ in range(3):
                    winsound.Beep(1000, 250)
            else:
                print("\a", end="", flush=True)
        finally:
            _beeping.clear()

    threading.Thread(target=_play, daemon=True).start()


def open_writer(path: str, frame, fps: float) -> cv2.VideoWriter:
    h, w = frame.shape[:2]
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time drowsiness detection")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None, help="path to a video file instead of the webcam")
    ap.add_argument("--save", type=str, default=None, help="save annotated video to this .mp4 path")
    ap.add_argument("--ear", type=float, default=None, help="fixed EAR threshold (disables auto-calibration)")
    ap.add_argument("--seconds", type=float, default=1.5, help="eyes closed this long -> alarm")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    cfg = DetectorConfig(drowsy_seconds=args.seconds)
    if args.ear is not None:
        cfg.ear_threshold, cfg.auto_calibrate = args.ear, False

    source = args.video if args.video else args.camera
    # CAP_DSHOW opens Windows webcams much faster than the default backend.
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pipeline = DrowsinessPipeline(cfg)
    writer = None
    pending: list = []            # frames held back until the real FPS is known (webcam + --save)
    t_first = None
    last_beep = 0.0
    is_video = args.video is not None
    # For a video file the detector must follow *video* time, not how fast we decode it.
    src_fps = cap.get(cv2.CAP_PROP_FPS) if is_video else 0.0
    if is_video and not src_fps > 0:
        src_fps = 30.0
    frame_idx = 0
    mirror = not args.no_mirror and args.video is None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            timestamp = frame_idx / src_fps if is_video else None
            frame_idx += 1
            out, state = pipeline.process(frame, timestamp=timestamp)

            now = time.monotonic()
            if state.alarm and now - last_beep >= BEEP_REPEAT_SECONDS:
                beep()
                last_beep = now

            if args.save:
                if writer is None:
                    if is_video:
                        writer = open_writer(args.save, out, src_fps)
                    else:
                        # A webcam's real FPS is only known after running for a moment.
                        t_first = t_first or now
                        pending.append(out)
                        if len(pending) >= 30:
                            fps = len(pending) / max(now - t_first, 1e-3)
                            writer = open_writer(args.save, out, fps)
                            for f in pending:
                                writer.write(f)
                            pending.clear()
                else:
                    writer.write(out)

            cv2.imshow("Drowsiness Detection  (q = quit, r = reset)", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                pipeline.detector.reset()
    finally:
        cap.release()
        if args.save and writer is None and pending:  # quit before 30 frames were collected
            elapsed = max(time.monotonic() - (t_first or time.monotonic()), 1e-3)
            writer = open_writer(args.save, pending[0], len(pending) / elapsed)
            for f in pending:
                writer.write(f)
        if writer is not None:
            writer.release()
        pipeline.close()
        cv2.destroyAllWindows()
        s = pipeline.detector.state
        print(f"Session summary: blinks={s.blinks} yawns={s.yawns} alarms={s.alarm_events}")


if __name__ == "__main__":
    main()
