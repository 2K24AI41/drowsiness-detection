"""Streamlit web app: real-time drowsiness detection from your webcam.

Run:  streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import io
import os
import struct
import threading
import time
import wave
from collections import deque

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from drowsiness.detector import DetectorConfig, Status
from drowsiness.landmarks import ensure_model
from drowsiness.pipeline import DrowsinessPipeline

st.set_page_config(page_title="Drowsiness Detection", page_icon="👁️", layout="wide")


# --------------------------------------------------------------------------- audio
@st.cache_data
def _base_wav(seconds: float = 1.2, freq: float = 880.0, rate: int = 22050) -> bytes:
    """Generate a pulsing beep as WAV bytes (no audio file needed). Cached once."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    pulse = (np.sin(2 * np.pi * 4 * t) > 0).astype(float)  # 4 beeps per second
    data = (tone * pulse * 0.6 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


def alarm_wav(variant: int = 0) -> bytes:
    """The alarm sound, made unique per `variant`.

    st.audio derives its element ID from the bytes, and not every Streamlit version has a
    `key` argument. Writing `variant` (base 64) into the first four samples, which are
    silent at <0.2% amplitude, gives every alarm distinct bytes without growing the cache.
    """
    wav = bytearray(_base_wav())
    digits = [(variant >> (6 * i)) & 63 for i in range(4)]   # 64**4 = 16M distinct clips
    wav[44:52] = struct.pack("<4h", *digits)                  # 44-byte PCM header, then samples
    return bytes(wav)


@st.cache_resource(show_spinner="Downloading face model (first run only)...")
def prepare_model():
    """Download the landmark model here, in the main thread, not inside the video worker."""
    return ensure_model()


def rtc_configuration() -> dict:
    """STUN by default; add a TURN relay via env vars when behind strict NATs / on a cloud host."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_url = os.environ.get("TURN_URL")
    if turn_url:
        servers.append({
            "urls": [turn_url],
            "username": os.environ.get("TURN_USERNAME", ""),
            "credential": os.environ.get("TURN_CREDENTIAL", ""),
        })
    return {"iceServers": servers}


# ----------------------------------------------------------------------- processor
class DrowsinessProcessor(VideoProcessorBase):
    """Runs in streamlit-webrtc's worker thread, one instance per webcam session."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config = DetectorConfig()
        self.pipeline: DrowsinessPipeline | None = None
        self.state = None
        self.history: deque[tuple[float, float, float]] = deque(maxlen=300)
        self.reset_requested = False
        self.mirror = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)
        if self.pipeline is None:
            self.pipeline = DrowsinessPipeline(self.config)
        if self.reset_requested:
            self.reset_requested = False
            self.pipeline.detector.reset()
            with self.lock:
                self.history.clear()
        if not self.config.auto_calibrate:
            # Applied here (video thread) so the UI thread never touches detector state.
            self.pipeline.detector.state.threshold = self.config.ear_threshold

        out, state = self.pipeline.process(img)
        with self.lock:
            # Publish a snapshot: `state` is mutated on every frame by the detector.
            self.state = dataclasses.replace(state)
            self.history.append((time.monotonic(), state.ear, state.threshold))
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def __del__(self) -> None:
        # Release the MediaPipe landmarker when the webcam session ends.
        try:
            if self.pipeline is not None:
                self.pipeline.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------- UI
st.title("👁️ Real-Time Drowsiness Detection")
st.caption(
    "Tracks your eyes with MediaPipe Face Landmarker, computes the Eye Aspect Ratio (EAR), "
    "and raises an alarm when your eyes stay closed too long or PERCLOS gets too high."
)

with st.sidebar:
    st.header("⚙️ Settings")
    auto_cal = st.toggle("Auto-calibrate threshold", value=True,
                         help="Learns your normal open-eye EAR during the first 3 seconds.")
    ear_thr = st.slider("EAR threshold (if not auto)", 0.10, 0.35, 0.21, 0.01, disabled=auto_cal)
    drowsy_s = st.slider("Eyes closed for (seconds) → alarm", 0.5, 4.0, 1.5, 0.1)
    perclos = st.slider("PERCLOS alarm level (%)", 10, 80, 30, 5)
    mar_thr = st.slider("Yawn MAR threshold", 0.3, 1.0, 0.6, 0.05)
    smoothing = st.slider("EAR smoothing", 0.0, 0.9, 0.5, 0.1)
    sound = st.toggle("Sound alarm", value=True)
    mirror = st.toggle("Mirror video", value=True)
    reset = st.button("🔄 Reset & recalibrate", use_container_width=True)

prepare_model()

col_video, col_stats = st.columns([3, 2])

with col_video:
    ctx = webrtc_streamer(
        key="drowsiness",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DrowsinessProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration=rtc_configuration(),
        async_processing=True,
    )

proc: DrowsinessProcessor | None = ctx.video_processor
if proc is not None:
    # Push the sidebar values into the running detector (dataclass is shared by reference).
    cfg = proc.config
    if cfg.auto_calibrate != auto_cal:
        cfg.auto_calibrate = auto_cal
        proc.reset_requested = True
    cfg.ear_threshold = ear_thr
    cfg.drowsy_seconds = drowsy_s
    cfg.perclos_alarm = perclos / 100
    cfg.mar_threshold = mar_thr
    cfg.smoothing = smoothing
    proc.mirror = mirror
    if reset:
        proc.reset_requested = True

with col_stats:
    status_box = st.empty()
    m1, m2, m3 = st.columns(3)
    blink_m, yawn_m, perclos_m = m1.empty(), m2.empty(), m3.empty()
    ear_m, closed_m, events_m = m1.empty(), m2.empty(), m3.empty()
    chart_box = st.empty()
    audio_box = st.empty()

if not ctx.state.playing:
    with col_stats:
        st.info("Click **START** and allow camera access. Keep your eyes open for 3 seconds "
                "while the system calibrates, then try closing them for ~2 seconds.")
else:
    SOUND_REPEAT_SECONDS = 1.5   # slightly longer than the 1.2 s clip
    last_sound = st.session_state.get("last_sound", 0.0)
    while ctx.state.playing:
        p = ctx.video_processor
        if p is None or p.state is None:
            time.sleep(0.1)
            continue
        with p.lock:
            s = p.state          # a snapshot copy, safe to read
            hist = list(p.history)

        if s.status == Status.DROWSY:
            status_box.error("🚨 **DROWSY — WAKE UP!**")
        elif s.status == Status.CALIBRATING:
            status_box.warning("⏳ Calibrating… keep your eyes open")
        elif s.status == Status.NO_FACE:
            status_box.info("🙈 No face detected")
        else:
            status_box.success("✅ Awake")

        blink_m.metric("Blinks", s.blinks)
        yawn_m.metric("Yawns", s.yawns)
        perclos_m.metric("PERCLOS", f"{s.perclos * 100:.0f}%")
        ear_m.metric("EAR", f"{s.ear:.3f}")
        closed_m.metric("Eyes closed", f"{s.closed_for:.1f}s")
        events_m.metric("Alarms", s.alarm_events)

        if len(hist) > 5:
            t0 = hist[0][0]
            df = pd.DataFrame(
                {"time (s)": [h[0] - t0 for h in hist], "EAR": [h[1] for h in hist],
                 "threshold": [h[2] for h in hist]}
            ).set_index("time (s)")
            chart_box.line_chart(df, height=220)

        # Keep sounding while the alarm is active (not just once when it starts). The
        # last-sound time lives in session_state, so a widget-triggered rerun of this
        # script doesn't replay an old alarm.
        now = time.monotonic()
        if sound and s.alarm and now - last_sound >= SOUND_REPEAT_SECONDS:
            n = st.session_state.get("sound_count", 0) + 1
            st.session_state["sound_count"] = n
            audio_box.audio(alarm_wav(variant=n), format="audio/wav", autoplay=True)
            last_sound = now
            st.session_state["last_sound"] = now
        time.sleep(0.25)

with st.expander("ℹ️ How it works"):
    st.markdown(
        """
1. **Face landmarks** – MediaPipe Face Landmarker finds 478 points on your face every frame (CPU-friendly).
2. **Eye Aspect Ratio** – for each eye, `EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)`. Open eyes ≈ 0.25–0.35, closed ≈ < 0.2.
3. **Calibration** – the first 3 s learn *your* open-eye EAR; threshold = 75 % of it.
4. **Decision** – alarm if eyes stay closed longer than the time limit **or** PERCLOS
   (% of the last 60 s with eyes closed) is above the limit. Short closures are counted as blinks.
5. **Yawns** – Mouth Aspect Ratio above the threshold for > 1 s.
"""
    )
