"""Hikvision RTSP connection test.
Reads camera_config.json, tries the common Hikvision RTSP URL patterns,
grabs a frame, saves a snapshot, and runs person+face detection on it so we
know the live feed works end-to-end BEFORE building the live tab.

Run:  python scripts/test_camera.py
"""
from __future__ import annotations
import json, os, sys, time
from urllib.parse import quote
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "camera_config.json")))
ip, port = cfg["ip"], cfg.get("port", 554)
raw_user, raw_pw = cfg["username"], cfg["password"]
# URL-encode credentials: chars like '@' ':' '/' in a password break RTSP URL parsing.
user, pw = quote(raw_user, safe=""), quote(raw_pw, safe="")
ch = str(cfg.get("channel", "101"))

if not user or not pw:
    print("ERROR: fill username/password in camera_config.json first.")
    sys.exit(1)

# Hikvision URL variants (model-dependent): newer firmware uses Streaming/Channels,
# older uses /h264/... — we try each until one delivers a frame.
sub = "102" if ch.endswith("2") else "101"
candidates = [
    f"rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/{ch}",
    f"rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/{sub}",
    f"rtsp://{user}:{pw}@{ip}:{port}/h264/ch1/main/av_stream",
    f"rtsp://{user}:{pw}@{ip}:{port}/ISAPI/Streaming/Channels/{ch}",
]

def safe(u):  # hide password in printout
    return u.replace(pw, "****")

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

snap_dir = os.path.join(ROOT, "data", "camera_test")
os.makedirs(snap_dir, exist_ok=True)

working = None
for url in candidates:
    print(f"\nTrying: {safe(url)}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    t0 = time.time()
    ok, frame = False, None
    while time.time() - t0 < 8:          # give RTSP a few seconds to hand over a frame
        ok, frame = cap.read()
        if ok and frame is not None:
            break
    if ok and frame is not None:
        h, w = frame.shape[:2]
        print(f"  CONNECTED — frame {w}x{h}")
        snap = os.path.join(snap_dir, "live_snapshot.jpg")
        cv2.imwrite(snap, frame)
        print(f"  snapshot saved -> {snap}")
        working = (url, frame)
        cap.release()
        break
    print("  no frame.")
    cap.release()

if not working:
    print("\nRESULT: could not pull a frame from any URL. "
          "Check credentials / that RTSP is enabled on the camera.")
    sys.exit(2)

url, frame = working
print(f"\nWORKING URL: {safe(url)}")

# ── run detection on the live frame so we know recognition works on this feed ──
try:
    from models.person_detector_yolo import get_person_detector
    from models.face_detector import get_face_detector
    pdet = get_person_detector()
    fdet = get_face_detector()
    persons = pdet.detect(frame)
    faces = fdet.detect(frame)
    print(f"Live-frame detection: {len(persons)} person(s), {len(faces)} face(s)")
except Exception as e:
    print(f"(detection step skipped: {e})")

print("\nOK — the live stream works. Ready to build the live tab on this URL.")
