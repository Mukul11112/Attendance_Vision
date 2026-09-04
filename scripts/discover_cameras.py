"""Discover the live camera channels on a Hikvision NVR.
Hikvision NVR RTSP channel numbering: <cam><stream> where cam=1..N and
stream=1 (main) / 2 (sub). So cam 1 main = 101, cam 2 main = 201, ...
Grabs one frame from each candidate channel and reports resolution + a thumbnail.

Run:  python scripts/discover_cameras.py
"""
from __future__ import annotations
import json, os, sys, time
from urllib.parse import quote
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

cfg = json.load(open(os.path.join(ROOT, "camera_config.json")))
ip, port = cfg["ip"], cfg.get("port", 554)
user, pw = quote(str(cfg["username"]), safe=""), quote(str(cfg["password"]), safe="")

out_dir = os.path.join(ROOT, "data", "camera_test", "channels")
os.makedirs(out_dir, exist_ok=True)

MAX_CAMERAS = 10           # probe cam 1..10 (main stream)
found = []
for cam in range(1, MAX_CAMERAS + 1):
    ch = f"{cam}01"
    url = f"rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/{ch}"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    ok, frame = False, None
    t0 = time.time()
    while time.time() - t0 < 5:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
    if ok and frame is not None:
        h, w = frame.shape[:2]
        thumb = cv2.resize(frame, (320, int(320 * h / w)))
        cv2.imwrite(os.path.join(out_dir, f"cam{cam}_ch{ch}.jpg"), thumb)
        found.append((cam, ch, w, h))
        print(f"  cam {cam}  channel {ch}  ->  LIVE  {w}x{h}")
    else:
        print(f"  cam {cam}  channel {ch}  ->  (no stream)")
    cap.release()

print(f"\n{len(found)} live camera(s) found: "
      + ", ".join(f"{c[0]}(ch{c[1]})" for c in found))
print(f"thumbnails saved in {out_dir}")
