"""
scripts/tune_person_detection.py
Makes person detection VISIBLE so you can verify that far-away and seated
people are found, before caring about recognition.

Samples frames from a video, runs the person detector at one or more input
resolutions, and writes annotated JPEGs to data/detection_preview/.
Green box = high-confidence detection (starts tracks)
Orange box = low-confidence detection (sustains existing tracks via ByteTrack)
Each image is titled with the count, so "correct persons in frame" can be
checked against what your eyes see.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\tune_person_detection.py "C:\\path\\video.mp4"
  .\\.venv\\Scripts\\python.exe scripts\\tune_person_detection.py "video.mp4" --sizes 640 960 1280 --samples 8
"""
from __future__ import annotations
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from config import settings  # noqa: E402

OUT_DIR = os.path.join(settings.DATA_DIR, "detection_preview")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 1
    video_path = sys.argv[1]
    sizes = [settings.YOLO_INPUT_SIZE]
    if "--sizes" in sys.argv:
        i = sys.argv.index("--sizes") + 1
        sizes = []
        while i < len(sys.argv) and sys.argv[i].isdigit():
            sizes.append(int(sys.argv[i])); i += 1
    n_samples = 6
    if "--samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--samples") + 1])

    from models.registry import missing_required, status_report
    if missing_required():
        print(status_report()); return 1
    from models.person_detector_yolo import YoloPersonDetector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}"); return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(OUT_DIR, exist_ok=True)

    idxs = np.linspace(total * 0.05, total * 0.95, num=n_samples, dtype=int)
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok:
            frames.append((int(fi), frame))
    cap.release()
    if not frames:
        print("Could not read frames."); return 1

    print(f"floor={getattr(settings, 'PERSON_DETECTION_FLOOR', '?')}  "
          f"high-band(track start)={settings.TRACK_HIGH_THRESH}  "
          f"min box={settings.PERSON_MIN_BOX_W}x{settings.PERSON_MIN_BOX_H}px  "
          f"max aspect={settings.PERSON_MAX_ASPECT}")
    print(f"writing annotated frames -> {OUT_DIR}\n")

    header = "size | " + " | ".join(f"frame{k}" for k in range(len(frames))) + " | avg"
    print(header); print("-" * len(header))
    for size in sizes:
        det = YoloPersonDetector(input_size=size)
        counts = []
        for k, (fi, frame) in enumerate(frames):
            dets = det.detect(frame)
            counts.append(len(dets))
            vis = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = [int(v) for v in d.box]
                high = d.score >= settings.TRACK_HIGH_THRESH
                color = (0, 200, 0) if high else (0, 140, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{d.score:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(vis, f"input={size}px  persons={len(dets)}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            out = os.path.join(OUT_DIR, f"size{size}_frame{k}_{fi}.jpg")
            cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{size:4d} | " + " | ".join(f"{c:6d}" for c in counts)
              + f" | {np.mean(counts):.1f}")

    print("\nOpen the folder above and CHECK BY EYE:")
    print("  - is every visible person boxed (green or orange)?")
    print("  - far people should at least be orange")
    print("  - no boxes on chairs/bags/monitors")
    print("Pick the smallest size that boxes everyone; set YOLO_INPUT_SIZE in")
    print("config/settings.py to that value (bigger = slower processing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
