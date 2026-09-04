"""
scripts/benchmark.py — measure this machine's inference speed and estimate
processing throughput. Use it to size deployments ("how many camera-hours can
this box process per hour?") and to verify GPU acceleration is active.

    .\\.venv\\Scripts\\python.exe scripts\\benchmark.py
"""
from __future__ import annotations
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from config import settings  # noqa: E402


def _time(fn, warmup=3, iters=25) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> int:
    print("Inference backend:", settings.ORT_PROVIDERS[0])
    from models.registry import missing_required, status_report
    if missing_required():
        print(status_report()); return 1

    from models.person_detector_yolo import get_person_detector
    from models.face_detector import get_face_detector
    from models.face_embedder import get_embedder

    frame = (np.random.rand(1080, 1920, 3) * 255).astype("uint8")
    roi = (np.random.rand(220, 90, 3) * 255).astype("uint8")
    face112 = (np.random.rand(112, 112, 3) * 255).astype("uint8")

    pdet, fdet, femb = get_person_detector(), get_face_detector(), get_embedder()
    ms_yolo = _time(lambda: pdet.detect(frame))
    ms_yunet = _time(lambda: fdet.detect(roi))
    ms_arc = _time(lambda: femb.embed_aligned(face112))
    print(f"YOLOv8n person detection : {ms_yolo:7.1f} ms / frame")
    print(f"YuNet face probe (1 ROI) : {ms_yunet:7.1f} ms")
    print(f"ArcFace embedding        : {ms_arc:7.1f} ms")

    body_line = "not installed (optional)"
    try:
        from models.body_embedder import get_body_embedder
        be = get_body_embedder()
        if be is not None:
            body_line = f"{_time(lambda: be.embed(roi)):.1f} ms"
    except Exception:
        pass
    print(f"OSNet body embedding     : {body_line}")

    # per-sample cost model: detection every 2nd sample (BALANCED), ~8 probes,
    # 1 embedding; motion-gated static scenes ~free
    for mode, det_n, sps in (("FAST", 3, 1.0), ("BALANCED", 2, 2.0)):
        per_sample = ms_yolo / det_n + 8 * ms_yunet / 3 + ms_arc + 8.0
        fps = 1000.0 / per_sample
        rt = fps / sps
        print(f"{mode:8s}: ~{fps:5.1f} samples/s  ≈ {rt:5.1f}x realtime per worker "
              f"(active scenes)")
    print("\nCapacity example: 10 cameras x 10 h = 100 camera-hours;")
    print("hours to process ≈ 100 / (realtime-multiple x workers x ~2 for")
    print("motion-gated static time). Install onnxruntime-directml or")
    print("onnxruntime-gpu and re-run to see the accelerated numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
