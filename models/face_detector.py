"""
models/face_detector.py
Face detection with YuNet through OpenCV's FaceDetectorYN (bundled with
opencv-python >= 4.5.4, so no extra runtime is needed for detection).

Returns Face(box, landmarks, score) where:
  box        = (x1, y1, x2, y2) ints in FULL-FRAME coordinates
  landmarks  = 5x2 float array (right eye, left eye, nose, right mouth,
               left mouth) in FULL-FRAME coordinates — ArcFace alignment order
  score      = detector confidence

detect(image, offset) accepts an ROI crop plus the ROI's top-left offset so
the pipeline can search faces only inside person boxes and still get
full-frame coordinates back.
"""
from __future__ import annotations
import os
import threading
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from config import settings


@dataclass
class Face:
    box: Tuple[int, int, int, int]
    landmarks: np.ndarray          # (5, 2) float32, full-frame coords
    score: float


class YuNetFaceDetector:
    def __init__(self) -> None:
        self._det = cv2.FaceDetectorYN.create(
            model=settings.YUNET_MODEL,
            config="",
            input_size=(320, 320),
            score_threshold=settings.FACE_DETECTION_CONFIDENCE,
            nms_threshold=settings.FACE_NMS_IOU,
            top_k=50,
        )

    def detect(self, image: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> List[Face]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        if min(h, w) < settings.MIN_FACE_SIZE:      # ROI too small to hold a usable face
            return []
        self._det.setInputSize((w, h))
        _, dets = self._det.detect(image)
        if dets is None:
            return []
        ox, oy = offset
        faces: List[Face] = []
        for d in dets:
            x, y, bw, bh = d[:4]
            score = float(d[14])
            if min(bw, bh) < settings.MIN_FACE_SIZE:
                continue
            lm = d[4:14].reshape(5, 2).astype(np.float32)
            lm[:, 0] += ox
            lm[:, 1] += oy
            faces.append(Face(
                box=(int(x) + ox, int(y) + oy, int(x + bw) + ox, int(y + bh) + oy),
                landmarks=lm, score=score,
            ))
        return faces


_tls = threading.local()


def get_face_detector() -> YuNetFaceDetector:
    """One detector PER THREAD.

    cv2.FaceDetectorYN is stateful: detect() runs the graph that the preceding
    setInputSize() configured. A single shared instance across the live-camera
    threads let one camera resize the net while another was mid-forward, which
    raised `buf.shape() == m.shape()` in OpenCV's graph engine and killed whole
    camera threads. YuNet is ~340 KB, so per-thread copies are cheap.

    SCRFD is preferred when settings.FACE_DETECTOR says so and its weights are
    present; it finds the 20-30 px faces this site's wide cams produce, which
    YuNet only reaches via pipeline.probe_faces' 2x super-sampling detour.
    """
    det = getattr(_tls, "detector", None)
    if det is not None:
        return det
    if getattr(settings, "FACE_DETECTOR", "yunet") == "scrfd" \
            and os.path.isfile(settings.SCRFD_MODEL):
        from models.face_detector_scrfd import ScrfdFaceDetector
        det = ScrfdFaceDetector(input_size=settings.SCRFD_INPUT_SIZE)
    else:
        det = YuNetFaceDetector()
    _tls.detector = det
    return det
