"""
models/person_detector_yolo.py
Person detection with a YOLOv8n ONNX model on ONNX Runtime CPU.

Accepts ONLY class 0 (person) and then applies the false-positive gates from
settings: confidence, NMS, minimum box width/height, minimum area fraction,
and aspect-ratio validation. Everything else (temporal confirmation) is the
tracker's job (TRACK_MIN_HITS).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import settings

PERSON_CLASS = 0


@dataclass
class PersonDet:
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2 in frame coords
    score: float
    kpts: "np.ndarray | None" = None         # (17,3) x,y,conf frame coords
                                             # (pose models only; NOT identity)


def _letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, (left, top)


def decode_yolo(pred: np.ndarray, r: float, dx: int, dy: int,
                W: int, H: int, floor: float):
    """Decode one YOLOv8 output (channels-first (C,N) or (N,C)) into person
    candidates. Handles BOTH heads:
      detection: C>=84 -> cxcywh + 80 class scores (person = argmax class 0)
      pose:      C==56 -> cxcywh + person conf + 17x(x,y,conf) keypoints
    Returns (boxes_xyxy[N,4], scores[N], kpts[N,17,3]|None) in frame coords."""
    KNOWN_C = (56, 84)                      # pose / 80-class detection heads
    if pred.shape[1] in KNOWN_C:
        pass                                # already (N, C)
    elif pred.shape[0] in KNOWN_C:
        pred = pred.T                       # channels-first -> (N, C)
    elif pred.shape[0] < pred.shape[1]:
        pred = pred.T                       # generic: smaller dim = channels
    C = pred.shape[1]
    if C == 56:                             # pose head
        scores = pred[:, 4]
        keep = scores >= floor
        pred, scores = pred[keep], scores[keep]
        kpts = pred[:, 5:].reshape(-1, 17, 3).copy()
        kpts[:, :, 0] = (kpts[:, :, 0] - dx) / r
        kpts[:, :, 1] = (kpts[:, :, 1] - dy) / r
    else:                                   # detection head
        person_scores = pred[:, 4 + PERSON_CLASS]
        keep = (person_scores >= floor) & (pred[:, 4:].argmax(axis=1) == PERSON_CLASS)
        pred, scores = pred[keep], person_scores[keep]
        kpts = None
    if len(scores) == 0:
        return np.zeros((0, 4), np.float32), scores, None
    b = pred[:, :4]
    x1 = np.clip((b[:, 0] - b[:, 2] / 2 - dx) / r, 0, W - 1)
    y1 = np.clip((b[:, 1] - b[:, 3] / 2 - dy) / r, 0, H - 1)
    x2 = np.clip((b[:, 0] + b[:, 2] / 2 - dx) / r, 0, W - 1)
    y2 = np.clip((b[:, 1] + b[:, 3] / 2 - dy) / r, 0, H - 1)
    return np.stack([x1, y1, x2, y2], axis=1), scores, kpts


class YoloPersonDetector:
    def __init__(self, input_size: int = None) -> None:
        import onnxruntime as ort
        import os
        so = ort.SessionOptions()
        so.intra_op_num_threads = settings.ORT_INTRA_THREADS
        model_path = settings.YOLO_PERSON_MODEL
        pose_path = getattr(settings, "YOLO_POSE_MODEL", "")
        if pose_path and os.path.isfile(pose_path):
            model_path = pose_path          # pose model preferred when present
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=settings.ORT_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.size = input_size or settings.YOLO_INPUT_SIZE
        self.has_pose = "pose" in model_path.lower()

    def detect(self, frame: np.ndarray) -> List[PersonDet]:
        H, W = frame.shape[:2]
        img, r, (dx, dy) = _letterbox(frame, self.size)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: blob})[0]
        floor = getattr(settings, "PERSON_DETECTION_FLOOR", settings.TRACK_LOW_THRESH)
        boxes, scores, kpts = decode_yolo(out[0], r, dx, dy, W, H, floor)
        if len(scores) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        idx = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores.tolist(),
            floor, settings.PERSON_NMS_IOU)
        if idx is None or len(idx) == 0:
            return []
        idx = np.array(idx).flatten()

        dets: List[PersonDet] = []
        frame_area = float(H * W)
        for i in idx:
            bw, bh = x2[i] - x1[i], y2[i] - y1[i]
            if bw < settings.PERSON_MIN_BOX_W or bh < settings.PERSON_MIN_BOX_H:
                continue
            if (bw * bh) / frame_area < settings.PERSON_MIN_AREA_FRACTION:
                continue
            aspect = bw / max(bh, 1e-6)
            # NOTE: sitting people are wider than standing pedestrians; the
            # upper bound in settings (1.10) already allows that.
            if not (settings.PERSON_MIN_ASPECT <= aspect <= settings.PERSON_MAX_ASPECT):
                continue
            dets.append(PersonDet(
                box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                score=float(scores[i]),
                kpts=(kpts[i] if kpts is not None else None)))
        return dets


_detector: Optional[YoloPersonDetector] = None


def get_person_detector() -> YoloPersonDetector:
    global _detector
    if _detector is None:
        _detector = YoloPersonDetector()
    return _detector
