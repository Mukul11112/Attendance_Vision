"""
models/face_detector_scrfd.py
SCRFD face detector — the stronger alternative to YuNet on small faces.

Why this exists: office CCTV faces here run 20-90 px. YuNet's own floor is
MIN_FACE_SIZE (34 px) and below that the pipeline has to 2x super-sample the
person ROI just to get a detection at all (see pipeline.probe_faces). SCRFD-10G
was trained with heavy small-face augmentation and finds those faces natively,
which means fewer upscale round-trips and better landmarks — and landmark
quality is what the ArcFace alignment (and therefore every similarity score)
rests on.

Drop-in for YuNetFaceDetector: same Face dataclass, same detect() signature, so
core/pipeline.py does not care which one it is holding.

Weights: antelopev2/scrfd_10g_bnkps.onnx (InsightFace v0.7 release).
"""
from __future__ import annotations
import threading
from typing import List, Tuple

import cv2
import numpy as np

from config import settings
from models.face_detector import Face

# SCRFD-10G "bnkps": 3 FPN levels, 2 anchors per location, 5 keypoints.
_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
_FMC = 3                       # feature map count (one score/bbox/kps set per stride)
_INPUT_MEAN = 127.5
_INPUT_STD = 128.0


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> List[int]:
    x1, y1, x2, y2, scores = (dets[:, 0], dets[:, 1], dets[:, 2],
                              dets[:, 3], dets[:, 4])
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


class ScrfdFaceDetector:
    """SCRFD-10G. Letterboxes the ROI to a fixed square, so anchor centres can
    be cached per input size instead of rebuilt on every call."""

    def __init__(self, input_size: int = 320) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = settings.ORT_INTRA_THREADS
        self.session = ort.InferenceSession(
            settings.SCRFD_MODEL, sess_options=so,
            providers=settings.ORT_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(input_size)
        self._centres: dict = {}

    def _anchor_centres(self, stride: int, h: int, w: int) -> np.ndarray:
        key = (stride, h, w)
        c = self._centres.get(key)
        if c is None:
            ys, xs = np.mgrid[:h, :w][:2]
            c = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
            c = c.reshape(-1, 2)
            if _NUM_ANCHORS > 1:
                c = np.repeat(c, _NUM_ANCHORS, axis=0)
            self._centres[key] = c
        return c

    def detect(self, image: np.ndarray,
               offset: Tuple[int, int] = (0, 0)) -> List[Face]:
        if image is None or image.size == 0:
            return []
        ih, iw = image.shape[:2]
        if min(ih, iw) < 12:
            return []

        # letterbox into a square canvas: SCRFD needs side % 32 == 0
        side = self.input_size
        scale = min(side / iw, side / ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh))

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / _INPUT_STD, (side, side),
            (_INPUT_MEAN, _INPUT_MEAN, _INPUT_MEAN), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})

        thresh = settings.FACE_DETECTION_CONFIDENCE
        boxes_all, kps_all, scores_all = [], [], []
        for idx, stride in enumerate(_STRIDES):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + _FMC].reshape(-1, 4) * stride
            kps_preds = outs[idx + _FMC * 2].reshape(-1, 10) * stride
            h, w = side // stride, side // stride
            centres = self._anchor_centres(stride, h, w)
            pos = np.where(scores >= thresh)[0]
            if pos.size == 0:
                continue
            boxes_all.append(_distance2bbox(centres[pos], bbox_preds[pos]))
            kps_all.append(_distance2kps(centres[pos], kps_preds[pos]))
            scores_all.append(scores[pos])

        if not boxes_all:
            return []
        boxes = np.vstack(boxes_all) / scale
        kpss = np.vstack(kps_all).reshape(-1, 5, 2) / scale
        scores = np.concatenate(scores_all)

        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, settings.FACE_NMS_IOU)

        ox, oy = offset
        faces: List[Face] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            if min(x2 - x1, y2 - y1) < settings.MIN_FACE_SIZE_SCRFD:
                continue
            lm = kpss[i].astype(np.float32).copy()
            lm[:, 0] += ox
            lm[:, 1] += oy
            faces.append(Face(
                box=(int(x1) + ox, int(y1) + oy, int(x2) + ox, int(y2) + oy),
                landmarks=lm, score=float(scores[i])))
        return faces


_tls = threading.local()


def get_scrfd_detector() -> ScrfdFaceDetector:
    """One detector PER THREAD — same rule as YuNet: ORT sessions are cheap to
    share but the cached anchor grids are keyed by input size, and the live
    camera threads each drive their own ROI shapes."""
    det = getattr(_tls, "scrfd", None)
    if det is None:
        det = ScrfdFaceDetector(input_size=settings.SCRFD_INPUT_SIZE)
        _tls.scrfd = det
    return det
