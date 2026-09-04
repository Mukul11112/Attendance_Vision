"""
core/face_quality.py
Quality gate for face crops BEFORE embedding/matching. A bad crop is worse
than no crop: it either wastes CPU or, far worse, produces a confident-looking
wrong match. Observations that fail here never contribute identity votes.

assess() combines:
  - sharpness   (variance of Laplacian)
  - brightness  (mean luma inside sane bounds)
  - size        (face pixels; tiny faces carry almost no identity information)
  - pose        (yaw/pitch estimated from the 5 landmarks; extreme side/down
                 faces are rejected because ArcFace similarity degrades there)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import settings


@dataclass
class Quality:
    passed: bool
    score: float            # 0..1 combined quality
    strong: bool            # counts as a high-quality observation in voting
    reason: str = ""


def _pose_from_landmarks(lm: np.ndarray) -> tuple:
    """Rough (yaw_deg, pitch_deg) from the 5-point layout.
    Yaw: nose offset between the eyes. Pitch: nose vertical position between
    eye line and mouth line. Rough is fine - this only gates, never identifies."""
    right_eye, left_eye, nose, rm, lmth = lm
    eye_cx = (right_eye[0] + left_eye[0]) / 2.0
    eye_w = max(abs(left_eye[0] - right_eye[0]), 1e-6)
    yaw = np.degrees(np.arctan2(nose[0] - eye_cx, eye_w))
    eye_cy = (right_eye[1] + left_eye[1]) / 2.0
    mouth_cy = (rm[1] + lmth[1]) / 2.0
    face_h = max(mouth_cy - eye_cy, 1e-6)
    # nose normally sits ~55-65% of the way from eyes to mouth
    ratio = (nose[1] - eye_cy) / face_h
    pitch = float((ratio - 0.60) * 90.0)
    return float(yaw), pitch


def assess(face_crop: np.ndarray, landmarks: Optional[np.ndarray],
           det_score: float, blur_min: Optional[float] = None) -> Quality:
    if face_crop is None or face_crop.size == 0:
        return Quality(False, 0.0, False, "empty crop")
    h, w = face_crop.shape[:2]
    if min(h, w) < settings.MIN_FACE_SIZE:
        return Quality(False, 0.0, False, f"too small ({w}x{h})")

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _blur_min = blur_min if blur_min is not None else settings.QUALITY_BLUR_MIN
    if blur < _blur_min:
        return Quality(False, 0.0, False, f"blurry (lapvar={blur:.0f})")

    luma = float(gray.mean())
    if not (settings.QUALITY_BRIGHT_MIN <= luma <= settings.QUALITY_BRIGHT_MAX):
        return Quality(False, 0.0, False, f"bad exposure (luma={luma:.0f})")

    yaw = pitch = 0.0
    if landmarks is not None and len(landmarks) == 5:
        yaw, pitch = _pose_from_landmarks(np.asarray(landmarks, dtype=np.float32))
        if abs(yaw) > settings.QUALITY_MAX_YAW_DEG:
            return Quality(False, 0.0, False, f"extreme yaw ({yaw:.0f} deg)")
        if abs(pitch) > settings.QUALITY_MAX_PITCH_DEG:
            return Quality(False, 0.0, False, f"extreme pitch ({pitch:.0f} deg)")

    # ── combined 0..1 score ────────────────────────────────────────────────
    s_blur = min(blur / (settings.QUALITY_BLUR_MIN * 4.0), 1.0)
    s_size = min(min(h, w) / 112.0, 1.0)
    s_pose = max(0.0, 1.0 - (abs(yaw) / settings.QUALITY_MAX_YAW_DEG) * 0.7
                 - (abs(pitch) / settings.QUALITY_MAX_PITCH_DEG) * 0.3)
    s_det = float(np.clip(det_score, 0.0, 1.0))
    score = float(0.30 * s_blur + 0.25 * s_size + 0.25 * s_pose + 0.20 * s_det)

    if score < settings.FACE_QUALITY_MIN:
        return Quality(False, score, False, f"low combined quality ({score:.2f})")
    return Quality(True, score, score >= settings.FACE_QUALITY_STRONG, "")
