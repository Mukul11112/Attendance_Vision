"""
models/face_embedder.py
ArcFace-compatible 512-d face embeddings via ONNX Runtime CPU.

Pipeline per face:  5-landmark similarity alignment to the canonical ArcFace
112x112 template  ->  BGR->RGB, normalize to [-1, 1]  ->  ONNX forward  ->
L2 normalization.  Cosine similarity between two embeddings is then a plain
dot product, which the gallery exploits with vectorized matmul.
"""
from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from config import settings

# Canonical ArcFace 112x112 destination landmarks (insightface standard).
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

INPUT_SIZE = 112


def align_face(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Similarity-transform the face to the canonical 112x112 ArcFace crop."""
    M, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)
    if M is None:                       # degenerate landmarks: fall back to bbox resize
        x1 = int(max(0, landmarks[:, 0].min() - 8)); x2 = int(landmarks[:, 0].max() + 8)
        y1 = int(max(0, landmarks[:, 1].min() - 8)); y2 = int(landmarks[:, 1].max() + 8)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    return cv2.warpAffine(frame, M, (INPUT_SIZE, INPUT_SIZE), borderValue=0)


class ArcFaceEmbedder:
    """ArcFace ONNX embedder. `model_path` selects the backbone — r50
    (arcface_w600k_r50) or r100 (glintr100). Both emit 512-d and share the
    same preprocessing, so they differ only in accuracy and cost; the
    embeddings are NOT interchangeable, and a gallery built with one is
    meaningless to the other."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = settings.ORT_INTRA_THREADS
        self.model_path = model_path or settings.ARCFACE_MODEL
        self.session = ort.InferenceSession(
            self.model_path, sess_options=so,
            providers=settings.ORT_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Aligned, L2-normalized 512-d embedding for one face."""
        aligned = align_face(frame, landmarks)
        return self.embed_aligned(aligned)

    def embed_aligned(self, aligned_112: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(aligned_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        blob = img.transpose(2, 0, 1)[None]                 # 1x3x112x112
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        emb = emb.astype(np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb


_embedder: Optional[ArcFaceEmbedder] = None


def get_embedder() -> ArcFaceEmbedder:
    """The embedder the whole app shares, chosen by settings.FACE_EMBEDDER.

    Falls back to r50 if the r100 weights are absent rather than crashing at
    startup — a missing 260 MB download should degrade the system, not stop it.
    """
    global _embedder
    if _embedder is None:
        import os
        path = settings.ARCFACE_MODEL
        if getattr(settings, "FACE_EMBEDDER", "r50") == "r100":
            r100 = getattr(settings, "ARCFACE_R100_MODEL", "")
            if r100 and os.path.isfile(r100):
                path = r100
            else:
                import logging
                logging.getLogger("face_embedder").warning(
                    "FACE_EMBEDDER='r100' but %s is missing — falling back to "
                    "r50. Run scripts/download_models.py.", r100)
        _embedder = ArcFaceEmbedder(path)
    return _embedder
