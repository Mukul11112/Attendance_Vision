"""
models/body_embedder.py
OSNet (x0_25) person re-identification embeddings via ONNX Runtime CPU.

Input: person crop (BGR) -> resize 128x256 -> RGB, ImageNet normalize ->
NCHW -> 512-d embedding -> L2 normalize. Cosine similarity = dot product.

The model file is OPTIONAL: if missing, the pipeline silently falls back to
the Phase 1 color-histogram descriptor (linking only, weaker).
"""
from __future__ import annotations
import os
from typing import Optional

import cv2
import numpy as np

from config import settings

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OSNetEmbedder:
    def __init__(self) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = settings.ORT_INTRA_THREADS
        self.session = ort.InferenceSession(
            settings.OSNET_MODEL, sess_options=so,
            providers=settings.ORT_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 48 or w < 20:                     # too small to describe a body
            return None
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        blob = img.transpose(2, 0, 1)[None]
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        emb = emb.astype(np.float32).ravel()
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else None


_embedder: Optional[OSNetEmbedder] = None


def body_model_available() -> bool:
    return os.path.isfile(settings.OSNET_MODEL)


def get_body_embedder() -> Optional[OSNetEmbedder]:
    """Returns the embedder, or None if the OSNet model isn't installed."""
    global _embedder
    if _embedder is None and body_model_available():
        _embedder = OSNetEmbedder()
    return _embedder
