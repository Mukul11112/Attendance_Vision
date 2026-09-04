"""
core/reid.py
Appearance memory used by the pipeline to re-link fragmented tracks.

Phase 1 descriptor: a spatially-split HSV histogram of the person crop.
This is deliberately treated as SUPPORT evidence only — it can lend identity
evidence (damped) from a lost track to a new one, but it can never confirm an
employee by itself, because clothing colour is not a biometric.

Phase 2 replaces appearance_descriptor() with OSNet ONNX embeddings behind
the same interface, so nothing else in the pipeline changes.
"""
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from config import settings

# 3 horizontal stripes (head/torso/legs) x HS histogram
_STRIPES = 3
_H_BINS, _S_BINS = 12, 8


def appearance_descriptor(crop: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalized appearance vector for a person crop, or None if unusable."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 40 or w < 16:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    parts = []
    step = h // _STRIPES
    for i in range(_STRIPES):
        band = hsv[i * step:(i + 1) * step if i < _STRIPES - 1 else h]
        hist = cv2.calcHist([band], [0, 1], None, [_H_BINS, _S_BINS],
                            [0, 180, 0, 256]).flatten()
        parts.append(hist)
    desc = np.concatenate(parts).astype(np.float32)
    n = np.linalg.norm(desc)
    return desc / n if n > 0 else None


class ReIDMemory:
    """Keeps a rolling appearance profile per track and answers 'which recently
    seen track does this new descriptor resemble most?'."""

    def __init__(self, memory_samples: int = None) -> None:
        self.memory = memory_samples or settings.REID_MEMORY_FRAMES
        # track_id -> (deque of descriptors, last_sample_idx)
        self._store: Dict[int, Tuple[Deque[np.ndarray], int]] = {}

    def observe(self, track_id: int, desc: Optional[np.ndarray],
                sample_idx: int) -> None:
        if desc is None:
            return
        dq, _ = self._store.get(track_id, (deque(maxlen=10), sample_idx))
        dq.append(desc)
        self._store[track_id] = (dq, sample_idx)

    def best_link(self, desc: np.ndarray,
                  exclude: Optional[int] = None) -> Tuple[Optional[int], float]:
        """Most similar stored track (cosine of mean profile)."""
        best_id, best_sim = None, 0.0
        for tid, (dq, _) in self._store.items():
            if tid == exclude or not dq:
                continue
            profile = np.mean(np.stack(dq), axis=0)
            n = np.linalg.norm(profile)
            if n == 0:
                continue
            sim = float(np.dot(profile / n, desc))
            if sim > best_sim:
                best_id, best_sim = tid, sim
        return best_id, best_sim

    def can_transfer_identity(self, sim: float) -> bool:
        return (settings.REID_MAX_IDENTITY_TRANSFER
                and sim >= settings.REID_MIN_LINK_FOR_TRANSFER)

    def prune(self, current_sample_idx: int) -> None:
        stale = [tid for tid, (_, last) in self._store.items()
                 if current_sample_idx - last > self.memory]
        for tid in stale:
            del self._store[tid]
