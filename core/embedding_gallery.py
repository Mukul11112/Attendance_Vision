"""
core/embedding_gallery.py
Persistent multi-template face-embedding gallery.

Storage: one .npz per employee in data/face_embeddings/<employee_id>.npz
containing an (N, 512) float32 matrix of L2-normalized embeddings.

Matching (vectorized):
  per employee score = mean of the top-K cosine similarities of that
  employee's templates against the probe. Decision uses BOTH an absolute
  accept threshold and a best-vs-second-best margin, so two look-alike
  employees produce AMBIGUOUS instead of a wrong Present mark.
"""
from __future__ import annotations
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import settings

FACE_EMB_DIR = os.path.join(settings.DATA_DIR, "face_embeddings")
os.makedirs(FACE_EMB_DIR, exist_ok=True)


@dataclass
class Match:
    employee_id: Optional[str]
    similarity: float
    second_id: Optional[str]
    second_similarity: float
    margin: float
    accepted: bool
    ambiguous: bool


class EmbeddingGallery:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._emb: Dict[str, np.ndarray] = {}      # employee_id -> (N,512)
        self.reload()

    # ── persistence ───────────────────────────────────────────────────────
    def reload(self) -> None:
        with self._lock:
            self._emb.clear()
            for fn in os.listdir(FACE_EMB_DIR):
                if not fn.endswith(".npz"):
                    continue
                emp = fn[:-4]
                try:
                    data = np.load(os.path.join(FACE_EMB_DIR, fn))["emb"]
                    if data.ndim == 2 and data.shape[1] == settings.FACE_EMBED_DIM:
                        self._emb[emp] = data.astype(np.float32)
                except Exception:
                    continue

    def _save(self, employee_id: str) -> None:
        np.savez_compressed(os.path.join(FACE_EMB_DIR, f"{employee_id}.npz"),
                            emb=self._emb[employee_id])

    # ── enrollment ────────────────────────────────────────────────────────
    def add_embedding(self, employee_id: str, emb: np.ndarray) -> bool:
        """Add one template. Returns False if rejected as a near-duplicate of
        an existing template (diversity sampling) or if the per-employee cap
        is reached."""
        emb = emb.astype(np.float32).reshape(1, -1)
        with self._lock:
            cur = self._emb.get(employee_id)
            if cur is None:
                self._emb[employee_id] = emb
                self._save(employee_id)
                return True
            if len(cur) >= settings.EMB_PER_EMPLOYEE_MAX:
                return False
            sims = cur @ emb[0]
            if sims.max() >= settings.ENROLL_DUPLICATE_SIM:
                return False                       # nearly identical to a stored one
            self._emb[employee_id] = np.vstack([cur, emb])
            self._save(employee_id)
            return True

    def replace_templates(self, employee_id: str, embs: np.ndarray) -> int:
        """Install a chosen set of templates, discarding what was there.

        Used by best-of-N enrollment: the caller has looked at ALL candidates
        and picked the strongest, most varied subset, so appending one at a
        time (which stops dead at the cap) is the wrong operation.
        """
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
        with self._lock:
            self._emb[employee_id] = arr
            self._save(employee_id)
            return len(arr)

    def remove_employee(self, employee_id: str) -> None:
        with self._lock:
            self._emb.pop(employee_id, None)
            p = os.path.join(FACE_EMB_DIR, f"{employee_id}.npz")
            if os.path.exists(p):
                os.remove(p)

    # ── queries ───────────────────────────────────────────────────────────
    def employee_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._emb.keys())

    def count(self, employee_id: str) -> int:
        with self._lock:
            e = self._emb.get(employee_id)
            return 0 if e is None else len(e)

    def match(self, emb: np.ndarray) -> Match:
        emb = emb.astype(np.float32).ravel()
        with self._lock:
            if not self._emb:
                return Match(None, 0.0, None, 0.0, 0.0, False, False)
            scores: Dict[str, float] = {}
            for emp, mat in self._emb.items():
                sims = mat @ emb                    # cosine (all L2-normalized)
                k = min(settings.GALLERY_TOPK, len(sims))
                topk = np.sort(sims)[-k:]
                scores[emp] = float(topk.mean())
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best = ranked[0]
        second_id, second = (ranked[1] if len(ranked) > 1 else (None, 0.0))
        margin = best - second
        if best < settings.FACE_SIMILARITY_ACCEPT:
            return Match(best_id, best, second_id, second, margin, False, False)
        if len(ranked) > 1 and margin < settings.AMBIGUITY_MARGIN:
            return Match(best_id, best, second_id, second, margin, False, True)
        return Match(best_id, best, second_id, second, margin, True, False)


_gallery: Optional[EmbeddingGallery] = None
_g_lock = threading.Lock()


def get_gallery() -> EmbeddingGallery:
    global _gallery
    with _g_lock:
        if _gallery is None:
            _gallery = EmbeddingGallery()
        return _gallery
