"""
core/body_gallery.py
PER-DAY multi-template BODY embedding gallery (OSNet vectors), one .npz per
employee under data/body_embeddings/<YYYY-MM-DD>/. Mirrors the face gallery,
but its matches are SUPPORT evidence only — thresholds live in the BODY_*
settings and the evidence layer guarantees body matches can never confirm an
identity alone.

Why per-day (31 Jul 2026): a body embedding describes CLOTHES as much as build,
so it is only valid for the day it was harvested. The gallery used to be one
flat folder that accumulated forever — employee 01's stored body was a shirt
from 17 days earlier, and it was still being matched against people walking
past today. Day folders make yesterday's outfit simply invisible.
"""
from __future__ import annotations
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from config import settings

BODY_EMB_ROOT = os.path.join(settings.DATA_DIR, "body_embeddings")
os.makedirs(BODY_EMB_ROOT, exist_ok=True)


def day_dir(day: Optional[str] = None) -> str:
    d = os.path.join(BODY_EMB_ROOT, day or date.today().isoformat())
    os.makedirs(d, exist_ok=True)
    return d


@dataclass
class BodyMatch:
    employee_id: Optional[str]
    similarity: float
    margin: float
    supported: bool          # similarity + margin cleared support thresholds


class BodyGallery:
    def __init__(self, day: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._emb: Dict[str, np.ndarray] = {}
        self.day = day or date.today().isoformat()
        self.reload()

    def set_day(self, day: str) -> None:
        """Switch to another day's wardrobe (used when replaying a past day)."""
        if day == self.day:
            return
        self.day = day
        self.reload()

    def _dir(self) -> str:
        return day_dir(self.day)

    def reload(self) -> None:
        with self._lock:
            self._emb.clear()
            d = day_dir(self.day)
            for fn in os.listdir(d):
                if not fn.endswith(".npz"):
                    continue
                try:
                    data = np.load(os.path.join(d, fn))["emb"]
                    if data.ndim == 2 and data.shape[1] == settings.BODY_EMBED_DIM:
                        self._emb[fn[:-4]] = data.astype(np.float32)
                except Exception:
                    continue

    def _save(self, emp: str) -> None:
        np.savez_compressed(os.path.join(self._dir(), f"{emp}.npz"),
                            emb=self._emb[emp])

    def add_embedding(self, emp: str, emb: np.ndarray) -> bool:
        emb = emb.astype(np.float32).reshape(1, -1)
        with self._lock:
            cur = self._emb.get(emp)
            if cur is None:
                self._emb[emp] = emb; self._save(emp); return True
            if len(cur) >= settings.BODY_EMB_PER_EMPLOYEE_MAX:
                return False
            if (cur @ emb[0]).max() >= settings.BODY_ENROLL_DUPLICATE_SIM:
                return False
            self._emb[emp] = np.vstack([cur, emb]); self._save(emp)
            return True

    def remove_employee(self, emp: str) -> None:
        with self._lock:
            self._emb.pop(emp, None)
            p = os.path.join(self._dir(), f"{emp}.npz")
            if os.path.exists(p):
                os.remove(p)

    def employee_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._emb.keys())

    def count(self, emp: str) -> int:
        with self._lock:
            e = self._emb.get(emp)
            return 0 if e is None else len(e)

    def match(self, emb: np.ndarray) -> BodyMatch:
        emb = emb.astype(np.float32).ravel()
        with self._lock:
            if not self._emb:
                return BodyMatch(None, 0.0, 0.0, False)
            scores = {}
            for emp, mat in self._emb.items():
                sims = mat @ emb
                k = min(3, len(sims))
                scores[emp] = float(np.sort(sims)[-k:].mean())
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best - second
        supported = (best >= settings.BODY_SIMILARITY_SUPPORT
                     and (len(ranked) == 1
                          or margin >= settings.BODY_AMBIGUITY_MARGIN))
        return BodyMatch(best_id, best, margin, supported)


_gallery: Optional[BodyGallery] = None
_g_lock = threading.Lock()


def get_body_gallery(day: Optional[str] = None) -> BodyGallery:
    global _gallery
    with _g_lock:
        if _gallery is None:
            _gallery = BodyGallery(day)
        elif day and day != _gallery.day:
            _gallery.set_day(day)
        elif not day and _gallery.day != date.today().isoformat():
            _gallery.set_day(date.today().isoformat())   # rolled past midnight
        return _gallery


def migrate_legacy_flat_files() -> int:
    """Move pre-31-Jul flat <emp>.npz files out of the way.

    They carry no date, so there is no honest day to file them under, and
    leaving them in the root would make them load for every day. They go to
    body_embeddings/_legacy_undated/ rather than being deleted.
    """
    moved = 0
    legacy = os.path.join(BODY_EMB_ROOT, "_legacy_undated")
    for fn in os.listdir(BODY_EMB_ROOT):
        src = os.path.join(BODY_EMB_ROOT, fn)
        if os.path.isfile(src) and fn.endswith(".npz"):
            os.makedirs(legacy, exist_ok=True)
            shutil.move(src, os.path.join(legacy, fn))
            moved += 1
    return moved
