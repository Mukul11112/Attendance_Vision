"""
core/recognition_scheduler.py
Focuses the expensive identification work on ONE person at a time (configurable).

Everyone stays DETECTED and TRACKED every sample — that part is cheap and must
never pause, or people would be lost. What this scheduler serializes is the
expensive step: face detection in the person ROI + embedding + gallery match.

Behaviour:
  - pick the highest-priority unresolved track and keep focusing it until it
    resolves (CONFIRMED/LOCKED, or no longer needs recognition)
  - priority: tracks that already collected accepted votes first (finish the
    person we started), then the largest box (closest person = best face odds)
  - patience: if the focused person yields no usable face for N consecutive
    attempts (looking down / turned away), rotate to the next person and put
    this one on a short cooldown; they will be retried later
  - once resolved -> immediately move to the next person

Pure logic; unit tested without models.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Set

from config import settings


class RecognitionScheduler:
    def __init__(self, focus_limit: int = None, patience: int = None,
                 cooldown: int = None) -> None:
        self.limit = focus_limit or getattr(settings, "RECOGNITION_FOCUS_LIMIT", 1)
        self.patience = patience or getattr(settings, "RECOGNITION_FOCUS_PATIENCE", 10)
        self.cooldown = cooldown or getattr(settings, "RECOGNITION_FOCUS_COOLDOWN", 24)
        self.focus: List[int] = []                # track ids currently focused
        self._stale: Dict[int, int] = {}          # consecutive no-progress attempts
        self._cooldown_until: Dict[int, int] = {} # track id -> sample idx

    # ── selection ─────────────────────────────────────────────────────────
    def pick(self, tracks: Sequence, identities: Dict[int, object],
             sample_idx: int, face_sizes: Optional[Dict[int, float]] = None) -> Set[int]:
        """Track ids that may run recognition this sample.
        face_sizes (optional): tid -> pixel size of the face visible on that
        track RIGHT NOW (0/absent = no face). A visible face beats a big box:
        on ceiling cameras the biggest box is the nearest person, who is often
        facing AWAY — a terrible identification target."""
        cands = [tr for tr in tracks
                 if (ident := identities.get(tr.track_id)) is not None
                 and ident.needs_recognition()]
        if not cands:
            self.focus = []
            return set()
        cand_ids = {tr.track_id for tr in cands}
        fsz = face_sizes or {}

        # keep current focus while still valid and within patience
        self.focus = [tid for tid in self.focus
                      if tid in cand_ids and self._stale.get(tid, 0) < self.patience]

        # preemption: if a focused person shows NO face right now but someone
        # else clearly does, switch immediately instead of burning patience
        if face_sizes is not None and self.focus:
            best_waiting = max((fsz.get(tr.track_id, 0.0) for tr in cands
                                if tr.track_id not in self.focus), default=0.0)
            if best_waiting > 0:
                for tid in list(self.focus):
                    if fsz.get(tid, 0.0) <= 0:
                        self.focus.remove(tid)
                        self._stale.pop(tid, None)   # no cooldown: face may return

        if len(self.focus) < self.limit:
            def prio(tr):
                ident = identities.get(tr.track_id)
                started = 1 if getattr(ident, "n_accepted", 0) > 0 else 0
                x1, y1, x2, y2 = tr.box
                return (started, fsz.get(tr.track_id, 0.0), (x2 - x1) * (y2 - y1))
            eligible = [tr for tr in cands if tr.track_id not in self.focus
                        and self._cooldown_until.get(tr.track_id, 0) <= sample_idx]
            if not eligible:      # everyone waiting -> ignore cooldowns, keep working
                eligible = [tr for tr in cands if tr.track_id not in self.focus]
            eligible.sort(key=prio, reverse=True)
            for tr in eligible[: self.limit - len(self.focus)]:
                self.focus.append(tr.track_id)
                self._stale.setdefault(tr.track_id, 0)
        return set(self.focus)

    # ── feedback after an attempt ─────────────────────────────────────────
    def report(self, tid: int, sample_idx: int, progressed: bool,
               resolved: bool) -> None:
        """progressed = a usable face was embedded/matched this attempt.
        resolved = track no longer needs recognition (confirmed/locked)."""
        if resolved:
            self._drop(tid, sample_idx, cooled=False)
            return
        if progressed:
            self._stale[tid] = 0
        else:
            self._stale[tid] = self._stale.get(tid, 0) + 1
            if self._stale[tid] >= self.patience:
                self._drop(tid, sample_idx, cooled=True)

    def _drop(self, tid: int, sample_idx: int, cooled: bool) -> None:
        if tid in self.focus:
            self.focus.remove(tid)
        self._stale.pop(tid, None)
        if cooled:
            self._cooldown_until[tid] = sample_idx + self.cooldown
