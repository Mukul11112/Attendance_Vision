"""
core/byte_track.py
ByteTrack-style multi-object tracker adapted for the SAMPLED pipeline
(frames arrive every 0.25-1.0 s, not at native FPS).

Core ByteTrack idea preserved:
  1. match active tracks against HIGH-confidence detections (IoU + Hungarian)
  2. match the leftover tracks against LOW-confidence detections, so partially
     occluded people keep their track instead of spawning a new one
  3. unmatched high-confidence detections start new tentative tracks
  4. tracks survive TRACK_MAX_AGE missed samples in a lost buffer before dying

Motion model: constant velocity on box centre + smoothed size. A full Kalman
filter adds little at 0.5 s sampling; the linear predictor keeps IoU matching
meaningful while people walk between samples.

Pure logic (numpy + scipy only) -> unit tested without any model weights.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import settings

Box = Tuple[float, float, float, float]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two (N,4) and (M,4) xyxy arrays -> (N,M)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]; b = boxes_b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return (inter / np.clip(union, 1e-9, None)).astype(np.float32)


@dataclass
class Track:
    track_id: int
    box: Box
    score: float
    hits: int = 1                 # matched samples so far
    age: int = 1                  # samples since birth
    missed: int = 0               # consecutive missed samples
    velocity: Tuple[float, float] = (0.0, 0.0)   # centre px / sample
    confirmed: bool = False
    _history: List[Box] = field(default_factory=list)

    @property
    def centre(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def predicted_box(self) -> Box:
        """Constant-velocity prediction of where the box should be now."""
        x1, y1, x2, y2 = self.box
        dx, dy = self.velocity
        steps = self.missed + 1
        return (x1 + dx * steps, y1 + dy * steps, x2 + dx * steps, y2 + dy * steps)

    def update(self, box: Box, score: float) -> None:
        pcx, pcy = self.centre
        self.box = box
        ncx, ncy = self.centre
        # exponential smoothing of velocity: robust to a single jumpy detection
        self.velocity = (0.6 * (ncx - pcx) + 0.4 * self.velocity[0],
                         0.6 * (ncy - pcy) + 0.4 * self.velocity[1])
        self.score = score
        self.hits += 1
        self.missed = 0
        if self.hits >= settings.TRACK_MIN_HITS:
            self.confirmed = True
        self._history.append(box)
        if len(self._history) > 120:
            self._history = self._history[-120:]

    def mark_missed(self) -> None:
        self.missed += 1

    @property
    def is_lost(self) -> bool:
        return self.missed > 0


class ByteTracker:
    """update(detections) -> list of ACTIVE, CONFIRMED tracks this sample."""

    def __init__(self,
                 high_thresh: float = None, low_thresh: float = None,
                 match_iou: float = None, max_age: int = None) -> None:
        self.high_thresh = high_thresh if high_thresh is not None else settings.TRACK_HIGH_THRESH
        self.low_thresh = low_thresh if low_thresh is not None else settings.TRACK_LOW_THRESH
        self.match_iou = match_iou if match_iou is not None else settings.TRACK_MATCH_IOU
        self.max_age = max_age if max_age is not None else settings.TRACK_MAX_AGE
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    # ── helpers ───────────────────────────────────────────────────────────
    def _match(self, tracks: List[Track], dets: List[Tuple[Box, float]],
               min_iou: float) -> Tuple[List[Tuple[Track, int]], List[Track], List[int]]:
        """Hungarian IoU matching. Returns (matches, unmatched_tracks, unmatched_det_idx)."""
        if not tracks or not dets:
            return [], list(tracks), list(range(len(dets)))
        tb = np.array([t.predicted_box() for t in tracks], dtype=np.float32)
        db = np.array([d[0] for d in dets], dtype=np.float32)
        iou = iou_matrix(tb, db)
        row, col = linear_sum_assignment(-iou)
        matches, um_t, um_d = [], [], set(range(len(dets)))
        matched_t = set()
        for r, c in zip(row, col):
            if iou[r, c] >= min_iou:
                matches.append((tracks[r], c))
                matched_t.add(r)
                um_d.discard(c)
        um_t = [t for i, t in enumerate(tracks) if i not in matched_t]
        return matches, um_t, sorted(um_d)

    # ── main step ─────────────────────────────────────────────────────────
    def update(self, detections: Sequence[Tuple[Box, float]]) -> List[Track]:
        high = [d for d in detections if d[1] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d[1] < self.high_thresh]

        all_tracks = list(self._tracks.values())

        # 1) all live tracks (active + lost buffer) vs HIGH detections
        matches, unmatched_tracks, um_high = self._match(all_tracks, high, self.match_iou)
        for tr, di in matches:
            tr.update(high[di][0], high[di][1])

        # 2) remaining tracks vs LOW detections (occlusion recovery)
        matches2, unmatched_tracks, _ = self._match(unmatched_tracks, low, self.match_iou)
        for tr, di in matches2:
            tr.update(low[di][0], low[di][1])

        # 2.5) rescue association: detection boxes for SEATED people flicker
        # between "upper body" and "full body with chair" sizes, which breaks
        # IoU even though it's obviously the same person. Match leftover
        # tracks to leftover HIGH detections by centre distance instead, so
        # the track survives instead of fragmenting into a new ID.
        if unmatched_tracks and um_high:
            used = set()
            still_unmatched = []
            for tr in unmatched_tracks:
                px1, py1, px2, py2 = tr.predicted_box()
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                diag = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
                best_i, best_d = None, float("inf")
                for di in um_high:
                    if di in used:
                        continue
                    bx = high[di][0]
                    cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
                    d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                    if d < best_d:
                        best_i, best_d = di, d
                if best_i is not None and best_d <= 0.5 * diag:
                    tr.update(high[best_i][0], high[best_i][1])
                    used.add(best_i)
                else:
                    still_unmatched.append(tr)
            unmatched_tracks = still_unmatched
            um_high = [di for di in um_high if di not in used]

        # 3) unmatched tracks age; drop after max_age missed samples
        for tr in unmatched_tracks:
            tr.mark_missed()
            if tr.missed > self.max_age or (not tr.confirmed and tr.missed > 2):
                self._tracks.pop(tr.track_id, None)

        # 4) unmatched HIGH detections start new tracks
        for di in um_high:
            tr = Track(track_id=self._next_id, box=high[di][0], score=high[di][1])
            if settings.TRACK_MIN_HITS <= 1:
                tr.confirmed = True
            self._tracks[self._next_id] = tr
            self._next_id += 1

        for tr in self._tracks.values():
            tr.age += 1

        # return only tracks that are confirmed AND matched this sample
        return [t for t in self._tracks.values() if t.confirmed and t.missed == 0]

    def coast(self) -> List[Track]:
        """Predicted current tracks WITHOUT consuming an update cycle — used on
        samples where person detection is skipped (seated offices barely move,
        so detection every sample is wasted CPU)."""
        return [t for t in self._tracks.values()
                if t.confirmed and t.missed <= self.max_age]

    # exposure for diagnostics / conflict resolution
    def all_tracks(self) -> List[Track]:
        return list(self._tracks.values())
