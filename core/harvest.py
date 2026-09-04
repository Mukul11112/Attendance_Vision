"""
core/harvest.py
Build per-employee folders of TRAINABLE images from video.

The point of this module is purity. On 31 Jul 2026 the face gallery was found
to hold several different people under single employee IDs — one folder's own
templates were anti-correlated with each other — and a visitor in a white shirt
was being confirmed as an employee as a result. Harvesting more images the same
careless way would simply industrialise that mistake.

So every saved crop must clear FOUR independent rules:

  1. the TRACK is face-locked to that employee (not a guess from one frame)
  2. THIS frame's own match agrees with the track, and is not ambiguous
  3. the match is strong AND well clear of the runner-up (a wide margin)
  4. the crop holds ONE person — a body crop with somebody else inside it is
     rejected outright

and then, after the sweep, every saved face is RE-EMBEDDED and re-matched
against the gallery. Anything whose best match is not the employee whose folder
it sits in is moved to _rejected/. That final pass is what makes "they do not
mix faces" a checked property rather than a hope.

Layout:
    data/training_images/<employee_id>_<name>/face/*.jpg
                                             /body/*.jpg
                        _rejected/<employee_id>/...
"""
from __future__ import annotations
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from config import settings
from core.identity_evidence import CONFIRMED, LOCKED
from core.pipeline import VideoPipeline, VideoJob

log = logging.getLogger("harvest")

OUT_ROOT = os.path.join(settings.DATA_DIR, "training_images")

# Purity thresholds — deliberately stricter than the live matcher. Recognition
# can afford a borderline vote because evidence accumulates over a track; a
# training image cannot, because it becomes ground truth.
MIN_SIMILARITY = 0.55        # vs FACE_SIM_STRONG 0.50
MIN_MARGIN = 0.12            # vs AMBIGUITY_MARGIN 0.06
MIN_QUALITY = 0.45
DEDUP_SIM = 0.97             # near-identical frames add nothing
MIN_FACE_PX = 48
MIN_BODY_PX = 96
BODY_OVERLAP_MAX = 0.02      # any real overlap with another person = reject

# ── unregistered people ────────────────────────────────────────────────────
# Faces that match NO registered employee get their own auto-named folder, so
# they can be reviewed and enrolled later. Two rules keep those folders honest:
#   • a stranger must be clearly nobody on the roster (not merely below the
#     accept bar) — otherwise a registered employee's off-angle face would be
#     filed as a new person
#   • strangers must be separated FROM EACH OTHER, which is what the clustering
#     threshold does. It is set above the 0.395 measured on 31 Jul 2026 as the
#     closest two DIFFERENT registered employees.
UNKNOWN_MAX_ROSTER_SIM = 0.35   # above this we cannot say they are a stranger
UNKNOWN_MERGE_SIM = 0.50        # same-stranger threshold for joining a folder
UNKNOWN_MIN_TRACK_SAMPLES = 3   # ignore blink-and-gone detections
UNKNOWN_DIR = "_unknown"

# ── "find THIS person" ─────────────────────────────────────────────────────
# Seed the sweep with a few photos of someone and their images are collected
# into a named folder — whether or not they are on the roster. Matching against
# the reference photos has to be strict, because everything collected here is
# destined to become training data for that person.
TARGET_MATCH_SIM = 0.50         # same bar as FACE_SIM_STRONG
TARGET_MATCH_MARGIN = 0.08      # and clear of any other seeded person


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (text or "").strip()).strip("_")


@dataclass
class EmployeeHarvest:
    employee_id: str
    name: str = ""
    faces: int = 0
    bodies: int = 0
    rejected: int = 0
    duplicates: int = 0
    embeddings: List[np.ndarray] = field(default_factory=list)
    # similarity to the reference photos, recorded WHEN the crop was saved.
    # Measuring it later by re-embedding the saved crop understates it badly —
    # re-detecting a face inside an already-tight crop loses the landmark
    # alignment the embedder relies on.
    sims: List[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.faces + self.bodies


def build_reference(image_paths: List[str], max_refs: int = 12) -> tuple:
    """Turn a handful of photos of one person into reference embeddings.

    Returns (embeddings (N,512) L2-normalised, report). Photos that disagree
    with the rest are dropped — if two different people are handed in as "one
    person", the odd ones out must not become part of the reference, or the
    sweep will collect both of them into the same folder.
    """
    from core import enrollment

    cands = []
    unreadable = 0
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            unreadable += 1
            continue
        cands.append(enrollment.process_image(img))

    picked, rep = enrollment.select_best(cands, max_refs)
    rep["unreadable"] = unreadable
    rep["no_face"] = sum(1 for c in cands if not c.accepted)
    if not picked:
        return np.zeros((0, settings.FACE_EMBED_DIM), dtype=np.float32), rep
    M = np.stack([c.embedding.astype(np.float32).ravel() for c in picked])
    M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    return M, rep


class UnknownClusters:
    """Auto-named folders for people who are on no roster.

    Persisted: each folder keeps its own cluster.npz of face embeddings, so a
    second sweep — tomorrow, or of another camera — files the same stranger
    back into the SAME folder instead of inventing unknown_009 for someone who
    is already unknown_003.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.join(root, UNKNOWN_DIR)
        os.makedirs(self.root, exist_ok=True)
        self._emb: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name, "cluster.npz")
            if os.path.isfile(p):
                try:
                    self._emb[name] = np.load(p)["emb"].astype(np.float32)
                except Exception:
                    log.warning("unreadable cluster file: %s", p)

    def _next_name(self) -> str:
        n = 1
        existing = set(self._emb) | set(os.listdir(self.root))
        while f"unknown_{n:03d}" in existing:
            n += 1
        return f"unknown_{n:03d}"

    def assign(self, emb: np.ndarray) -> str:
        """Existing folder for this face, or a brand-new one."""
        v = emb.astype(np.float32).ravel()
        v = v / (np.linalg.norm(v) or 1.0)
        best, best_sim = None, -1.0
        for name, M in self._emb.items():
            sim = float((M @ v).max())
            if sim > best_sim:
                best, best_sim = name, sim
        if best is not None and best_sim >= UNKNOWN_MERGE_SIM:
            return best
        name = self._next_name()
        self._emb[name] = v[None]
        os.makedirs(os.path.join(self.root, name), exist_ok=True)
        return name

    def add(self, name: str, emb: np.ndarray) -> None:
        v = emb.astype(np.float32).ravel()
        v = v / (np.linalg.norm(v) or 1.0)
        cur = self._emb.get(name)
        self._emb[name] = v[None] if cur is None else np.vstack([cur, v])
        if len(self._emb[name]) > 40:            # keep the file small
            self._emb[name] = self._emb[name][-40:]

    def save(self) -> None:
        for name, M in self._emb.items():
            d = os.path.join(self.root, name)
            os.makedirs(d, exist_ok=True)
            try:
                np.savez_compressed(os.path.join(d, "cluster.npz"), emb=M)
            except Exception as e:
                log.warning("could not save cluster %s: %s", name, e)

    def folder(self, name: str, kind: str) -> str:
        d = os.path.join(self.root, name, kind)
        os.makedirs(d, exist_ok=True)
        return d


class HarvestJob:
    """Sweep videos and write one clean folder per employee."""

    def __init__(self, videos: List[str], target_per_employee: int = 500,
                 mode: str = "BALANCED", date: str = "2026-01-01",
                 out_root: Optional[str] = None,
                 collect_unknown: bool = True,
                 targets: Optional[Dict[str, np.ndarray]] = None,
                 only_targets: bool = False) -> None:
        # targets: {folder label -> reference embeddings} from photos the user
        # supplied. These are collected whether or not the person is enrolled.
        # only_targets: collect NOTHING else — no roster-wide folders, no
        # stranger clusters. This is the "find the person I gave photos of"
        # mode, where anything extra is just clutter.
        self.targets = dict(targets or {})
        self.only_targets = bool(only_targets)
        self.target_counts: Dict[str, EmployeeHarvest] = {}
        self.videos = list(videos)
        self.target = int(target_per_employee)
        self.mode = mode
        self.date = date
        self.out_root = out_root or OUT_ROOT
        self.collect_unknown = collect_unknown
        self.people: Dict[str, EmployeeHarvest] = {}
        self.unknown: Optional[UnknownClusters] = None
        self.unknown_counts: Dict[str, EmployeeHarvest] = {}
        self._track_cluster: Dict[int, str] = {}   # track -> stranger folder
        self._track_seen: Dict[int, int] = {}      # track -> good samples so far
        self.error: Optional[str] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._names = self._load_names()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, on_status: Optional[Callable[[dict], None]] = None) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(on_status,),
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _load_names() -> Dict[str, str]:
        try:
            from database import db_manager
            return {str(e["employee_id"]): (e.get("name") or "")
                    for e in db_manager.get_employees()}
        except Exception:
            return {}

    def _folder(self, emp: str, kind: str) -> str:
        name = _slug(self._names.get(emp, ""))
        label = f"{emp}_{name}" if name else emp
        d = os.path.join(self.out_root, label, kind)
        os.makedirs(d, exist_ok=True)
        return d

    def _rec(self, emp: str) -> EmployeeHarvest:
        if emp not in self.people:
            self.people[emp] = EmployeeHarvest(emp, self._names.get(emp, ""))
        return self.people[emp]

    @staticmethod
    def _overlap_fraction(box, other) -> float:
        ax1, ay1, ax2, ay2 = box
        bx1, by1, bx2, by2 = other
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        return inter / area

    # ── the purity gate ────────────────────────────────────────────────────
    def _accept(self, s: dict) -> Optional[str]:
        """Employee this sample may be filed under, or None to discard."""
        tracked = s.get("track_identity")
        status = s.get("track_status")
        if tracked is None or status not in (CONFIRMED, LOCKED):
            return None                      # rule 1: track must be locked in
        if not s.get("accepted") or s.get("ambiguous"):
            return None
        if s.get("employee_id") != tracked:
            return None                      # rule 2: this frame must agree
        if s.get("similarity", 0.0) < MIN_SIMILARITY:
            return None                      # rule 3: strong…
        if s.get("margin", 0.0) < MIN_MARGIN:
            return None                      # …and clear of the runner-up
        if s.get("quality", 0.0) < MIN_QUALITY:
            return None
        return tracked

    # ── "find THIS person" from supplied photos ────────────────────────────
    def _match_target(self, emb: np.ndarray) -> Optional[str]:
        """Which seeded person this face is, or None."""
        if not self.targets or emb is None:
            return None
        v = emb.astype(np.float32).ravel()
        v = v / (np.linalg.norm(v) or 1.0)
        scored = sorted(((float((M @ v).max()), label)
                         for label, M in self.targets.items() if len(M)),
                        reverse=True)
        if not scored:
            return None
        best_sim, best_label = scored[0]
        if best_sim < TARGET_MATCH_SIM:
            return None
        if len(scored) > 1 and (best_sim - scored[1][0]) < TARGET_MATCH_MARGIN:
            return None                      # two seeded people look alike here
        return best_label

    def _save_target(self, label: str, s: dict) -> None:
        """Face crop, plus the FULL-BODY crop from the same moment.

        The body crop is taken from a frame where this person's face was just
        matched, so the face is visible in it — which is what makes it usable
        for training rather than an anonymous back view.
        """
        rec = self.target_counts.setdefault(label, EmployeeHarvest(label, label))
        emb = s.get("embedding")
        base = os.path.join(self.out_root, _slug(label))

        if rec.faces < self.target:
            crop = s.get("face_crop")
            if crop is not None and crop.size and min(crop.shape[:2]) >= MIN_FACE_PX:
                if self._is_duplicate(rec, emb):
                    rec.duplicates += 1
                else:
                    d = os.path.join(base, "face")
                    os.makedirs(d, exist_ok=True)
                    if cv2.imwrite(os.path.join(d, f"face_{rec.faces:04d}.jpg"), crop):
                        v = emb.astype(np.float32).ravel()
                        v = v / (np.linalg.norm(v) or 1.0)
                        rec.embeddings.append(v)
                        rec.sims.append(float((self.targets[label] @ v).max()))
                        rec.faces += 1

        if rec.bodies < self.target:
            frame, box = s.get("frame"), s.get("person_box")
            if frame is None or box is None:
                return
            x1, y1, x2, y2 = [int(v) for v in box]
            for tid, other in (s.get("other_boxes") or ()):
                if tid != s.get("track_id") and \
                        self._overlap_fraction((x1, y1, x2, y2), other) > BODY_OVERLAP_MAX:
                    return                   # somebody else is in the shot
            H, W = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) < MIN_BODY_PX // 2 or (y2 - y1) < MIN_BODY_PX:
                return
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                d = os.path.join(base, "body")
                os.makedirs(d, exist_ok=True)
                if cv2.imwrite(os.path.join(d, f"body_{rec.bodies:04d}.jpg"), crop):
                    rec.bodies += 1

    # ── unregistered people ────────────────────────────────────────────────
    def _accept_unknown(self, s: dict) -> bool:
        """Is this definitely somebody who is NOT on the roster?"""
        if s.get("track_identity") is not None:
            return False                     # the recogniser knows them
        if s.get("accepted") or s.get("ambiguous"):
            return False
        if s.get("quality", 0.0) < MIN_QUALITY:
            return False
        # Must be CLEARLY nobody on the roster. A face sitting just under the
        # accept bar is more likely a bad angle of a known employee than a new
        # person, and filing it as a stranger would split that employee in two.
        if s.get("similarity", 0.0) >= UNKNOWN_MAX_ROSTER_SIM:
            return False
        emb = s.get("embedding")
        return emb is not None

    def _cluster_for(self, s: dict) -> Optional[str]:
        """Which stranger folder this sample belongs to.

        A track is one person, so once a track has a folder every later sample
        from it goes to the same place — far more reliable than re-matching
        each frame. Embedding matching is only used the first time, to merge
        this track with a stranger already seen in another video.
        """
        tid = s.get("track_id")
        if tid in self._track_cluster:
            return self._track_cluster[tid]
        seen = self._track_seen.get(tid, 0) + 1
        self._track_seen[tid] = seen
        if seen < UNKNOWN_MIN_TRACK_SAMPLES:
            return None                      # too fleeting to be worth a folder
        name = self.unknown.assign(s["embedding"])
        self._track_cluster[tid] = name
        return name

    def _save_unknown(self, s: dict) -> None:
        name = self._cluster_for(s)
        if not name:
            return
        rec = self.unknown_counts.setdefault(name, EmployeeHarvest(name, "stranger"))
        emb = s.get("embedding")

        if rec.faces < self.target:
            crop = s.get("face_crop")
            if crop is not None and crop.size and min(crop.shape[:2]) >= MIN_FACE_PX:
                if self._is_duplicate(rec, emb):
                    rec.duplicates += 1
                else:
                    p = os.path.join(self.unknown.folder(name, "face"),
                                     f"{name}_face_{rec.faces:04d}.jpg")
                    if cv2.imwrite(p, crop):
                        v = emb.astype(np.float32).ravel()
                        rec.embeddings.append(v / (np.linalg.norm(v) or 1.0))
                        self.unknown.add(name, emb)
                        rec.faces += 1

        if rec.bodies < self.target:
            frame, box = s.get("frame"), s.get("person_box")
            if frame is None or box is None:
                return
            x1, y1, x2, y2 = [int(v) for v in box]
            for tid, other in (s.get("other_boxes") or ()):
                if tid != s.get("track_id") and \
                        self._overlap_fraction((x1, y1, x2, y2), other) > BODY_OVERLAP_MAX:
                    return
            H, W = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) < MIN_BODY_PX // 2 or (y2 - y1) < MIN_BODY_PX:
                return
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                p = os.path.join(self.unknown.folder(name, "body"),
                                 f"{name}_body_{rec.bodies:04d}.jpg")
                if cv2.imwrite(p, crop):
                    rec.bodies += 1

    def _is_duplicate(self, rec: EmployeeHarvest, emb: np.ndarray) -> bool:
        if emb is None or not rec.embeddings:
            return False
        v = emb.astype(np.float32).ravel()
        n = np.linalg.norm(v) or 1.0
        v = v / n
        M = np.stack(rec.embeddings)
        return bool((M @ v).max() >= DEDUP_SIM)

    def _save_face(self, emp: str, s: dict) -> bool:
        rec = self._rec(emp)
        if rec.faces >= self.target:
            return False
        crop = s.get("face_crop")
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if min(h, w) < MIN_FACE_PX:
            return False
        emb = s.get("embedding")
        if self._is_duplicate(rec, emb):
            rec.duplicates += 1
            return False
        path = os.path.join(self._folder(emp, "face"),
                            f"{emp}_face_{rec.faces:04d}.jpg")
        if not cv2.imwrite(path, crop):
            return False
        if emb is not None:
            v = emb.astype(np.float32).ravel()
            rec.embeddings.append(v / (np.linalg.norm(v) or 1.0))
        rec.faces += 1
        return True

    def _save_body(self, emp: str, s: dict) -> bool:
        """Whole-person crop, only when nobody else is inside the box."""
        rec = self._rec(emp)
        if rec.bodies >= self.target:
            return False
        frame = s.get("frame")
        box = s.get("person_box")
        if frame is None or box is None:
            return False
        x1, y1, x2, y2 = [int(v) for v in box]
        # rule 4: a second person inside the crop makes it untrainable
        for tid, other in (s.get("other_boxes") or ()):
            if tid == s.get("track_id"):
                continue
            if self._overlap_fraction((x1, y1, x2, y2), other) > BODY_OVERLAP_MAX:
                return False
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) < MIN_BODY_PX // 2 or (y2 - y1) < MIN_BODY_PX:
            return False
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        path = os.path.join(self._folder(emp, "body"),
                            f"{emp}_body_{rec.bodies:04d}.jpg")
        if not cv2.imwrite(path, crop):
            return False
        rec.bodies += 1
        return True

    # ── worker ─────────────────────────────────────────────────────────────
    def _run(self, on_status) -> None:
        def emit(**kw):
            if not on_status:
                return
            payload = {"phase": "", "message": "",
                       "people": {e: (r.faces, r.bodies)
                                  for e, r in self.people.items()},
                       "unknown": {n: (r.faces, r.bodies)
                                   for n, r in self.unknown_counts.items()},
                       "targets": {n: (r.faces, r.bodies)
                                   for n, r in self.target_counts.items()}}
            payload.update(kw)
            try:
                on_status(payload)
            except Exception:
                log.exception("harvest status callback failed")

        try:
            os.makedirs(self.out_root, exist_ok=True)

            if self.collect_unknown and not self.only_targets:
                self.unknown = UnknownClusters(self.out_root)

            def on_sample(s: dict) -> None:
                if self._stop:
                    return
                # Seeded people first: they may or may not be on the roster,
                # and checking them before the unknown clustering stops the
                # same person being filed twice (named folder AND unknown_00N).
                if self.targets and s.get("quality", 0.0) >= MIN_QUALITY:
                    label = self._match_target(s.get("embedding"))
                    if label:
                        self._save_target(label, s)
                        return
                if self.only_targets:
                    return                   # nobody else is being collected
                emp = self._accept(s)
                if emp is None:
                    if self.unknown is not None and self._accept_unknown(s):
                        self._save_unknown(s)
                    return
                rec = self._rec(emp)
                if rec.faces >= self.target and rec.bodies >= self.target:
                    return
                self._save_face(emp, s)
                self._save_body(emp, s)

            pipe = VideoPipeline(mode=self.mode, want_preview=lambda: False,
                                 on_sample=on_sample)

            for i, path in enumerate(self.videos, 1):
                if self._stop:
                    break
                emit(phase="scanning",
                     message=f"[{i}/{len(self.videos)}] {os.path.basename(path)}")
                job = VideoJob(path=path, date=self.date, start_time="00:00:00",
                               camera_id=f"HARVEST-{i}", camera_location="office",
                               camera_type="office_room")

                def prog(pr) -> None:
                    emit(phase="scanning",
                         message=f"[{i}/{len(self.videos)}] "
                                 f"{os.path.basename(path)} {pr.percent:.0f}% — "
                                 f"{sum(r.total for r in self.people.values())} images")

                try:
                    pipe.process(job, progress_cb=prog, cancel=lambda: self._stop)
                except Exception as e:
                    log.warning("harvest failed on %s: %s", path, e)
                    emit(phase="scanning", message=f"{os.path.basename(path)}: {e}")

            if self.unknown is not None:
                self.unknown.save()          # so a later sweep reuses folders

            if self.only_targets:
                for label, r in self.target_counts.items():
                    if r.sims:
                        emit(phase="checked", message=(
                            f"{label}: {r.faces} face + {r.bodies} body images; "
                            f"match to your photos min={min(r.sims):.2f} "
                            f"mean={sum(r.sims)/len(r.sims):.2f} "
                            f"(bar was {TARGET_MATCH_SIM})"))
                emit(phase="done",
                     message=("stopped early" if self._stop else
                              "done — " + (", ".join(
                                  f"{n}: {r.faces} face / {r.bodies} body"
                                  for n, r in self.target_counts.items())
                                  or "no images matched your photos")))
                return

            emit(phase="verifying", message="re-checking every saved face…")
            moved = self.verify()
            n_unknown = len(self.unknown_counts)
            unknown_imgs = sum(r.total for r in self.unknown_counts.values())
            emit(phase="done",
                 message=("stopped early" if self._stop else
                          f"done — {sum(r.total for r in self.people.values())} "
                          f"images across {len(self.people)} known people; "
                          f"{unknown_imgs} images in {n_unknown} unregistered "
                          f"folder(s); {moved} rejected in verification"))
        except Exception as e:
            self.error = str(e)
            log.exception("harvest job failed")
            emit(phase="error", message=str(e))

    # ── final guarantee ────────────────────────────────────────────────────
    def verify(self) -> int:
        """Re-embed every saved face; move any that match somebody else.

        This is the check that turns "should not mix faces" into a property
        that has actually been tested on the files as written to disk.
        """
        from core.embedding_gallery import get_gallery
        from models.face_detector import get_face_detector
        from models.face_embedder import get_embedder

        gallery = get_gallery()
        det = get_face_detector()
        emb = get_embedder()
        moved = 0

        for emp, rec in self.people.items():
            folder = self._folder(emp, "face")
            reject_dir = os.path.join(self.out_root, "_rejected", emp)
            for fn in sorted(os.listdir(folder)):
                if self._stop or not fn.lower().endswith(".jpg"):
                    continue
                p = os.path.join(folder, fn)
                img = cv2.imread(p)
                if img is None:
                    continue
                faces = det.detect(img)
                if not faces:
                    continue                 # unverifiable, not proof of a mix
                v = emb.embed(img, faces[0].landmarks)
                m = gallery.match(v)
                if m.employee_id != emp or not m.accepted:
                    os.makedirs(reject_dir, exist_ok=True)
                    try:
                        os.replace(p, os.path.join(reject_dir, fn))
                        rec.faces = max(0, rec.faces - 1)
                        rec.rejected += 1
                        moved += 1
                    except OSError as e:
                        log.warning("could not quarantine %s: %s", p, e)
        return moved
