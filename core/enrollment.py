"""
core/enrollment.py
Face enrollment. Two entry points:

  process_image(img)             -> quality-gated embedding for ONE photo
  enroll_from_images(emp, paths) -> add several photos to the gallery
  enroll_from_video(emp, path)   -> sample a short enrollment video, keep only
                                    diverse, high-quality faces (the gallery's
                                    duplicate filter drops near-identical
                                    adjacent frames automatically)

Enrollment is stricter than video-time recognition: reference templates define
identity, so only clearly usable faces are stored.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from config import settings
from core import face_quality
from core.embedding_gallery import get_gallery

# A candidate this far from the person's own dominant face is somebody else's
# photo. 0.40 matches scripts/clean_gallery.py, which was calibrated against
# the contaminated gallery found on 31 Jul 2026.
OUTLIER_SIM = 0.40

# Ceiling on how many video frames are scored before selecting. Enrollment
# clips are short; this only stops someone pointing the tool at an hour of
# footage and waiting all afternoon.
MAX_VIDEO_CANDIDATES = 600


@dataclass
class EnrollResult:
    accepted: bool
    quality: float
    reason: str = ""
    embedding: Optional[np.ndarray] = None


def process_image(img: np.ndarray) -> EnrollResult:
    """Detect the single dominant face, gate quality, return its embedding."""
    from models.face_detector import get_face_detector
    from models.face_embedder import get_embedder

    faces = get_face_detector().detect(img)
    if not faces:
        return EnrollResult(False, 0.0, "no face detected")
    faces.sort(key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]),
               reverse=True)
    f = faces[0]
    x1, y1, x2, y2 = f.box
    crop = img[max(0, y1):y2, max(0, x1):x2]
    q = face_quality.assess(crop, f.landmarks, f.score)
    if not q.passed:
        return EnrollResult(False, q.score, q.reason)
    emb = get_embedder().embed(img, f.landmarks)
    return EnrollResult(True, q.score, "", emb)


def select_best(candidates: List["EnrollResult"], target: int,
                drop_outliers: bool = True) -> tuple:
    """Choose the strongest, most VARIED subset of candidate templates.

    Adding templates one by one stops at the cap, so with 300 photos you keep
    the first 40 you happened to read — which are usually 40 near-identical
    frames from the same few seconds. This picks instead:

      1. outliers are dropped first — a face that disagrees with the person's
         own dominant cluster is somebody else's photo, and one of those in an
         employee's folder is what let a stranger be confirmed on 31 Jul 2026
      2. the highest-quality template seeds the set
      3. each further pick MAXIMISES distance from everything already chosen,
         weighted by quality (farthest-point sampling)

    Step 3 is the important one: coverage of poses and lighting is what makes a
    gallery robust, and 40 photos of one expression is a worse gallery than 15
    varied ones.

    Returns (selected_results, report_dict).
    """
    usable = [c for c in candidates if c.accepted and c.embedding is not None]
    report = {"candidates": len(candidates), "usable": len(usable),
              "outliers": 0, "selected": 0}
    if not usable:
        return [], report

    M = np.stack([c.embedding.astype(np.float32).ravel() for c in usable])
    M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)

    if drop_outliers and len(usable) >= 4:
        S = M @ M.T
        medoid = int(np.argmax(S.sum(axis=1)))          # the dominant face
        keep = [i for i in range(len(usable)) if S[medoid][i] >= OUTLIER_SIM]
        if len(keep) >= max(3, target // 4):            # refuse to gut the set
            report["outliers"] = len(usable) - len(keep)
            usable = [usable[i] for i in keep]
            M = M[keep]

    if len(usable) <= target:
        report["selected"] = len(usable)
        return usable, report

    # Two stages, deliberately. Mixing quality into the diversity score lets a
    # run of sharp near-identical photos crowd out the rare angles — measured:
    # a combined score scored WORSE on coverage than taking the first N.
    #   stage 1: keep a high-quality pool, several times the target
    #   stage 2: pick for coverage alone within that pool
    q = np.array([c.quality for c in usable], dtype=np.float32)
    pool_n = min(len(usable), max(target * 3, target))
    pool = np.argsort(-q)[:pool_n]
    report["quality_pool"] = int(pool_n)
    M = M[pool]
    usable = [usable[i] for i in pool]

    chosen = [int(np.argmax([usable[i].quality for i in range(len(usable))]))]
    sim_to_chosen = M @ M[chosen[0]]
    while len(chosen) < target:
        # farthest-point sampling: take whatever is LEAST like everything held
        score = 1.0 - sim_to_chosen
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        sim_to_chosen = np.maximum(sim_to_chosen, M @ M[nxt])

    report["selected"] = len(chosen)
    return [usable[i] for i in chosen], report


def enroll_best_from_images(employee_id: str, image_paths: List[str],
                            target: Optional[int] = None,
                            replace: bool = True,
                            progress=None) -> dict:
    """Read every image, then install the best `target` templates.

    This is the answer to "I uploaded 300 photos" — all 300 are examined and
    scored, and the gallery ends up with the best spread rather than whichever
    ones happened to be read first.
    """
    target = target or settings.EMB_PER_EMPLOYEE_MAX
    gallery = get_gallery()

    candidates: List[EnrollResult] = []
    unreadable = 0
    for i, p in enumerate(image_paths):
        if progress:
            progress(i + 1, len(image_paths))
        img = cv2.imread(p)
        if img is None:
            unreadable += 1
            continue
        candidates.append(process_image(img))

    picked, report = select_best(candidates, target)
    report["unreadable"] = unreadable
    report["rejected_quality"] = sum(1 for c in candidates if not c.accepted)

    if picked:
        embs = np.stack([c.embedding.astype(np.float32).ravel() for c in picked])
        if replace:
            report["total_templates"] = gallery.replace_templates(employee_id, embs)
        else:
            added = sum(1 for e in embs if gallery.add_embedding(employee_id, e))
            report["added"] = added
            report["total_templates"] = gallery.count(employee_id)
    else:
        report["total_templates"] = gallery.count(employee_id)
    return report


def enroll_from_images(employee_id: str, image_paths: List[str],
                       target: Optional[int] = None) -> dict:
    """Enrol from photos, keeping the BEST subset rather than the first N.

    Every photo is read and scored, then select_best() picks the strongest and
    most varied set. Adding one at a time simply stopped at the cap, so 300
    uploaded photos used to become the first 40 read — usually 40 near-copies
    from the same burst, with the rare angles thrown away unseen.
    """
    gallery = get_gallery()
    target = target or settings.EMB_PER_EMPLOYEE_MAX
    reasons: List[str] = []
    candidates: List[EnrollResult] = []

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            reasons.append(f"{p}: unreadable")
            continue
        r = process_image(img)
        if not r.accepted:
            reasons.append(f"{p}: {r.reason}")
        candidates.append(r)

    picked, report = select_best(candidates, target)
    added = 0
    for c in picked:
        if gallery.add_embedding(employee_id, c.embedding):
            added += 1
    if report.get("outliers"):
        reasons.append(f"{report['outliers']} photo(s) dropped — they do not "
                       f"match this person's other photos")
    return {"added": added,
            "rejected": len(image_paths) - added,
            "examined": len(image_paths), "usable": report.get("usable", 0),
            "selected": report.get("selected", 0),
            "outliers": report.get("outliers", 0),
            "reasons": reasons,
            "total_templates": gallery.count(employee_id)}


def enroll_from_video(employee_id: str, video_path: str,
                      sample_every_s: float = 0.4,
                      max_templates: int = None) -> dict:
    """Enrol from a short video, choosing the best frames in the WHOLE clip.

    This used to stop reading the moment the gallery hit its cap, so a
    three-minute enrollment video was abandoned after its first few seconds and
    the person's other angles were never even looked at. Now the whole clip is
    sampled (bounded by MAX_VIDEO_CANDIDATES) and select_best() picks the
    strongest, most varied frames from all of it.
    """
    max_templates = max_templates or settings.EMB_PER_EMPLOYEE_MAX
    gallery = get_gallery()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"added": 0, "rejected": 0,
                "reasons": [f"cannot open video: {video_path}"],
                "total_templates": gallery.count(employee_id)}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(sample_every_s * fps)), 1)

    candidates: List[EnrollResult] = []
    no_face = 0
    idx = -1
    while len(candidates) < MAX_VIDEO_CANDIDATES:
        if not cap.grab():
            break
        idx += 1
        if idx % step != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        r = process_image(frame)
        if not r.accepted:
            no_face += 1
        candidates.append(r)
    cap.release()

    picked, report = select_best(candidates, max_templates)
    added = 0
    for c in picked:
        if gallery.add_embedding(employee_id, c.embedding):
            added += 1
    reasons = []
    if report.get("outliers"):
        reasons.append(f"{report['outliers']} frame(s) dropped — a different "
                       f"face appeared in the clip")
    return {"added": added, "rejected": len(candidates) - added,
            "examined": len(candidates), "usable": report.get("usable", 0),
            "selected": report.get("selected", 0),
            "outliers": report.get("outliers", 0),
            "unusable_frames": no_face, "reasons": reasons,
            "total_templates": gallery.count(employee_id)}


def enroll_body_from_video(employee_id: str, video_path: str,
                           sample_every_s: float = 0.5) -> dict:
    """BODY enrollment (Phase 2): sample a short video of the employee ALONE
    (walking toward/away, turning, standing) and store diverse OSNet body
    templates. The largest detected person per frame is assumed to be the
    subject — record them alone."""
    from models.body_embedder import get_body_embedder
    from models.person_detector_yolo import get_person_detector
    from core.body_gallery import get_body_gallery

    body = get_body_embedder()
    if body is None:
        return {"added": 0, "rejected": 0, "total_templates": 0,
                "reasons": ["OSNet body model not installed — run "
                            "scripts/download_models.py"]}
    gallery = get_body_gallery()
    pdet = get_person_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"added": 0, "rejected": 0,
                "total_templates": gallery.count(employee_id),
                "reasons": [f"cannot open video: {video_path}"]}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(sample_every_s * fps)), 1)
    added = rejected = 0
    idx = -1
    while gallery.count(employee_id) < settings.BODY_EMB_PER_EMPLOYEE_MAX:
        if not cap.grab():
            break
        idx += 1
        if idx % step != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        dets = pdet.detect(frame)
        if not dets:
            rejected += 1
            continue
        dets.sort(key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
                  reverse=True)
        x1, y1, x2, y2 = [int(v) for v in dets[0].box]
        emb = body.embed(frame[max(0, y1):y2, max(0, x1):x2])
        if emb is None:
            rejected += 1
            continue
        if gallery.add_embedding(employee_id, emb):
            added += 1
        else:
            rejected += 1
    cap.release()
    return {"added": added, "rejected": rejected, "reasons": [],
            "total_templates": gallery.count(employee_id)}
