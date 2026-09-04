"""
scripts/diagnose_video.py
Answers: "why is nobody being recognized in this video?"

Samples frames from a video and reports, stage by stage:
  1. gallery state (who is enrolled, how many templates)
  2. person detections per sample
  3. faces found inside person boxes, and their pixel sizes
  4. quality-gate outcomes with reasons
  5. gallery similarity of every face that passed quality

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\diagnose_video.py "C:\\path\\video.mp4"
  .\\.venv\\Scripts\\python.exe scripts\\diagnose_video.py "video.mp4" --samples 60
"""
from __future__ import annotations
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from config import settings  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 1
    video_path = sys.argv[1]
    n_samples = 40
    if "--samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--samples") + 1])

    from models.registry import missing_required, status_report
    if missing_required():
        print(status_report()); return 1

    from core.embedding_gallery import get_gallery
    from core import face_quality
    from models.person_detector_yolo import get_person_detector
    from models.face_detector import get_face_detector
    from models.face_embedder import get_embedder

    # ── 1. gallery ─────────────────────────────────────────────────────────
    g = get_gallery()
    ids = g.employee_ids()
    print("=" * 68)
    print("GALLERY")
    if not ids:
        print("  EMPTY — nobody is enrolled. Recognition is impossible until")
        print("  employees have face templates. Fix enrollment first.")
    for i in ids:
        print(f"  {i}: {g.count(i)} templates")
    print(f"  accept threshold={settings.FACE_SIMILARITY_ACCEPT}  "
          f"ambiguity margin={settings.AMBIGUITY_MARGIN}  "
          f"min face={settings.MIN_FACE_SIZE}px  "
          f"quality min={settings.FACE_QUALITY_MIN}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}"); return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("=" * 68)
    print(f"VIDEO  {os.path.basename(video_path)}  {W}x{H}  {total} frames")

    pdet = get_person_detector()
    fdet = get_face_detector()
    femb = get_embedder()

    persons_per_sample = []
    face_sizes = []                    # min(w,h) of every detected face
    quality_reasons = Counter()
    passed_faces = 0
    sims_best = []                     # (best_sim, best_id, margin, accepted, ambiguous)
    samples_with_face = 0

    idxs = np.linspace(0, max(total - 1, 0), num=min(n_samples, max(total, 1)),
                       dtype=int)
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        dets = pdet.detect(frame)
        persons_per_sample.append(len(dets))
        found_face = False
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.box]
            pad = int(0.05 * (y2 - y1))
            roi = frame[max(0, y1 - pad):y2, max(0, x1 - pad):min(frame.shape[1], x2 + pad)]
            faces = fdet.detect(roi, offset=(max(0, x1 - pad), max(0, y1 - pad)))
            for f in faces:
                fx1, fy1, fx2, fy2 = f.box
                size = min(fx2 - fx1, fy2 - fy1)
                face_sizes.append(size)
                found_face = True
                crop = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                q = face_quality.assess(crop, f.landmarks, f.score)
                if not q.passed:
                    quality_reasons[q.reason.split("(")[0].strip()] += 1
                    continue
                passed_faces += 1
                if ids:
                    m = g.match(femb.embed(frame, f.landmarks))
                    sims_best.append((m.similarity, m.employee_id, m.margin,
                                      m.accepted, m.ambiguous))
        if found_face:
            samples_with_face += 1
        print(f"\r  scanning sample {k + 1}/{len(idxs)}…", end="")
    print()
    cap.release()

    # ── report ─────────────────────────────────────────────────────────────
    print("=" * 68)
    print("PERSON DETECTION")
    if persons_per_sample:
        print(f"  people per sampled frame: min={min(persons_per_sample)} "
              f"avg={np.mean(persons_per_sample):.1f} max={max(persons_per_sample)}")
    print("=" * 68)
    print("FACES")
    if not face_sizes:
        print("  NO faces detected in any sampled frame.")
        print("  With people visible, this means faces are too small, too")
        print("  downward-facing, or too blurred at this camera distance.")
    else:
        a = np.array(face_sizes)
        print(f"  faces detected: {len(a)} across {samples_with_face}/{len(idxs)} samples")
        print(f"  face size (min side, px): p10={np.percentile(a,10):.0f} "
              f"median={np.median(a):.0f} p90={np.percentile(a,90):.0f} max={a.max():.0f}")
        print(f"  >= MIN_FACE_SIZE({settings.MIN_FACE_SIZE}px): "
              f"{(a >= settings.MIN_FACE_SIZE).mean()*100:.0f}%")
        print(f"  passed quality gate: {passed_faces}")
        if quality_reasons:
            print("  rejection reasons:")
            for r, c in quality_reasons.most_common():
                print(f"    {c:4d}  {r}")
    print("=" * 68)
    print("GALLERY MATCHING (faces that passed quality)")
    if not ids:
        print("  skipped — gallery empty.")
    elif not sims_best:
        print("  no faces passed quality, so matching never ran.")
    else:
        s = np.array([x[0] for x in sims_best])
        acc = sum(1 for x in sims_best if x[3])
        amb = sum(1 for x in sims_best if x[4])
        print(f"  attempts={len(s)}  accepted={acc}  ambiguous={amb}")
        print(f"  best-similarity distribution: p50={np.percentile(s,50):.3f} "
              f"p90={np.percentile(s,90):.3f} max={s.max():.3f} "
              f"(accept threshold {settings.FACE_SIMILARITY_ACCEPT})")
        who = Counter(x[1] for x in sims_best if x[3])
        if who:
            print(f"  accepted matches by employee: {dict(who)}")
    print("=" * 68)
    print("VERDICT HINTS")
    if not ids:
        print("  -> Enroll employees first. Nothing else matters until then.")
    elif not face_sizes:
        print("  -> Camera never yields a detectable face: recognition cannot")
        print("     work on this footage. Consider a camera that sees faces")
        print("     (entry path / lower mounting) or clips where people look up.")
    elif face_sizes and passed_faces == 0:
        print("  -> Faces are found but ALL fail quality. See reasons above:")
        print("     'too small' -> people too far; 'extreme pitch/yaw' -> looking")
        print("     down; 'blurry' -> motion/compression. Threshold changes can")
        print("     help ONLY if values are near the limits.")
    elif sims_best and max(x[0] for x in sims_best) < settings.FACE_SIMILARITY_ACCEPT:
        print("  -> Faces pass quality but similarity is below threshold:")
        print("     enrollment photos probably don't resemble CCTV conditions.")
        print("     Add enrollment shots taken from the CCTV itself (frames where")
        print("     the person looks toward the camera).")
    else:
        print("  -> Chain looks healthy; if attendance is still empty, send this")
        print("     full output for analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
