"""
scripts/verify_recognition.py
One-command sanity check of the whole recognition chain on a single photo.

Usage (from the project folder, venv active):
    python scripts\\verify_recognition.py path\\to\\photo.jpg

Prints, in plain language: whether a face was found, its quality, which
employee it matched, the similarity, and whether the video pipeline would
accept that match. Use a photo that is NOT one of the enrolled images.
"""
from __future__ import annotations
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts\\verify_recognition.py <photo.jpg>")
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"[FAIL] File not found: {path}")
        print("       Check the path (and that the extension isn't hidden, e.g. .jpg.jpg)")
        return 1

    import cv2
    img = cv2.imread(path)
    if img is None:
        print(f"[FAIL] File exists but is not a readable image: {path}")
        return 1
    print(f"Image loaded: {img.shape[1]}x{img.shape[0]} px")

    from models.registry import missing_required, status_report
    if missing_required():
        print(status_report())
        return 1

    from core.enrollment import process_image
    from core.embedding_gallery import get_gallery

    r = process_image(img)
    print(f"Face found & quality gate: accepted={r.accepted}  quality={r.quality:.2f}"
          + (f"  reason: {r.reason}" if not r.accepted else ""))
    if not r.accepted:
        print("-> Fix the photo (bigger/sharper/frontal/better light) and retry.")
        return 1

    g = get_gallery()
    ids = g.employee_ids()
    if not ids:
        print("[FAIL] Gallery is EMPTY. Run: python scripts\\rebuild_gallery.py")
        return 1
    print(f"Gallery: {len(ids)} employees "
          f"({', '.join(f'{i}:{g.count(i)}emb' for i in ids)})")

    m = g.match(r.embedding)
    print(f"Best match: {m.employee_id}  similarity={m.similarity:.3f}  "
          f"second={m.second_id} ({m.second_similarity:.3f})  margin={m.margin:.3f}")
    print(f"Thresholds: accept>={settings.FACE_SIMILARITY_ACCEPT}  "
          f"margin>={settings.AMBIGUITY_MARGIN}")
    if m.accepted:
        print(f"[PASS] The pipeline WOULD count this as {m.employee_id}. "
              "Recognition chain is healthy.")
        if m.similarity < 0.5:
            print("       (Similarity is on the low side — adding more varied "
                  "enrollment photos will make video matching more reliable.)")
        return 0
    if m.ambiguous:
        print("[WARN] Match is AMBIGUOUS (two employees too close). Add more "
              "distinct photos for both people.")
    else:
        print("[FAIL] Similarity below the accept threshold — the enrolled "
              "embeddings don't represent this person well. Replace enrollment "
              "photos with better ones and re-run rebuild_gallery.py.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
