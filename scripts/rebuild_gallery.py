"""
scripts/rebuild_gallery.py
Rebuilds the face-embedding gallery from data/employee_images/<EMP_ID>/*.jpg.
Use after re-organising enrollment photos or changing quality thresholds.

    python scripts/rebuild_gallery.py            # rebuild everyone
    python scripts/rebuild_gallery.py E001       # rebuild one employee
"""
from __future__ import annotations
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings  # noqa: E402

IMAGES_ROOT = os.path.join(settings.DATA_DIR, "employee_images")


def main() -> int:
    from models.registry import missing_required, status_report
    if missing_required():
        print(status_report()); return 1

    from core.embedding_gallery import get_gallery
    from core.enrollment import enroll_from_images

    only = sys.argv[1] if len(sys.argv) > 1 else None
    if not os.path.isdir(IMAGES_ROOT):
        print(f"No image folder at {IMAGES_ROOT}"); return 1

    gallery = get_gallery()
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for emp in sorted(os.listdir(IMAGES_ROOT)):
        if only and emp != only:
            continue
        folder = os.path.join(IMAGES_ROOT, emp)
        if not os.path.isdir(folder):
            continue
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(exts)]
        gallery.remove_employee(emp)
        r = enroll_from_images(emp, paths)
        print(f"{emp}: {r['added']} templates added, {r['rejected']} rejected "
              f"(total {r['total_templates']})")
        for reason in r["reasons"][:5]:
            print(f"    - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
