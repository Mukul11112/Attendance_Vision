"""
app.py — Office Video Attendance System (v2 pipeline).

Tabs:
  1. Registration     — employee master data
  2. Enrollment       — face templates from photos or a short video
  3. Process Videos   — batch video processing with live progress
  4. All Cameras      — live attendance across every NVR camera at once
  5. Past Day (NVR)   — replay a chosen day's recordings to mark it after the
                        fact (the Hikvision Playback tab, automated)
  6. Training Images  — sweep videos into one clean folder per employee, faces
                        and bodies kept separate and verified not to mix people
  7. Attendance       — daily roster, review resolution, Excel export

The single-camera "Live Camera" and "Why not recognized?" tabs were removed
from the notebook on 31 Jul 2026; gui/live_camera_tab.py and gui/review_tab.py
are still on disk, so re-adding either is a one-line nb.add() call.

Run:  python app.py
"""
from __future__ import annotations
import logging
import tkinter as tk
from tkinter import ttk

from config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(settings.LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("app")


def main() -> None:
    from database import db_setup
    db_setup.init_db()

    root = tk.Tk()
    root.title(settings.APP_TITLE)
    root.geometry("1180x760")
    root.minsize(1024, 640)
    try:
        root.state("zoomed")          # maximized on Windows
    except Exception:
        pass
    root.configure(bg="#0f1626")
    try:
        root.iconbitmap(settings.ICON_PATH)
    except Exception:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    from gui.registration_tab import build_registration_tab
    from gui.enrollment_tab import build_enrollment_tab
    from gui.batch_processing_tab import build_batch_processing_tab
    from gui.multi_camera_tab import build_multi_camera_tab
    from gui.playback_tab import build_playback_tab
    from gui.harvest_tab import build_harvest_tab
    from gui.attendance_tab import build_attendance_tab

    nb.add(build_registration_tab(nb), text="  1. Registration  ")
    nb.add(build_enrollment_tab(nb), text="  2. Enrollment  ")
    nb.add(build_batch_processing_tab(nb), text="  3. Process Videos  ")
    nb.add(build_multi_camera_tab(nb), text="  4. All Cameras  ")
    nb.add(build_playback_tab(nb), text="  5. Past Day (NVR)  ")
    nb.add(build_harvest_tab(nb), text="  6. Training Images  ")
    nb.add(build_attendance_tab(nb), text="  7. Attendance  ")

    from models.registry import missing_required
    missing = missing_required()
    if missing:
        log.warning("Model weights missing: %s — run scripts/download_models.py",
                    ", ".join(missing))

    root.mainloop()


if __name__ == "__main__":
    main()
