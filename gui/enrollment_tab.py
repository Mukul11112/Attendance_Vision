"""gui/enrollment_tab.py — enroll face templates from photos or a short video.
Enrollment runs in a worker thread; results are pumped back to the UI."""
from __future__ import annotations
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BG, FG, SUB = "#0f1626", "#cfe0ff", "#9fb7d8"


def build_enrollment_tab(parent):
    frame = tk.Frame(parent, bg=BG)
    q: queue.Queue = queue.Queue()

    tk.Label(frame, text="Face Enrollment", bg=BG, fg=FG,
             font=("Verdana", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
    tk.Label(frame, text=("Give each employee 5-15 varied photos (front, slight "
                          "left/right, slight up/down, office lighting) or a short "
                          "10-20 s video of them slowly turning their head.\n"
                          "Blurry/dark/extreme-angle samples and near-duplicates "
                          "are rejected automatically."),
             bg=BG, fg=SUB, justify="left").pack(anchor="w", padx=12)

    row = tk.Frame(frame, bg=BG); row.pack(anchor="w", padx=12, pady=8)
    tk.Label(row, text="Employee", bg=BG, fg=SUB).pack(side="left")
    emp_box = ttk.Combobox(row, width=30, state="readonly")
    emp_box.pack(side="left", padx=6)

    # action buttons BEFORE the expanding log box, so they can never be
    # pushed below the visible window area
    def enroll_body_video():
        emp = _selected_id()
        if not emp or not _check_models():
            return
        path = filedialog.askopenfilename(
            title="Select BODY enrollment video (person alone: walking, turning)",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if not path:
            return
        from core.enrollment import enroll_body_from_video
        _log(f"BODY enrollment from video for {emp} (OSNet)…")
        _run(enroll_body_from_video, emp, path)

    btns = tk.Frame(frame, bg=BG); btns.pack(anchor="w", padx=12, pady=6)

    gallery_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=gallery_var, bg=BG, fg=SUB,
             justify="left").pack(anchor="w", padx=12)

    log_box = tk.Text(frame, height=14, width=90, bg="#0a1020", fg=FG)
    log_box.pack(padx=12, pady=8, fill="both", expand=True)

    def _log(msg):
        log_box.insert("end", msg + "\n"); log_box.see("end")

    def refresh_employees():
        from database import db_manager, db_setup
        from core.embedding_gallery import get_gallery
        db_setup.init_db()
        emps = db_manager.get_employees()
        emp_box["values"] = [f"{e['employee_id']} — {e['name']}" for e in emps]
        g = get_gallery()
        from core.body_gallery import get_body_gallery
        bg = get_body_gallery()
        ids = sorted(set(g.employee_ids()) | set(bg.employee_ids()))
        counts = ", ".join(f"{i}: {g.count(i)} face / {bg.count(i)} body"
                           for i in ids)
        gallery_var.set("Gallery: " + (counts or "empty — enroll someone"))

    def _selected_id():
        v = emp_box.get()
        if not v:
            messagebox.showwarning("Pick employee", "Select an employee first "
                                   "(register them on the Registration tab).")
            return None
        return v.split(" — ")[0]

    def _check_models() -> bool:
        from models.registry import missing_required, status_report
        if missing_required():
            messagebox.showerror("Models missing", status_report()
                                 + "\n\nRun:  python scripts/download_models.py")
            return False
        return True

    def _run(fn, *args):
        def worker():
            try:
                q.put(("result", fn(*args)))
            except Exception as e:
                q.put(("error", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    def enroll_images():
        emp = _selected_id()
        if not emp or not _check_models():
            return
        paths = filedialog.askopenfilenames(
            title="Select face photos",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if not paths:
            return
        from core.enrollment import enroll_from_images
        _log(f"Enrolling {len(paths)} photos for {emp}…")
        _run(enroll_from_images, emp, list(paths))

    def enroll_video():
        emp = _selected_id()
        if not emp or not _check_models():
            return
        path = filedialog.askopenfilename(
            title="Select enrollment video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if not path:
            return
        from core.enrollment import enroll_from_video
        _log(f"Enrolling from video for {emp} (diversity sampling)…")
        _run(enroll_from_video, emp, path)

    tk.Button(btns, text="🖼 Enroll from photos", command=enroll_images,
              bg="#2f6fd0", fg="white").pack(side="left", padx=4)
    tk.Button(btns, text="🎞 Enroll from video", command=enroll_video,
              bg="#2f6fd0", fg="white").pack(side="left", padx=4)
    tk.Button(btns, text="🧍 Enroll BODY from video", command=enroll_body_video,
              bg="#6a4fd0", fg="white").pack(side="left", padx=4)
    tk.Button(btns, text="⟳ Refresh", command=refresh_employees).pack(side="left", padx=4)

    def _pump():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "result":
                    r = payload
                    if r.get("examined"):
                        # best-of-N: say what was looked at, not just what stuck
                        _log(f"Examined {r['examined']}, {r.get('usable', 0)} usable, "
                             f"picked the best {r.get('selected', 0)}"
                             + (f", dropped {r['outliers']} that look like "
                                f"someone else" if r.get("outliers") else "") + ".")
                    _log(f"Added {r['added']} templates, rejected {r['rejected']}. "
                         f"Total now: {r['total_templates']}.")
                    for reason in r.get("reasons", [])[:8]:
                        _log(f"   - {reason}")
                    refresh_employees()
                else:
                    _log(f"ERROR: {payload}")
        except queue.Empty:
            pass
        frame.after(150, _pump)

    frame.after(200, _pump)
    frame.bind("<Visibility>", lambda e: refresh_employees())
    refresh_employees()
    return frame
