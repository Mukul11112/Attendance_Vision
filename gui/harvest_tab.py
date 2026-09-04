"""
gui/harvest_tab.py
Build a training-image folder for ONE person from a few photos of them.

Give the person a name, hand in a handful of clear photos, add the videos to
search, and their folder is filled with the clearest usable images found:

    data/training_images/<name>/face/   tight face crops
                               /body/   full-body crops from the SAME moments,
                                        so the face is visible in them too

Nothing else is collected — no roster-wide sweep, no stranger folders. What
does NOT get saved is the point: see core/harvest.py for the match bar, the
quality gate, the one-person-per-crop rule and near-duplicate rejection.
"""
from __future__ import annotations
import logging
import os
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from config import settings
from core import harvest

log = logging.getLogger("harvest_tab")


def build_harvest_tab(parent, tts=None):
    frame = tk.Frame(parent, bg="#0f1626")
    q: "queue.Queue" = queue.Queue()
    state = {"job": None, "videos": [], "name": "", "refs": None}

    # ── step 1: who ─────────────────────────────────────────────────────────
    bar = tk.Frame(frame, bg="#0f1626"); bar.pack(fill="x", padx=10, pady=(8, 2))
    tk.Label(bar, text="Training Images — collect one person from their photos",
             bg="#0f1626", fg="#cfe0ff", font=("Verdana", 13, "bold")).pack(side="left")

    step1 = tk.Frame(frame, bg="#0f1626"); step1.pack(fill="x", padx=10, pady=(6, 2))
    tk.Label(step1, text="1.", bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 10, "bold")).pack(side="left")
    person_btn = tk.Button(step1, text="👤 Choose person's photos", bg="#2f6fd0",
                           fg="white", font=("Verdana", 10, "bold"))
    person_btn.pack(side="left", padx=6)
    person_var = tk.StringVar(value="no photos chosen yet")
    tk.Label(step1, textvariable=person_var, bg="#0f1626",
             fg="#9fb7d8").pack(side="left", padx=8)

    # ── step 2: where to look ───────────────────────────────────────────────
    step2 = tk.Frame(frame, bg="#0f1626"); step2.pack(fill="x", padx=10, pady=2)
    tk.Label(step2, text="2.", bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 10, "bold")).pack(side="left")
    add_btn = tk.Button(step2, text="🎞 Add videos to search", bg="#1b2740",
                        fg="#cfe0ff", relief="flat")
    add_btn.pack(side="left", padx=6)
    clear_btn = tk.Button(step2, text="Clear", bg="#3a3f52", fg="#cfe0ff",
                          relief="flat")
    clear_btn.pack(side="left")
    tk.Label(step2, text="Images to collect", bg="#0f1626",
             fg="#9fb7d8").pack(side="left", padx=(18, 4))
    target_var = tk.StringVar(value="500")
    tk.Entry(step2, textvariable=target_var, width=6).pack(side="left")
    tk.Label(step2, text="Mode", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(14, 4))
    mode_var = tk.StringVar(value=settings.DEFAULT_PROCESSING_MODE)
    ttk.Combobox(step2, textvariable=mode_var, state="readonly", width=10,
                 values=list(settings.PROCESSING_MODES.keys())).pack(side="left")

    # ── step 3: go ──────────────────────────────────────────────────────────
    step3 = tk.Frame(frame, bg="#0f1626"); step3.pack(fill="x", padx=10, pady=(4, 2))
    tk.Label(step3, text="3.", bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 10, "bold")).pack(side="left")
    start_btn = tk.Button(step3, text="▶ Collect this person's images",
                          bg="#1f7a3f", fg="white", font=("Verdana", 10, "bold"),
                          width=26)
    start_btn.pack(side="left", padx=6)
    stop_btn = tk.Button(step3, text="■ Stop", bg="#7a2f2f", fg="white",
                         font=("Verdana", 10, "bold"), width=9, state="disabled")
    stop_btn.pack(side="left", padx=3)
    tk.Button(step3, text="📂 Open folder", bg="#1b2740", fg="#cfe0ff", relief="flat",
              command=lambda: _open_out()).pack(side="left", padx=(16, 0))
    tk.Button(step3, text="⤓ Add to gallery", bg="#2f5aa8", fg="white",
              font=("Verdana", 9, "bold"),
              command=lambda: _enrol_best()).pack(side="left", padx=8)

    tk.Label(frame, text="Saved only when the face clearly matches your photos, "
                         "the crop is sharp enough, and nobody else is standing "
                         "in it. Body shots are taken from the same moments as "
                         "the face matches, so the face is visible in them.",
             bg="#0f1626", fg="#6f86a8", justify="left", wraplength=1100,
             anchor="w").pack(fill="x", padx=10, pady=(4, 6))

    # ── body ────────────────────────────────────────────────────────────────
    body = tk.Frame(frame, bg="#0f1626"); body.pack(fill="both", expand=True,
                                                    padx=10, pady=4)
    left = tk.Frame(body, bg="#0f1626"); left.pack(side="left", fill="both", expand=True)
    tk.Label(left, text="Videos to search", bg="#0f1626", fg="#9fb7d8",
             font=("Verdana", 9, "bold"), anchor="w").pack(fill="x")
    vid_list = tk.Listbox(left, height=7, bg="#0a1020", fg="#cfe0ff",
                          highlightthickness=0, borderwidth=0, activestyle="none",
                          font=("Consolas", 9))
    vid_list.pack(fill="both", expand=True, pady=(2, 6))
    tk.Label(left, text="Activity", bg="#0f1626", fg="#9fb7d8",
             font=("Verdana", 9, "bold"), anchor="w").pack(fill="x")
    log_box = tk.Listbox(left, height=9, bg="#0a1020", fg="#8fa8c8",
                         highlightthickness=0, borderwidth=0, activestyle="none",
                         font=("Consolas", 9))
    log_box.pack(fill="both", expand=True, pady=2)

    right = tk.Frame(body, bg="#0f1626"); right.pack(side="right", fill="y", padx=(12, 0))
    tk.Label(right, text="Collected", bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 10, "bold"), anchor="w").pack(fill="x")
    tk.Label(right, text=f"{'PERSON':<20}{'FACE':>6}{'BODY':>6}", bg="#0f1626",
             fg="#6f86a8", font=("Consolas", 9), anchor="w").pack(fill="x", pady=(4, 0))
    counts = tk.Listbox(right, height=16, width=34, bg="#0a1020", fg="#cfe0ff",
                        highlightthickness=0, borderwidth=0, activestyle="none",
                        font=("Consolas", 10))
    counts.pack(fill="y", expand=True)

    status_var = tk.StringVar(value="Start by choosing a person's photos.")
    tk.Label(frame, textvariable=status_var, bg="#0f1626", fg="#cfe0ff",
             anchor="w").pack(fill="x", padx=10, pady=(0, 8))

    def _log(msg: str) -> None:
        log_box.insert(tk.END, f" {msg}")
        if log_box.size() > 400:
            log_box.delete(0)
        log_box.see(tk.END)

    def _open_out() -> None:
        d = os.path.join(harvest.OUT_ROOT, harvest._slug(state["name"])) \
            if state["name"] else harvest.OUT_ROOT
        d = d if os.path.isdir(d) else harvest.OUT_ROOT
        os.makedirs(d, exist_ok=True)
        try:
            subprocess.Popen(["explorer", d])
        except Exception as e:
            messagebox.showinfo("Folder", f"{d}\n\n({e})")

    # ── step 1 handler ──────────────────────────────────────────────────────
    def choose_person() -> None:
        name = simpledialog.askstring("Person", "Name for this person's folder:",
                                      parent=frame.winfo_toplevel())
        if not name or not name.strip():
            return
        name = name.strip()
        paths = filedialog.askopenfilenames(
            title=f"Clear photos of {name} (3-10 is plenty)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if not paths:
            return
        person_var.set(f"{name}: reading {len(paths)} photo(s)…")
        _log(f"{name}: reading {len(paths)} reference photo(s)…")

        def work() -> None:
            try:
                refs, rep = harvest.build_reference(list(paths))
            except Exception as e:
                q.put({"phase": "person", "message": f"{name}: FAILED {e}",
                       "person_ok": False})
                return
            if len(refs) == 0:
                q.put({"phase": "person", "person_ok": False, "message":
                       f"{name}: no usable face in those photos "
                       f"({rep.get('no_face', 0)} had no detectable face) — "
                       f"try clearer, front-facing pictures"})
                return
            state["name"], state["refs"] = name, refs
            q.put({"phase": "person", "person_ok": True, "person": name,
                   "message": f"{name}: {len(refs)} reference face(s) from "
                              f"{rep.get('candidates', 0)} photo(s)"
                              + (f", dropped {rep['outliers']} that look like a "
                                 f"different person" if rep.get("outliers") else "")})

        threading.Thread(target=work, daemon=True).start()

    person_btn.configure(command=choose_person)

    # ── step 2 handlers ─────────────────────────────────────────────────────
    def add_videos() -> None:
        paths = filedialog.askopenfilenames(
            title="Videos to search for this person",
            filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov *.dav"), ("All", "*.*")])
        for p in paths:
            if p not in state["videos"]:
                state["videos"].append(p)
                vid_list.insert(tk.END, f" {os.path.basename(p)}")
        status_var.set(f"{len(state['videos'])} video(s) queued.")

    def clear_videos() -> None:
        state["videos"].clear()
        vid_list.delete(0, tk.END)
        status_var.set("Videos cleared.")

    add_btn.configure(command=add_videos)
    clear_btn.configure(command=clear_videos)

    # ── enrol what was collected ────────────────────────────────────────────
    def _enrol_best() -> None:
        """Add this person's collected faces to the recognition gallery."""
        from core import enrollment
        from database import db_manager

        if not state["name"]:
            messagebox.showinfo("No person", "Choose a person's photos first.")
            return
        face_dir = os.path.join(harvest.OUT_ROOT, harvest._slug(state["name"]), "face")
        if not os.path.isdir(face_dir):
            messagebox.showinfo("Nothing collected",
                                "No images collected for this person yet.")
            return
        imgs = [os.path.join(face_dir, f) for f in sorted(os.listdir(face_dir))
                if f.lower().endswith((".jpg", ".png"))]
        if not imgs:
            messagebox.showinfo("Nothing collected", "That folder has no images.")
            return

        emps = db_manager.get_employees()
        choices = "\n".join(f"  {e['employee_id']} — {e.get('name','')}"
                            for e in emps[:40])
        eid = simpledialog.askstring(
            "Add to gallery",
            f"Which employee ID do these {len(imgs)} images belong to?\n\n"
            f"{choices}", parent=frame.winfo_toplevel())
        if not eid or not eid.strip():
            return
        eid = eid.strip()
        if not any(str(e["employee_id"]) == eid for e in emps):
            messagebox.showerror("Unknown ID",
                                 f"{eid} is not a registered employee. "
                                 f"Register them first.")
            return

        try:
            target = int(target_var.get().strip())
        except ValueError:
            target = settings.EMB_PER_EMPLOYEE_MAX
        target = min(target, settings.EMB_PER_EMPLOYEE_MAX)

        def work() -> None:
            try:
                rep = enrollment.enroll_best_from_images(eid, imgs, target=target,
                                                        replace=False)
                q.put({"phase": "enrolled", "message":
                       f"{eid}: picked {rep.get('selected', 0)} of "
                       f"{rep.get('usable', 0)}, added {rep.get('added', 0)} "
                       f"-> {rep.get('total_templates', '?')} templates "
                       f"(restart the app to reload the gallery)"})
            except Exception as e:
                log.exception("enrol failed")
                q.put({"phase": "enrolled", "message": f"{eid}: FAILED {e}"})

        threading.Thread(target=work, daemon=True).start()

    # ── status drain ────────────────────────────────────────────────────────
    def _drain() -> None:
        last = None
        try:
            while True:
                info = q.get_nowait()
                if info.get("message"):
                    _log(info["message"])
                if info.get("phase") == "person":
                    person_var.set(info["message"] if not info.get("person_ok")
                                   else f"{info['person']} — ready")
                last = info
        except queue.Empty:
            pass
        if last:
            tg = last.get("targets") or {}
            if tg:
                counts.delete(0, tk.END)
                for name in sorted(tg):
                    f, b = tg[name]
                    counts.insert(tk.END, f"{name[:18]:<20}{f:>6}{b:>6}")
            phase = last.get("phase", "")
            if phase and phase not in ("person",):
                status_var.set(f"{phase} — {last.get('message', '')}"[:160])
            if phase in ("done", "error"):
                start_btn.configure(state="normal"); stop_btn.configure(state="disabled")
                if phase == "error":
                    messagebox.showerror("Collection failed", last.get("message", ""))
        job = state["job"]
        if job is not None and not job.is_running() and start_btn["state"] == "disabled":
            start_btn.configure(state="normal"); stop_btn.configure(state="disabled")
        frame.after(250, _drain)

    # ── step 3 handlers ─────────────────────────────────────────────────────
    def start() -> None:
        if state["job"] and state["job"].is_running():
            return
        if state["refs"] is None or not state["name"]:
            messagebox.showinfo("Choose a person first",
                                "Step 1: pick a name and a few clear photos of "
                                "the person you want images of.")
            return
        if not state["videos"]:
            messagebox.showinfo("No videos",
                                "Step 2: add the videos to search.")
            return
        try:
            target = int(target_var.get().strip())
            if target < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Bad input", "Images to collect must be a number.")
            return

        job = harvest.HarvestJob(videos=list(state["videos"]),
                                 target_per_employee=target, mode=mode_var.get(),
                                 collect_unknown=False, only_targets=True,
                                 targets={state["name"]: state["refs"]})
        state["job"] = job
        counts.delete(0, tk.END)
        log_box.delete(0, tk.END)
        _log(f"searching {len(state['videos'])} video(s) for {state['name']}, "
             f"up to {target} images")
        status_var.set(f"Looking for {state['name']}…")
        start_btn.configure(state="disabled"); stop_btn.configure(state="normal")
        job.start(on_status=lambda info: q.put(info))

    def stop() -> None:
        if state["job"]:
            status_var.set("Stopping…")
            state["job"].stop()

    start_btn.configure(command=start)
    stop_btn.configure(command=stop)
    frame.after(250, _drain)
    return frame