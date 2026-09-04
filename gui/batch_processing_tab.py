"""
gui/batch_processing_tab.py
Batch video processing screen for the v2 pipeline.

Lets the user add multiple videos, set per-video metadata (date, start time,
camera id/location/type, optional entry/exit line), pick a processing mode, and
run everything in a background thread. Progress (current video, timestamp, %,
FPS, tracks, confirmed, unknown, ETA, preview frame) is streamed to the UI via a
thread-safe queue polled with Tk's after(), so the window stays responsive.

On completion it fuses evidence across all videos into one daily record per
employee, writes them to the DB, and exports the Excel workbook.
"""
from __future__ import annotations
import os
import queue
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import date as _date

import numpy as np

from config import settings

log = logging.getLogger("batch_tab")

try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None


def _ui_state_path():
    return os.path.join(settings.DATA_DIR, "ui_state.json")


def _load_ui_state() -> dict:
    import json
    try:
        with open(_ui_state_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_ui_state(d: dict) -> None:
    import json
    try:
        with open(_ui_state_path(), "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass


def build_batch_processing_tab(parent, tts=None):
    frame = tk.Frame(parent, bg="#0f1626")
    state = {"videos": [], "worker": None, "cancel": False, "queue": queue.Queue(),
             "preview_ref": None}

    # ── left: video list + controls ────────────────────────────────────────
    left = tk.Frame(frame, bg="#0f1626"); left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
    tk.Label(left, text="Videos for the selected date", bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 12, "bold")).pack(anchor="w")

    cols = ("file", "date", "start", "cam_id", "location", "type", "line")
    tree = ttk.Treeview(left, columns=cols, show="headings", height=8)
    for c, w in zip(cols, (150, 90, 70, 70, 100, 90, 50)):
        tree.heading(c, text=c); tree.column(c, width=w)
    tree.pack(fill="both", expand=True, pady=6)

    form = tk.Frame(left, bg="#0f1626"); form.pack(fill="x")
    fields = {}
    def _row(lbl, default, r):
        tk.Label(form, text=lbl, bg="#0f1626", fg="#9fb7d8").grid(row=r, column=0, sticky="w", pady=2)
        e = tk.Entry(form, width=22); e.insert(0, default); e.grid(row=r, column=1, pady=2, padx=4)
        fields[lbl] = e
    _ui = _load_ui_state()
    _row("Date (YYYY-MM-DD)", _date.today().isoformat(), 0)
    _row("Start time (HH:MM:SS)", _ui.get("start", "09:00:00"), 1)
    _row("Camera ID", _ui.get("cam_id", "CAM1"), 2)
    _row("Camera location", _ui.get("location", "Main Entrance"), 3)
    tk.Label(form, text="Camera type", bg="#0f1626", fg="#9fb7d8").grid(row=4, column=0, sticky="w")
    cam_type = ttk.Combobox(form, values=settings.CAMERA_TYPES, width=20, state="readonly")
    cam_type.set(_ui.get("cam_type", settings.CAMERA_TYPES[0])); cam_type.grid(row=4, column=1, pady=2, padx=4)
    _row("Entry/exit line (0-1, blank=none)", "", 5)

    def add_videos():
        paths = filedialog.askopenfilenames(
            title="Select video(s)",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All", "*.*")])
        line_raw = fields["Entry/exit line (0-1, blank=none)"].get().strip()
        line_val = float(line_raw) if line_raw else ""
        for p in paths:
            meta = (os.path.basename(p), fields["Date (YYYY-MM-DD)"].get(),
                    fields["Start time (HH:MM:SS)"].get(), fields["Camera ID"].get(),
                    fields["Camera location"].get(), cam_type.get(), line_val)
            tree.insert("", "end", values=meta)
            state["videos"].append({"path": p, "meta": meta})

    def clear_videos():
        for i in tree.get_children():
            tree.delete(i)
        state["videos"].clear()

    btns = tk.Frame(left, bg="#0f1626"); btns.pack(fill="x", pady=6)
    tk.Button(btns, text="➕ Add Videos", command=add_videos).pack(side="left", padx=3)
    tk.Button(btns, text="🗑 Clear", command=clear_videos).pack(side="left", padx=3)
    tk.Label(btns, text="Mode:", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(12, 2))
    mode_box = ttk.Combobox(btns, values=list(settings.PROCESSING_MODES.keys()),
                            width=10, state="readonly")
    mode_box.set(_ui.get("mode", settings.DEFAULT_PROCESSING_MODE)); mode_box.pack(side="left")
    tk.Label(btns, text="Workers:", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(12, 2))
    workers_var = tk.StringVar(value=str(_ui.get("workers", getattr(settings, "PARALLEL_VIDEOS", 2))))
    tk.Spinbox(btns, from_=1, to=12, width=3, textvariable=workers_var).pack(side="left")

    # ── right: progress + preview ──────────────────────────────────────────
    right = tk.Frame(frame, bg="#0f1626"); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)
    tk.Label(right, text="Processing", bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 12, "bold")).pack(anchor="w")
    status_var = tk.StringVar(value="Idle.")
    tk.Label(right, textvariable=status_var, bg="#0f1626", fg="#cfe0ff",
             wraplength=420, justify="left").pack(anchor="w", pady=4)
    bar = ttk.Progressbar(right, length=420, mode="determinate"); bar.pack(pady=4)
    stats_var = tk.StringVar(value="")
    tk.Label(right, textvariable=stats_var, bg="#0f1626", fg="#9fb7d8",
             justify="left").pack(anchor="w")
    # preview (left) + live "marked present" roster (right)
    media = tk.Frame(right, bg="#0f1626"); media.pack(fill="x", pady=8)
    preview_lbl = tk.Label(media, bg="#0a1020"); preview_lbl.pack(side="left")
    roster = tk.Frame(media, bg="#0f1626")
    roster.pack(side="left", fill="both", expand=True, padx=(10, 0))
    roster_hdr = tk.StringVar(value="Marked present (0)")
    tk.Label(roster, textvariable=roster_hdr, bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 11, "bold")).pack(anchor="w")
    present_list = tk.Listbox(roster, height=13, width=24, bg="#0a1020",
                              fg="#cfe0ff", highlightthickness=0, borderwidth=0,
                              activestyle="none", font=("Verdana", 10))
    present_list.pack(fill="both", expand=True, pady=4)

    def _emp_names() -> dict:
        try:
            from database import db_manager
            out = {}
            for e in db_manager.get_employees():
                eid = e["employee_id"] if isinstance(e, dict) else getattr(e, "employee_id", None)
                nm = (e.get("name") if isinstance(e, dict) else getattr(e, "name", "")) or ""
                if eid is not None:
                    out[str(eid)] = nm
            return out
        except Exception:
            return {}

    # ── worker ─────────────────────────────────────────────────────────────
    def _progress_cb(pr):
        state["queue"].put(("progress", pr))

    def _run(jobs, mode):
        try:
            from core.pipeline import process_batch
            from core import attendance_engine as ae
            from database import db_manager, db_setup
            db_setup.init_db()
            results = process_batch(jobs, mode=mode, progress_cb=_progress_cb,
                                    cancel=lambda: state["cancel"],
                                    workers=int(workers_var.get() or 1))
            # fuse across videos
            all_ev, all_gates = [], []
            for r in results.values():
                all_ev.extend(r.evidences); all_gates.extend(r.gate_events)
            the_date = jobs[0].date if jobs else _date.today().isoformat()
            records = ae.fuse_day(the_date, all_ev, all_gates)
            for rec in records.values():
                db_manager.save_daily_record(rec)
            state["queue"].put(("done", (the_date, records)))
        except Exception as e:
            log.exception("batch processing failed")
            state["queue"].put(("error", str(e)))

    def start():
        _save_ui_state({"start": fields["Start time (HH:MM:SS)"].get(),
                        "cam_id": fields["Camera ID"].get(),
                        "location": fields["Camera location"].get(),
                        "cam_type": cam_type.get(), "mode": mode_box.get(),
                        "workers": int(workers_var.get() or 1)})
        if not state["videos"]:
            messagebox.showwarning("No videos", "Add at least one video first."); return
        if state["worker"] and state["worker"].is_alive():
            return
        remote = [v for v in state["videos"] if "tsclient" in v["path"].lower()]
        if remote:
            if not messagebox.askyesno(
                    "Remote Desktop files — 10-15x slower",
                    f"{len(remote)} of {len(state['videos'])} selected videos are on the "
                    "Remote Desktop drive (\\\\tsclient), which reads at under 1 MB/s.\n\n"
                    "Processing from there runs SLOWER than realtime (a 1 h video can "
                    "take hours). The same video on a local disk runs ~10x realtime.\n\n"
                    "Copy the files to this PC (e.g. E:\\videos) or a normal network "
                    "share first, then add them from there.\n\n"
                    "Process from the slow Remote Desktop drive anyway?"):
                return
        from models.registry import missing_required, status_report
        if missing_required():
            messagebox.showerror("Models missing",
                                 "Model weights are not downloaded yet.\n\n"
                                 + status_report()
                                 + "\n\nRun:  python scripts/download_models.py")
            return
        from core.pipeline import VideoJob
        jobs = []
        for v in state["videos"]:
            f, d, st, cid, loc, typ, line = v["meta"]
            jobs.append(VideoJob(path=v["path"], date=d, start_time=st, camera_id=cid,
                                 camera_location=loc, camera_type=typ,
                                 line_fraction=(line if line != "" else None)))
        state["cancel"] = False
        state["emp_names"] = _emp_names()
        state["present_shown"] = None
        present_list.delete(0, "end")
        roster_hdr.set("Marked present (0)")
        status_var.set("Loading models…")
        state["worker"] = threading.Thread(target=_run, args=(jobs, mode_box.get()), daemon=True)
        state["worker"].start()

    def cancel():
        state["cancel"] = True
        status_var.set("Cancelling…")

    run_btns = tk.Frame(right, bg="#0f1626"); run_btns.pack(pady=6)
    tk.Button(run_btns, text="▶ Start Processing", command=start,
              bg="#2f6fd0", fg="white").pack(side="left", padx=4)
    tk.Button(run_btns, text="■ Cancel", command=cancel).pack(side="left", padx=4)

    # ── UI pump (main thread) ──────────────────────────────────────────────
    def _pump():
        try:
            while True:
                kind, payload = state["queue"].get_nowait()
                if kind == "progress":
                    pr = payload
                    status_var.set(f"{pr.video_name}  •  t={pr.frame_ts:.1f}s  •  {pr.percent:.1f}%")
                    bar["value"] = pr.percent
                    stats_var.set(f"PRESENT SO FAR {getattr(pr, 'n_present', 0)} | "
                                  f"FPS {pr.fps:.1f} | people in frame {pr.n_tracks} | "
                                  f"confirmed on screen {pr.n_confirmed} | unidentified now {pr.n_unknown} | "
                                  f"track segments {pr.n_segments} | ETA {pr.eta_s:.0f}s")
                    ids = getattr(pr, "present_ids", ())
                    if ids != state.get("present_shown"):
                        state["present_shown"] = ids
                        names = state.get("emp_names") or {}
                        present_list.delete(0, "end")
                        for i in ids:
                            label = f"  {i}   {names.get(str(i), '')}".rstrip()
                            present_list.insert("end", label)
                        roster_hdr.set(f"Marked present ({len(ids)})")
                    if ImageTk is not None and pr.preview is not None:
                        img = pr.preview[:, :, ::-1]
                        pim = Image.fromarray(img)
                        pim.thumbnail((400, 300))
                        ph = ImageTk.PhotoImage(pim)
                        state["preview_ref"] = ph
                        preview_lbl.configure(image=ph)
                elif kind == "done":
                    the_date, records = payload
                    present = sum(1 for r in records.values() if r.attendance_status == "PRESENT")
                    review = sum(1 for r in records.values() if r.attendance_status == "REVIEW")
                    status_var.set(f"Done for {the_date}: {present} present, {review} to review. "
                                   f"Saved to database.")
                    bar["value"] = 100
                    _offer_export(the_date)
                elif kind == "error":
                    status_var.set(f"Error: {payload}")
                    messagebox.showerror("Processing error", payload)
        except queue.Empty:
            pass
        frame.after(120, _pump)

    def _offer_export(the_date):
        if messagebox.askyesno("Export", f"Export Excel report for {the_date}?"):
            try:
                from database import db_manager
                from reports import excel_report
                path = excel_report.export(the_date, db_manager.get_daily_attendance(the_date),
                                           db_manager.get_events(the_date))
                messagebox.showinfo("Exported", f"Saved:\n{path}")
            except Exception as e:
                messagebox.showerror("Export failed", str(e))

    frame.after(200, _pump)
    return frame
