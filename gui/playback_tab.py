"""
gui/playback_tab.py
Mark attendance for a PAST day from the NVR's own recordings.

Pick a date from the list the NVR reports (the same days the web UI's calendar
marks blue), and every camera's footage for that day is downloaded and run
through the ordinary batch pipeline — the same engine, speed and accuracy as
dropping downloaded files on the Process Videos tab.

Downloading rather than streaming is deliberate: see core/nvr_day.py. Segments
are fetched, processed and deleted in waves, so the day's ~126 GB never lands
on disk at once.
"""
from __future__ import annotations
import calendar
import datetime
import logging
import os
import queue
import shutil
import tkinter as tk
from tkinter import ttk, messagebox

from config import settings
from core import live_camera, nvr, nvr_day

log = logging.getLogger("playback_tab")

CAMS = [1, 2, 3, 4, 5, 6, 7]


def build_playback_tab(parent, tts=None):
    frame = tk.Frame(parent, bg="#0f1626")
    q: "queue.Queue" = queue.Queue()
    state = {"job": None, "names": None, "cams": {}}

    # ── top bar ─────────────────────────────────────────────────────────────
    bar = tk.Frame(frame, bg="#0f1626"); bar.pack(fill="x", padx=10, pady=(8, 2))
    tk.Label(bar, text="Past Day — from NVR recordings", bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 13, "bold")).pack(side="left")

    date_var = tk.StringVar(value="")
    tk.Label(bar, textvariable=date_var, bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 11, "bold")).pack(side="left", padx=(18, 2))

    tk.Label(bar, text="From", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(12, 2))
    from_var = tk.StringVar(value="00:00:00")
    tk.Entry(bar, textvariable=from_var, width=9).pack(side="left")
    tk.Label(bar, text="To", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(8, 2))
    to_var = tk.StringVar(value="23:59:59")
    tk.Entry(bar, textvariable=to_var, width=9).pack(side="left")

    tk.Label(bar, text="Mode", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(12, 2))
    mode_var = tk.StringVar(value=settings.DEFAULT_PROCESSING_MODE)
    ttk.Combobox(bar, textvariable=mode_var, state="readonly", width=10,
                 values=list(settings.PROCESSING_MODES.keys())).pack(side="left")

    bar2 = tk.Frame(frame, bg="#0f1626"); bar2.pack(fill="x", padx=10)
    start_btn = tk.Button(bar2, text="▶ Mark this day", bg="#1f7a3f", fg="white",
                          font=("Verdana", 10, "bold"), width=15)
    start_btn.pack(side="left", padx=(0, 3), pady=4)
    stop_btn = tk.Button(bar2, text="■ Stop", bg="#7a2f2f", fg="white",
                         font=("Verdana", 10, "bold"), width=9, state="disabled")
    stop_btn.pack(side="left", padx=3)

    cache_var = tk.StringVar(value="")
    tk.Label(bar2, textvariable=cache_var, bg="#0f1626", fg="#6f86a8").pack(side="left", padx=12)

    def _cache_note() -> None:
        d = settings.NVR_CACHE_DIR
        try:
            free = shutil.disk_usage(os.path.splitdrive(d)[0] + "\\").free / (1 << 30)
            cache_var.set(f"cache: {d}  ({free:.0f} GB free, "
                          f"{settings.NVR_WAVE_BUDGET_GB} GB at a time)")
        except Exception:
            cache_var.set(f"cache: {d}")

    _cache_note()

    # ── roster strip at the bottom ──────────────────────────────────────────
    roster = tk.Frame(frame, bg="#0f1626")
    roster.pack(side="bottom", fill="x", padx=10, pady=(2, 8))
    present_hdr = tk.StringVar(value="Marked present — 0")
    hdr = tk.Frame(roster, bg="#0f1626"); hdr.pack(fill="x")
    tk.Label(hdr, textvariable=present_hdr, bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 11, "bold")).pack(side="left")
    status_var = tk.StringVar(value="Idle.")
    tk.Label(hdr, textvariable=status_var, bg="#0f1626", fg="#9fb7d8").pack(side="right")
    tk.Label(roster, text=f"{'ID':<6}{'NAME':<28}", bg="#0f1626", fg="#6f86a8",
             font=("Consolas", 9), anchor="w").pack(fill="x", pady=(4, 0))
    present_list = tk.Listbox(roster, height=6, bg="#0a1020", fg="#cfe0ff",
                              highlightthickness=0, borderwidth=0, activestyle="none",
                              font=("Consolas", 10))
    present_list.pack(fill="x", pady=(0, 2))

    # ── body: date picker (left) + progress (right) ─────────────────────────
    body = tk.Frame(frame, bg="#0f1626"); body.pack(fill="both", expand=True, padx=10, pady=4)

    picker = tk.Frame(body, bg="#0f1626"); picker.pack(side="left", fill="y", padx=(0, 10))
    month_bar = tk.Frame(picker, bg="#0f1626"); month_bar.pack(fill="x")
    today = datetime.date.today()
    shown = {"year": today.year, "month": today.month}
    month_var = tk.StringVar(value="")

    prev_btn = tk.Button(month_bar, text="◀", width=2, bg="#1b2740", fg="#cfe0ff",
                         relief="flat")
    prev_btn.pack(side="left")
    tk.Label(month_bar, textvariable=month_var, bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 10, "bold"), width=12).pack(side="left")
    next_btn = tk.Button(month_bar, text="▶", width=2, bg="#1b2740", fg="#cfe0ff",
                         relief="flat")
    next_btn.pack(side="left")

    tk.Label(picker, text="Dates the NVR has recordings for", bg="#0f1626",
             fg="#9fb7d8", font=("Verdana", 8)).pack(anchor="w", pady=(6, 2))
    dates_list = tk.Listbox(picker, height=16, width=22, bg="#0a1020", fg="#cfe0ff",
                            highlightthickness=0, borderwidth=0, activestyle="none",
                            font=("Consolas", 10), exportselection=False)
    dates_list.pack(fill="y", expand=True)
    picker_msg = tk.StringVar(value="loading…")
    tk.Label(picker, textvariable=picker_msg, bg="#0f1626", fg="#6f86a8",
             wraplength=170, justify="left", font=("Verdana", 8)).pack(anchor="w", pady=4)

    right = tk.Frame(body, bg="#0f1626"); right.pack(side="left", fill="both", expand=True)

    prog_var = tk.StringVar(value="Pick a date, then “Mark this day”.")
    tk.Label(right, textvariable=prog_var, bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 10, "bold"), anchor="w").pack(fill="x")
    bar_pct = ttk.Progressbar(right, mode="determinate", maximum=100)
    bar_pct.pack(fill="x", pady=6)

    cams_frame = tk.Frame(right, bg="#0f1626"); cams_frame.pack(fill="x", pady=(2, 8))
    for cam in CAMS:
        v = tk.StringVar(value=f"Camera {cam} — waiting")
        tk.Label(cams_frame, textvariable=v, bg="#0f1626", fg="#9fb7d8",
                 font=("Consolas", 9), anchor="w").pack(fill="x")
        state["cams"][cam] = v

    tk.Label(right, text="Activity", bg="#0f1626", fg="#9fb7d8",
             font=("Verdana", 9, "bold"), anchor="w").pack(fill="x")
    log_box = tk.Listbox(right, height=10, bg="#0a1020", fg="#8fa8c8",
                         highlightthickness=0, borderwidth=0, activestyle="none",
                         font=("Consolas", 9))
    log_box.pack(fill="both", expand=True, pady=(2, 0))

    def _log(msg: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_box.insert(tk.END, f" {stamp}  {msg}")
        if log_box.size() > 300:
            log_box.delete(0)
        log_box.see(tk.END)

    # ── date picker behaviour ───────────────────────────────────────────────
    def _load_month() -> None:
        y, m = shown["year"], shown["month"]
        month_var.set(f"{calendar.month_abbr[m]} {y}")
        dates_list.delete(0, tk.END)
        picker_msg.set("querying NVR…")
        frame.update_idletasks()
        try:
            days = nvr.recorded_days(live_camera.load_config(), y, m)
        except Exception as e:
            picker_msg.set(f"NVR query failed:\n{e}")
            return
        if not days:
            picker_msg.set("no recordings in this month")
            return
        for d in days:
            dates_list.insert(tk.END, f" {datetime.date.fromisoformat(d):%a  %d %b}")
        dates_list.dates = days
        picker_msg.set(f"{len(days)} day(s) available — click one, then "
                       f"“Mark this day”.")

    def _shift_month(delta: int) -> None:
        m = shown["month"] + delta
        shown["year"] += (m - 1) // 12
        shown["month"] = (m - 1) % 12 + 1
        _load_month()

    prev_btn.configure(command=lambda: _shift_month(-1))
    next_btn.configure(command=lambda: _shift_month(1))

    def _pick_date(_evt=None) -> None:
        sel = dates_list.curselection()
        days = getattr(dates_list, "dates", [])
        if sel and sel[0] < len(days):
            date_var.set(days[sel[0]])
            prog_var.set(f"{days[sel[0]]} selected — press “Mark this day”.")

    dates_list.bind("<<ListboxSelect>>", _pick_date)

    # ── status drain (Tk thread) ────────────────────────────────────────────
    def _emp_names() -> dict:
        if state["names"] is None:
            try:
                from database import db_manager
                state["names"] = {str(e["employee_id"]): (e.get("name") or "")
                                  for e in db_manager.get_employees()}
            except Exception:
                return {}
        return state["names"]

    def _apply(info: dict) -> None:
        phase = info.get("phase", "")
        msg = info.get("message", "")
        done, total = info.get("done", 0), info.get("total", 0)
        mb = info.get("downloaded_mb", 0.0)

        if total:
            bar_pct["value"] = min(100.0, 100.0 * done / total)
            prog_var.set(f"{phase} — segment {done}/{total} · "
                         f"{mb/1024:.1f} GB fetched")
        elif phase:
            prog_var.set(phase + (f" — {msg}" if msg else ""))

        cam = info.get("camera")
        if cam in state["cams"] and msg:
            state["cams"][cam].set(f"Camera {cam} — {msg}")
        if msg:
            _log(msg)

        present = info.get("present_ids") or []
        if present:
            names = _emp_names()
            present_hdr.set(f"Marked present on {date_var.get()} — {len(present)}")
            present_list.delete(0, tk.END)
            for eid in present:
                present_list.insert(tk.END,
                                    f"{str(eid):<6}{names.get(str(eid), '')[:26]:<28}")

        if phase in ("done", "error"):
            start_btn.configure(state="normal"); stop_btn.configure(state="disabled")
            status_var.set(msg or "Finished.")
            if phase == "error":
                messagebox.showerror("Day replay failed", msg)

    def _drain() -> None:
        latest = None
        try:
            while True:
                info = q.get_nowait()
                _apply(info)                 # every tick carries its own message
                latest = info
        except queue.Empty:
            pass
        job = state["job"]
        if job is not None and not job.is_running() and start_btn["state"] == "disabled":
            start_btn.configure(state="normal"); stop_btn.configure(state="disabled")
            status_var.set(job.error or f"Finished — {len(job.present)} present.")
        frame.after(200, _drain)

    # ── start / stop ────────────────────────────────────────────────────────
    def start():
        if state["job"] and state["job"].is_running():
            return
        day = date_var.get().strip()
        if not day:
            messagebox.showinfo("Pick a date",
                                "Select one of the dates on the left — those are "
                                "the days the NVR actually holds recordings for.")
            return
        t_from, t_to = from_var.get().strip(), to_var.get().strip()
        try:
            datetime.datetime.strptime(t_from, "%H:%M:%S")
            datetime.datetime.strptime(t_to, "%H:%M:%S")
        except ValueError:
            messagebox.showerror("Bad input", "Times must be HH:MM:SS.")
            return
        if t_to <= t_from:
            messagebox.showerror("Bad input", "‘To’ must be after ‘From’.")
            return

        job = nvr_day.NvrDayJob(day=day, cams=CAMS, mode=mode_var.get(),
                                start_clock=t_from, end_clock=t_to)
        state["job"] = job
        state["names"] = None
        present_list.delete(0, tk.END)
        log_box.delete(0, tk.END)
        for cam in CAMS:
            state["cams"][cam].set(f"Camera {cam} — waiting")
        bar_pct["value"] = 0
        status_var.set(f"Replaying {day} {t_from}–{t_to}…")
        _log(f"start {day} {t_from}-{t_to}, cameras {CAMS}")
        start_btn.configure(state="disabled"); stop_btn.configure(state="normal")
        job.start(on_status=lambda info: q.put(info))

    def stop():
        job = state["job"]
        if job:
            status_var.set("Stopping after the current download…")
            job.stop()

    def _init_nvr() -> None:
        try:
            names = nvr.channel_names(live_camera.load_config())
            for cam in CAMS:
                if names.get(cam):
                    state["cams"][cam].set(f"Camera {cam} — {names[cam]} — waiting")
        except Exception as e:
            log.warning("channel names unavailable: %s", e)
        _load_month()

    start_btn.configure(command=start)
    stop_btn.configure(command=stop)
    frame.after(200, _drain)
    frame.after(300, _init_nvr)
    return frame
