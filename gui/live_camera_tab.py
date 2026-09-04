"""
gui/live_camera_tab.py
Continuous LIVE attendance from an RTSP camera (Hikvision).

Start opens the live stream (URL built from camera_config.json), runs the same
recognition pipeline used for files, and marks employees PRESENT the instant
they are CONFIRMED — updating a live preview and a "marked present" roster.
Stop ends the stream and finalizes stored confidences.

The recognition runs in a background thread inside core.live_camera; its
progress ticks are pushed to a thread-safe queue that Tk drains via after(),
so the window stays responsive.
"""
from __future__ import annotations
import logging
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from config import settings
from core import live_camera

log = logging.getLogger("live_tab")

try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None


def build_live_camera_tab(parent, tts=None):
    frame = tk.Frame(parent, bg="#0f1626")
    q: "queue.Queue" = queue.Queue()
    state = {"live": None, "preview_ref": None, "names": None,
             "present_shown": None}

    # ── left: controls ──────────────────────────────────────────────────────
    left = tk.Frame(frame, bg="#0f1626"); left.pack(side="left", fill="y", padx=10, pady=10)
    tk.Label(left, text="Live Camera Attendance", bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 13, "bold")).pack(anchor="w")

    try:
        cfg = live_camera.load_config()
        cam_desc = f"{cfg.get('ip')}  (port {cfg.get('port', 554)})"
    except Exception as e:
        cfg = {}
        cam_desc = f"camera_config.json error: {e}"
    tk.Label(left, text=f"Camera: {cam_desc}", bg="#0f1626", fg="#9fb7d8").pack(anchor="w", pady=(2, 8))

    form = tk.Frame(left, bg="#0f1626"); form.pack(fill="x")
    fields = {}

    def _row(lbl, default, r):
        tk.Label(form, text=lbl, bg="#0f1626", fg="#9fb7d8").grid(row=r, column=0, sticky="w", pady=2)
        e = tk.Entry(form, width=20); e.insert(0, default); e.grid(row=r, column=1, pady=2, padx=4)
        fields[lbl] = e
        return e

    _row("Camera ID", "LIVE-01", 0)
    _row("Camera location", "office", 1)

    tk.Label(form, text="Stream", bg="#0f1626", fg="#9fb7d8").grid(row=2, column=0, sticky="w", pady=2)
    stream_var = tk.StringVar(value="101 (main / full HD)")
    ttk.Combobox(form, textvariable=stream_var, state="readonly", width=17,
                 values=["101 (main / full HD)", "102 (sub / lighter)"]).grid(row=2, column=1, pady=2, padx=4)

    tk.Label(form, text="Mode", bg="#0f1626", fg="#9fb7d8").grid(row=3, column=0, sticky="w", pady=2)
    mode_var = tk.StringVar(value=settings.DEFAULT_PROCESSING_MODE)
    ttk.Combobox(form, textvariable=mode_var, state="readonly", width=17,
                 values=list(settings.PROCESSING_MODES.keys())).grid(row=3, column=1, pady=2, padx=4)

    btns = tk.Frame(left, bg="#0f1626"); btns.pack(fill="x", pady=10)
    start_btn = tk.Button(btns, text="▶ Start Live", bg="#1f7a3f", fg="white",
                          font=("Verdana", 10, "bold"), width=12)
    start_btn.pack(side="left", padx=3)
    stop_btn = tk.Button(btns, text="■ Stop", bg="#7a2f2f", fg="white",
                         font=("Verdana", 10, "bold"), width=8, state="disabled")
    stop_btn.pack(side="left", padx=3)

    status_var = tk.StringVar(value="Idle.")
    tk.Label(left, textvariable=status_var, bg="#0f1626", fg="#cfe0ff",
             wraplength=300, justify="left").pack(anchor="w", pady=4)
    tk.Label(left, text="On CPU the view lags real time; attendance is still\n"
                        "correct (anyone seen once = present). Use sub-stream\n"
                        "(102) to reduce lag.",
             bg="#0f1626", fg="#6f86a8", justify="left").pack(anchor="w", pady=(6, 0))

    # ── right: preview + present roster ─────────────────────────────────────
    right = tk.Frame(frame, bg="#0f1626"); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)
    tk.Label(right, text="Live feed", bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 12, "bold")).pack(anchor="w")
    preview_lbl = tk.Label(right, bg="#0a1020"); preview_lbl.pack(anchor="w", pady=6)

    roster_hdr = tk.StringVar(value="Marked present (0)")
    tk.Label(right, textvariable=roster_hdr, bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 11, "bold")).pack(anchor="w")
    present_list = tk.Listbox(right, height=12, width=40, bg="#0a1020",
                              fg="#cfe0ff", highlightthickness=0, borderwidth=0,
                              activestyle="none", font=("Verdana", 10))
    present_list.pack(fill="both", expand=True, pady=4)

    def _emp_names() -> dict:
        # cached: the roster changes only via Registration, not per frame —
        # querying SQLite on every preview repaint stalled the Tk thread.
        if state["names"] is None:
            try:
                from database import db_manager
                state["names"] = {str(e["employee_id"]): (e.get("name") or "")
                                  for e in db_manager.get_employees()}
            except Exception:
                return {}
        return state["names"]

    # ── UI updates (Tk thread) ──────────────────────────────────────────────
    def _render(info: dict) -> None:
        present = info.get("present_ids", [])
        if present != state["present_shown"]:      # only rebuild on change
            names = _emp_names()
            roster_hdr.set(f"Marked present ({len(present)})")
            present_list.delete(0, tk.END)
            for eid in present:
                nm = names.get(str(eid), "")
                present_list.insert(tk.END, f"  {eid}  {nm}".rstrip())
            state["present_shown"] = list(present)
        status_var.set(f"LIVE — {int(info.get('elapsed_s', 0))}s · "
                       f"{info.get('n_tracks', 0)} tracked · "
                       f"{info.get('fps', 0):.1f} fps (processing)")
        frame_img = info.get("preview")
        if frame_img is not None and Image is not None:
            try:
                import cv2
                # downscale FIRST with cv2 (INTER_AREA), then convert — resizing
                # a full-HD frame with PIL on the Tk thread was the main stall.
                h, w = frame_img.shape[:2]
                scale = min(560 / w, 380 / h, 1.0)
                if scale < 1.0:
                    frame_img = cv2.resize(frame_img, (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                ph = ImageTk.PhotoImage(Image.fromarray(rgb))
                preview_lbl.configure(image=ph)
                state["preview_ref"] = ph
            except Exception:
                pass

    def _drain() -> None:
        # keep only the newest tick: rendering every backlogged frame did the
        # full resize/PhotoImage work for frames that are thrown away one line
        # later, so a slow repaint (e.g. over RDP) fed back into more work.
        latest = None
        try:
            while True:
                latest = q.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            _render(latest)
        live = state["live"]
        if live is not None and not live.is_running() and start_btn["state"] == "disabled":
            # worker ended (stopped or errored)
            start_btn.configure(state="normal")
            stop_btn.configure(state="disabled")
            if live.error:
                status_var.set(f"Stopped — error: {live.error}")
                messagebox.showerror("Live camera error", live.error)
            else:
                status_var.set(f"Stopped. {len(live.seen_present)} marked present today.")
        frame.after(120, _drain)

    # ── start / stop ────────────────────────────────────────────────────────
    def start():
        if state["live"] and state["live"].is_running():
            return
        channel = "102" if stream_var.get().startswith("102") else "101"
        try:
            live = live_camera.LiveAttendance(
                camera_id=fields["Camera ID"].get().strip() or "LIVE-01",
                location=fields["Camera location"].get().strip() or "office",
                mode=mode_var.get(), channel=channel)
        except Exception as e:
            messagebox.showerror("Cannot start", str(e)); return
        state["live"] = live
        state["names"] = None                 # re-read roster on each start
        state["present_shown"] = None
        status_var.set("Connecting to camera & loading models…")
        start_btn.configure(state="disabled")
        stop_btn.configure(state="normal")
        live.start(on_update=lambda info: q.put(info))

    def stop():
        live = state["live"]
        if live:
            status_var.set("Stopping…")
            live.stop()

    start_btn.configure(command=start)
    stop_btn.configure(command=stop)
    frame.after(120, _drain)
    return frame
