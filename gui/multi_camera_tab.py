"""
gui/multi_camera_tab.py
LIVE attendance across ALL NVR cameras at once.

Shows a grid of camera tiles (one live preview each) and one shared "marked
present" roster fused across every camera. Start All opens every camera stream
and runs recognition on each in its own thread (models are shared, so memory
stays flat); Stop All ends them and finalizes the day's records.

CPU reality: on a machine without a GPU, running many full-HD pipelines at once
is impossible in real time. Sub-stream is the default and each camera is sampled
slowly — but attendance is a SET fused across all cameras, so a person seen on
any camera at any moment counts as present.
"""
from __future__ import annotations
import datetime
import logging
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from config import settings
from core import live_camera

log = logging.getLogger("multicam_tab")

try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None

CAMS = [1, 2, 3, 4, 5, 6, 7]
GRID_COLS = 4
_OFF_TEXT = "preview off\n\ntick “Show previews”\nabove to see video\n\n(recognition is running)"
TILE_W = 300


def build_multi_camera_tab(parent, tts=None):
    frame = tk.Frame(parent, bg="#0f1626")
    q: "queue.Queue" = queue.Queue()
    # marked: employee_id -> (camera label, HH:MM:SS) of the FIRST camera that
    # marked them present — the roster strip below the grid renders this.
    state = {"multi": None, "refs": {}, "names": None, "marked": {}}

    # ── top bar: controls ───────────────────────────────────────────────────
    bar = tk.Frame(frame, bg="#0f1626"); bar.pack(fill="x", padx=10, pady=8)
    tk.Label(bar, text="All Cameras — Live Attendance", bg="#0f1626", fg="#cfe0ff",
             font=("Verdana", 13, "bold")).pack(side="left")

    tk.Label(bar, text="Stream", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(20, 2))
    stream_var = tk.StringVar(value="sub (lighter — recommended)")
    ttk.Combobox(bar, textvariable=stream_var, state="readonly", width=24,
                 values=["sub (lighter — recommended)", "main (full HD — heavy)"]).pack(side="left")

    tk.Label(bar, text="Mode", bg="#0f1626", fg="#9fb7d8").pack(side="left", padx=(12, 2))
    mode_var = tk.StringVar(value=settings.DEFAULT_PROCESSING_MODE)
    ttk.Combobox(bar, textvariable=mode_var, state="readonly", width=10,
                 values=list(settings.PROCESSING_MODES.keys())).pack(side="left")

    # Previews are display-only: recognition and attendance are unaffected by
    # this box. Default OFF because 7 animated tiles over an RDP session is the
    # single biggest source of UI lag on this machine.
    #
    # preview_flag mirrors the Tk variable as a PLAIN dict: the camera threads
    # poll it, and Tcl/Tk is not thread-safe — calling BooleanVar.get() off the
    # Tk thread can wedge the interpreter.
    preview_on = tk.BooleanVar(value=False)
    preview_flag = {"on": False}

    def _toggle_preview() -> None:
        preview_flag["on"] = bool(preview_on.get())
        for t in tiles.values():
            t["img"].configure(text="" if preview_flag["on"] else _OFF_TEXT)

    tk.Checkbutton(bar, text="Show previews (laggy over RDP)",
                   variable=preview_on, command=_toggle_preview,
                   bg="#0f1626", fg="#9fb7d8",
                   selectcolor="#0a1020", activebackground="#0f1626",
                   activeforeground="#cfe0ff", highlightthickness=0,
                   ).pack(side="left", padx=(12, 0))

    tk.Button(bar, text="⚙ Exit line", bg="#1b2740", fg="#cfe0ff", relief="flat",
              command=lambda: _open_door()).pack(side="left", padx=(12, 0))

    def _open_door() -> None:
        from gui.door_dialog import open_door_dialog
        open_door_dialog(frame.winfo_toplevel())

    start_btn = tk.Button(bar, text="▶ Start All", bg="#1f7a3f", fg="white",
                          font=("Verdana", 10, "bold"), width=11)
    start_btn.pack(side="left", padx=(16, 3))
    stop_btn = tk.Button(bar, text="■ Stop All", bg="#7a2f2f", fg="white",
                         font=("Verdana", 10, "bold"), width=10, state="disabled")
    stop_btn.pack(side="left", padx=3)

    present_hdr = tk.StringVar(value="Present today (0)")   # shown on the roster strip below

    # ── exit alert banner (hidden until someone leaves) ─────────────────────
    exit_banner = tk.Label(frame, text="", bg="#7a2f2f", fg="white",
                           font=("Verdana", 13, "bold"), anchor="w", padx=12)
    exit_var = tk.StringVar(value="")

    def _show_entry(ev) -> None:
        """Walking IN marks attendance — logged, but no pop-up.

        Entries happen every morning for everyone; a modal box per arrival
        would be unusable. Exits are the thing worth interrupting for.
        """
        exit_list.insert(0, f" {ev.at}   IN  {ev.name or ev.employee_id}")

    def _show_exit(ev) -> None:
        """Banner + log + pop-up for one exit."""
        exit_var.set(f"🚪  {ev.headline()}   ({ev.camera_id}, matched by {ev.how})")
        exit_banner.configure(textvariable=exit_var)
        exit_banner.pack(side="top", fill="x", before=roster, pady=(0, 2))
        exit_list.insert(0, f" {ev.at}   {ev.name or ev.employee_id}"
                            f"   ({ev.how}, {ev.confidence:.2f})")
        messagebox.showinfo("Exit", ev.headline(), parent=frame.winfo_toplevel())

    # ── roster strip UNDER the camera grid (packed first so it keeps its
    #    height when the grid above expands) ──────────────────────────────────
    roster = tk.Frame(frame, bg="#0f1626")
    roster.pack(side="bottom", fill="x", padx=10, pady=(2, 8))
    hdr = tk.Frame(roster, bg="#0f1626"); hdr.pack(fill="x")
    tk.Label(hdr, textvariable=present_hdr, bg="#0f1626", fg="#8fe0a0",
             font=("Verdana", 11, "bold")).pack(side="left")
    status_var = tk.StringVar(value="Idle.")
    tk.Label(hdr, textvariable=status_var, bg="#0f1626", fg="#9fb7d8").pack(side="right")
    tk.Label(roster, text=f"{'ID':<6}{'NAME':<26}{'CAMERA':<12}{'MARKED AT':<10}",
             bg="#0f1626", fg="#6f86a8", font=("Consolas", 9),
             anchor="w").pack(fill="x", pady=(4, 0))
    lists = tk.Frame(roster, bg="#0f1626"); lists.pack(fill="x")
    present_list = tk.Listbox(lists, height=7, bg="#0a1020", fg="#cfe0ff",
                              highlightthickness=0, borderwidth=0, activestyle="none",
                              font=("Consolas", 10))
    present_list.pack(side="left", fill="both", expand=True, pady=(0, 2))

    exits_col = tk.Frame(lists, bg="#0f1626"); exits_col.pack(side="right", padx=(12, 0))
    tk.Label(exits_col, text="Left the office", bg="#0f1626", fg="#ff9a9a",
             font=("Verdana", 9, "bold"), anchor="w").pack(fill="x")
    exit_list = tk.Listbox(exits_col, height=6, width=34, bg="#1a0f14", fg="#ffc9c9",
                           highlightthickness=0, borderwidth=0, activestyle="none",
                           font=("Consolas", 9))
    exit_list.pack(fill="both", expand=True)

    # ── body: camera grid ───────────────────────────────────────────────────
    body = tk.Frame(frame, bg="#0f1626"); body.pack(fill="both", expand=True, padx=10, pady=4)
    grid = tk.Frame(body, bg="#0f1626"); grid.pack(side="left", fill="both", expand=True)

    # A blank image of the exact tile size: a Label WITHOUT an image measures
    # width/height in text units, so an empty tile would blow the grid apart
    # whenever previews are off. Holding a placeholder keeps geometry in pixels.
    blank = tk.PhotoImage(width=TILE_W, height=int(TILE_W * 9 / 16))
    state["blank"] = blank

    tiles = {}   # cam_index -> {"img": Label, "cap": StringVar}
    for i, cam in enumerate(CAMS):
        r, c = divmod(i, GRID_COLS)
        cell = tk.Frame(grid, bg="#0a1020", bd=1, relief="solid")
        cell.grid(row=r, column=c, padx=4, pady=4, sticky="n")
        cap = tk.StringVar(value=f"Camera {cam} — idle")
        tk.Label(cell, textvariable=cap, bg="#0a1020", fg="#cfe0ff",
                 font=("Verdana", 9, "bold")).pack(anchor="w", padx=4, pady=2)
        # compound="center" draws the text ON the blank image, so the tile keeps
        # its exact pixel size whether or not a frame is showing.
        img = tk.Label(cell, bg="#05080f", image=blank, text=_OFF_TEXT,
                       compound="center", fg="#5a6e8c", font=("Verdana", 8),
                       justify="center")
        img.pack()
        tiles[i] = {"img": img, "cap": cap}

    def _emp_names() -> dict:
        # cached — see live_camera_tab: this ran once per drain tick before.
        if state["names"] is None:
            try:
                from database import db_manager
                state["names"] = {str(e["employee_id"]): (e.get("name") or "")
                                  for e in db_manager.get_employees()}
            except Exception:
                return {}
        return state["names"]

    # ── rendering (Tk thread) ───────────────────────────────────────────────
    def _render(idx: int, info: dict) -> None:
        tile = tiles.get(idx)
        if tile is None:
            return
        cam = CAMS[idx]
        if info.get("reconnecting"):
            tile["cap"].set(f"Camera {cam} — ⚠ {info['reconnecting']}")
            return
        tile["cap"].set(f"Camera {cam} — {info.get('n_tracks', 0)} tracked · "
                        f"{info.get('fps', 0):.1f} fps · seen {len(info.get('present_ids', []))}")
        now = datetime.datetime.now().strftime("%H:%M:%S")
        for eid in info.get("present_ids", []):
            state["marked"].setdefault(str(eid), (f"CAM-{cam}", now))
        img = info.get("preview")
        if not preview_flag["on"]:
            if state["refs"].pop(idx, None) is not None:
                tile["img"].configure(image=state["blank"], text=_OFF_TEXT)
            return
        if img is not None and Image is not None:
            try:
                import cv2
                # cv2 INTER_AREA downscale before the PIL hand-off — with 7
                # tiles the old PIL resize dominated the Tk thread.
                h, w = img.shape[:2]
                scale = TILE_W / w
                if scale < 1.0:
                    img = cv2.resize(img, (TILE_W, int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ph = ImageTk.PhotoImage(Image.fromarray(rgb))
                tile["img"].configure(image=ph, text="")
                state["refs"][idx] = ph
            except Exception:
                pass

    def _refresh_roster() -> None:
        multi = state["multi"]
        if not multi:
            return
        names = _emp_names()
        present = sorted(multi.all_present(),
                         key=lambda e: (len(str(e)), str(e)))
        present_hdr.set(f"Marked present — {len(present)} of {len(names) or '?'}")
        present_list.delete(0, tk.END)
        for eid in present:
            cam, seen_at = state["marked"].get(str(eid), ("—", "—"))
            present_list.insert(
                tk.END,
                f"{str(eid):<6}{names.get(str(eid), '')[:24]:<26}{cam:<12}{seen_at:<10}")

    def _drain() -> None:
        # newest tick per camera only — rendering every backlogged frame meant
        # 7 tiles' worth of discarded resize work on each pass.
        latest: dict = {}
        events = []
        try:
            while True:
                idx, info = q.get_nowait()
                if info.get("exit") is not None:
                    events.append(("out", info["exit"]))   # each is an event:
                    continue                               # never collapse them
                if info.get("entry") is not None:
                    events.append(("in", info["entry"]))
                    continue
                latest[idx] = info
        except queue.Empty:
            pass
        for kind, ev in events:
            _show_exit(ev) if kind == "out" else _show_entry(ev)
        for idx, info in latest.items():
            _render(idx, info)
        if latest:
            _refresh_roster()
        multi = state["multi"]
        if multi is not None and not multi.is_running() and start_btn["state"] == "disabled":
            start_btn.configure(state="normal"); stop_btn.configure(state="disabled")
            errs = [r.error for r in multi.runners if r.error]
            status_var.set(f"Stopped. {len(multi.all_present())} present."
                           + (f" {len(errs)} camera error(s)." if errs else ""))
        frame.after(150, _drain)

    # ── start / stop ────────────────────────────────────────────────────────
    def start():
        if state["multi"] and state["multi"].is_running():
            return
        stream = "main" if stream_var.get().startswith("main") else "sub"
        try:
            multi = live_camera.MultiCameraLive(cams=CAMS, mode=mode_var.get(),
                                                stream=stream,
                                                want_preview=lambda: preview_flag["on"])
        except Exception as e:
            messagebox.showerror("Cannot start", str(e)); return
        state["multi"] = multi
        state["names"] = None                 # re-read roster on each start
        state["marked"].clear()
        status_var.set(f"Connecting {len(CAMS)} cameras & loading models…")
        start_btn.configure(state="disabled"); stop_btn.configure(state="normal")
        multi.start(on_update=lambda i, info: q.put((i, info)))

    def stop():
        if state["multi"]:
            status_var.set("Stopping all cameras…")
            state["multi"].stop()

    start_btn.configure(command=start)
    stop_btn.configure(command=stop)
    frame.after(150, _drain)
    return frame
