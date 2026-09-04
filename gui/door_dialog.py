"""
gui/door_dialog.py
Set the main-door exit line by looking at the camera.

Grabs one frame from the door camera, draws the candidate line over it, and
lets you drag it to sit where you want (just past the sofa) and say which side
is outside. Nobody can guess a line position from a config file — this makes it
a thing you see.
"""
from __future__ import annotations
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from core import door, live_camera

log = logging.getLogger("door_dialog")

try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None

PREVIEW_W = 720


def open_door_dialog(parent) -> None:
    cfg = door.load_config()
    win = tk.Toplevel(parent)
    win.title("Main-door exit line")
    win.configure(bg="#0f1626")
    win.transient(parent)
    win.grab_set()

    tk.Label(win, text="Drag either END of the green line onto the doorway "
                       "(the line may be diagonal).",
             bg="#0f1626", fg="#cfe0ff", font=("Verdana", 10, "bold")
             ).pack(anchor="w", padx=12, pady=(10, 0))
    tk.Label(win, text="Crossing towards OUT marks the person as gone; "
                       "crossing towards IN marks them present.",
             bg="#0f1626", fg="#9fb7d8").pack(anchor="w", padx=12, pady=(0, 2))

    top = tk.Frame(win, bg="#0f1626"); top.pack(fill="x", padx=12, pady=4)
    tk.Label(top, text="Camera", bg="#0f1626", fg="#9fb7d8").pack(side="left")
    cam_var = tk.IntVar(value=int(cfg.get("camera", 5)))
    ttk.Combobox(top, textvariable=cam_var, state="readonly", width=4,
                 values=[1, 2, 3, 4, 5, 6, 7]).pack(side="left", padx=(4, 14))

    out_var = tk.StringVar(value=cfg.get("out_direction", "down"))
    tk.Label(top, text="Leaving means crossing", bg="#0f1626",
             fg="#9fb7d8").pack(side="left")
    ttk.Combobox(top, textvariable=out_var, state="readonly", width=18,
                 values=["down (towards bottom)", "up (towards top)"]
                 ).pack(side="left", padx=4)
    # combobox shows a phrase; keep the stored value short
    if out_var.get() == "down":
        out_var.set("down (towards bottom)")
    elif out_var.get() == "up":
        out_var.set("up (towards top)")

    enabled_var = tk.BooleanVar(value=bool(cfg.get("enabled", True)))
    tk.Checkbutton(top, text="watch this door", variable=enabled_var,
                   bg="#0f1626", fg="#9fb7d8", selectcolor="#0a1020",
                   activebackground="#0f1626", highlightthickness=0
                   ).pack(side="left", padx=14)

    canvas = tk.Canvas(win, width=PREVIEW_W, height=int(PREVIEW_W * 9 / 16),
                       bg="#05080f", highlightthickness=0)
    canvas.pack(padx=12, pady=6)

    pts = list(cfg.get("line_points") or [0.15, 0.62, 0.36, 0.39])
    state = {"photo": None, "pts": pts, "h": int(PREVIEW_W * 9 / 16),
             "loading": False, "drag": None}

    def _px():
        x1, y1, x2, y2 = state["pts"]
        return (x1 * PREVIEW_W, y1 * state["h"], x2 * PREVIEW_W, y2 * state["h"])

    def _redraw_line() -> None:
        canvas.delete("line")
        x1, y1, x2, y2 = _px()
        canvas.create_line(x1, y1, x2, y2, fill="#22dd44", width=4, tags="line")
        for (hx, hy) in ((x1, y1), (x2, y2)):        # draggable end handles
            canvas.create_oval(hx - 7, hy - 7, hx + 7, hy + 7,
                               fill="#22dd44", outline="white", width=2,
                               tags="line")
        # arrow showing which way counts as leaving
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        nlen = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / nlen, dx / nlen               # unit normal
        sign = -1 if out_var.get().startswith("up") else 1
        canvas.create_line(mx, my, mx + sign * nx * 70, my + sign * ny * 70,
                           fill="#ff5555", width=3, arrow="last", tags="line")
        canvas.create_text(mx + sign * nx * 88, my + sign * ny * 88,
                           text="OUT", fill="#ff5555",
                           font=("Verdana", 10, "bold"), tags="line")
        canvas.create_text(mx - sign * nx * 88, my - sign * ny * 88,
                           text="IN", fill="#55ff88",
                           font=("Verdana", 10, "bold"), tags="line")

    def _grab_frame() -> None:
        """One frame from the door camera, in a thread (RTSP open is slow)."""
        if state["loading"]:
            return
        state["loading"] = True
        canvas.delete("hint")
        canvas.create_text(PREVIEW_W // 2, state["h"] // 2, tags="hint",
                           text="connecting to camera…", fill="#6f86a8")

        def work() -> None:
            frame = None
            err = ""
            try:
                import cv2
                ccfg = live_camera.load_config()
                url = live_camera.build_rtsp_url(ccfg, f"{cam_var.get()}01")
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    for _ in range(5):        # first frames can be torn
                        ok, f = cap.read()
                        if ok:
                            frame = f
                else:
                    err = "could not open the camera stream"
                cap.release()
            except Exception as e:
                err = str(e)
            win.after(0, lambda: _show(frame, err))

        threading.Thread(target=work, daemon=True).start()

    def _show(frame, err: str) -> None:
        state["loading"] = False
        canvas.delete("hint")
        if frame is None or Image is None:
            canvas.create_text(PREVIEW_W // 2, state["h"] // 2, tags="hint",
                               text=err or "no frame — you can still set the "
                                           "line by number below",
                               fill="#c88", width=PREVIEW_W - 40)
            _redraw_line()
            return
        import cv2
        h, w = frame.shape[:2]
        scale = PREVIEW_W / w
        state["h"] = int(h * scale)
        canvas.configure(height=state["h"])
        small = cv2.resize(frame, (PREVIEW_W, state["h"]),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.delete("bg")
        canvas.create_image(0, 0, anchor="nw", image=state["photo"], tags="bg")
        canvas.tag_lower("bg")
        _redraw_line()

    def _on_press(evt) -> None:
        """Grab whichever end of the line was clicked."""
        x1, y1, x2, y2 = _px()
        d1 = (evt.x - x1) ** 2 + (evt.y - y1) ** 2
        d2 = (evt.x - x2) ** 2 + (evt.y - y2) ** 2
        state["drag"] = 0 if d1 <= d2 else 1

    def _on_drag(evt) -> None:
        if state["drag"] is None:
            return
        fx = min(0.99, max(0.01, evt.x / PREVIEW_W))
        fy = min(0.99, max(0.01, evt.y / max(1, state["h"])))
        i = state["drag"] * 2
        state["pts"][i], state["pts"][i + 1] = fx, fy
        _redraw_line()

    canvas.bind("<Button-1>", _on_press)
    canvas.bind("<B1-Motion>", _on_drag)
    canvas.bind("<ButtonRelease-1>", lambda e: state.__setitem__("drag", None))
    out_var.trace_add("write", lambda *_a: _redraw_line())   # flip the arrows

    btns = tk.Frame(win, bg="#0f1626"); btns.pack(fill="x", padx=12, pady=(4, 12))
    tk.Button(btns, text="↻ Refresh frame", command=_grab_frame,
              bg="#1b2740", fg="#cfe0ff", relief="flat").pack(side="left")

    def _save() -> None:
        door.save_config({
            "camera": int(cam_var.get()),
            "line_points": [round(float(v), 4) for v in state["pts"]],
            "line_fraction": None,          # superseded by the two-point line
            "out_direction": "down" if out_var.get().startswith("down") else "up",
            "enabled": bool(enabled_var.get()),
        })
        messagebox.showinfo(
            "Saved",
            "Exit line saved.\n\nRestart the All Cameras run for it to take "
            "effect — the line is handed to each camera when the run starts.",
            parent=win)
        win.destroy()

    tk.Button(btns, text="Save", command=_save, bg="#1f7a3f", fg="white",
              font=("Verdana", 10, "bold"), width=10).pack(side="right")
    tk.Button(btns, text="Cancel", command=win.destroy, bg="#3a3f52",
              fg="#cfe0ff", relief="flat", width=8).pack(side="right", padx=6)

    _redraw_line()
    win.after(200, _grab_frame)
