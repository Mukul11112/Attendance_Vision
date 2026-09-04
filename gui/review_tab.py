"""gui/review_tab.py — shows WHY people were or weren't recognized, per video.
Displays data/recognition_review/<video>/: the report lines plus every saved
face crop (verdict text is burned into each image)."""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk

from config import settings

BG, FG, SUB = "#0f1626", "#cfe0ff", "#9fb7d8"
REVIEW_ROOT = os.path.join(settings.DATA_DIR, "recognition_review")


def build_review_tab(parent):
    try:
        from PIL import Image, ImageTk
    except Exception:
        Image = ImageTk = None

    frame = tk.Frame(parent, bg=BG)
    top = tk.Frame(frame, bg=BG); top.pack(fill="x", padx=12, pady=10)
    tk.Label(top, text="Video", bg=BG, fg=SUB).pack(side="left")
    combo = ttk.Combobox(top, width=44, state="readonly"); combo.pack(side="left", padx=6)

    summary_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=summary_var, bg=BG, fg="#ffd97a",
             justify="left", anchor="w").pack(fill="x", padx=12)

    report_box = tk.Text(frame, height=9, bg="#0a1020", fg=FG, wrap="none")
    report_box.pack(fill="x", padx=12)

    # scrollable image area
    holder = tk.Frame(frame, bg=BG); holder.pack(fill="both", expand=True, padx=12, pady=8)
    canvas = tk.Canvas(holder, bg=BG, highlightthickness=0)
    vsb = tk.Scrollbar(holder, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=BG)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
    frame._imgs = []          # keep PhotoImage references alive

    def refresh_videos():
        vids = []
        if os.path.isdir(REVIEW_ROOT):
            vids = sorted(d for d in os.listdir(REVIEW_ROOT)
                          if os.path.isdir(os.path.join(REVIEW_ROOT, d)))
        combo["values"] = vids
        summary_var.set(_overall_summary(vids))
        if vids and not combo.get():
            combo.set(vids[-1]); load()

    def _overall_summary(vids):
        """One answer across ALL processed videos: for each employee, the best
        similarity any unrecognized face ever reached, plus rejection totals."""
        import re
        best = {}                      # emp -> max near-miss sim
        reasons = {}                   # reason -> count
        never = ambiguous = idd = 0
        for v in vids:
            rp = os.path.join(REVIEW_ROOT, v, "report.txt")
            if not os.path.isfile(rp):
                continue
            for line in open(rp, encoding="utf-8"):
                if "IDENTIFIED" in line:
                    idd += 1
                m = re.search(r"best sim ([0-9.]+) -> (\S+)", line)
                if m:
                    s, emp = float(m.group(1)), m.group(2)
                    best[emp] = max(best.get(emp, 0.0), s)
                if "AMBIGUOUS" in line:
                    ambiguous += 1
                if "face never detected" in line:
                    never += 1
                for r, n in re.findall(r"([a-z ]+) x(\d+)", line):
                    reasons[r.strip()] = reasons.get(r.strip(), 0) + int(n)
        if not vids:
            return ""
        near = "  ".join(f"{e}: best {s:.2f}" for e, s in
                         sorted(best.items(), key=lambda kv: -kv[1]))
        rej = "  ".join(f"{r} x{n}" for r, n in
                        sorted(reasons.items(), key=lambda kv: -kv[1])[:4])
        return (f"ALL {len(vids)} VIDEOS — identified tracks: {idd} | "
                f"near-miss best similarity per employee (accept ≥ "
                f"{settings.FACE_SIMILARITY_ACCEPT}): {near or 'none'}\n"
                f"quality rejections: {rej or 'none'} | ambiguous: {ambiguous} | "
                f"face never detected: {never} tracks")

    def load(*_):
        for w in inner.winfo_children():
            w.destroy()
        frame._imgs.clear()
        report_box.delete("1.0", "end")
        vid = combo.get()
        if not vid:
            report_box.insert("end", "No review data yet — process a video first "
                              "(reports are written even on Cancel).")
            return
        d = os.path.join(REVIEW_ROOT, vid)
        rp = os.path.join(d, "report.txt")
        if os.path.isfile(rp):
            with open(rp, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            not_rec = [l for l in lines if "NOT RECOGNIZED" in l]
            idd = [l for l in lines if "IDENTIFIED" in l]
            report_box.insert("end", f"IDENTIFIED tracks: {len(idd)}   |   "
                              f"NOT recognized tracks with faces: {len(not_rec)}\n")
            report_box.insert("end", "\n".join(idd[:6] + not_rec[:40]))
        imgs = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
        if Image is None:
            tk.Label(inner, text="Install Pillow to see face crops here "
                     f"(images are in {d})", bg=BG, fg=SUB).grid(row=0, column=0)
            return
        cols = 4
        for i, fn in enumerate(imgs):
            try:
                im = Image.open(os.path.join(d, fn))
                ph = ImageTk.PhotoImage(im)
                frame._imgs.append(ph)
                cell = tk.Frame(inner, bg=BG); cell.grid(row=i // cols, column=i % cols,
                                                         padx=6, pady=6, sticky="nw")
                tk.Label(cell, image=ph, bg=BG).pack()
                tk.Label(cell, text=fn, bg=BG, fg=SUB).pack()
            except Exception:
                continue

    combo.bind("<<ComboboxSelected>>", load)
    tk.Button(top, text="⟳ Refresh", command=refresh_videos).pack(side="left", padx=6)
    tk.Label(top, text="Each unrecognized person's best face, with the exact "
             "reason under it.", bg=BG, fg=SUB).pack(side="left", padx=10)
    refresh_videos()
    frame.bind("<Visibility>", lambda e: refresh_videos())
    return frame
