"""gui/attendance_tab.py — view one date's roster (Present/Review/Absent),
resolve REVIEW rows manually, export the Excel report."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date as _date

BG, FG, SUB = "#0f1626", "#cfe0ff", "#9fb7d8"
COLS = ("employee_id", "name", "department", "status", "confidence",
        "evidence_type", "face_evidence_count", "body_evidence_count",
        "videos_seen", "review_required")


def build_attendance_tab(parent):
    from database import db_manager, db_setup
    db_setup.init_db()

    frame = tk.Frame(parent, bg=BG)
    top = tk.Frame(frame, bg=BG); top.pack(fill="x", padx=12, pady=10)
    tk.Label(top, text="Date (YYYY-MM-DD)", bg=BG, fg=SUB).pack(side="left")
    date_e = tk.Entry(top, width=12); date_e.insert(0, _date.today().isoformat())
    date_e.pack(side="left", padx=6)

    tree = ttk.Treeview(frame, columns=COLS, show="headings", height=18)
    widths = (90, 150, 110, 80, 85, 95, 60, 60, 220, 70)
    for c, w in zip(COLS, widths):
        tree.heading(c, text=c); tree.column(c, width=w)
    tree.tag_configure("PRESENT", background="#153a1c", foreground="#bff0c4")
    tree.tag_configure("REVIEW", background="#3a3115", foreground="#ffe9a8")
    tree.tag_configure("ABSENT", background="#3a1717", foreground="#f2c0c0")
    summary = tk.StringVar(value="")
    tk.Label(top, textvariable=summary, bg=BG, fg=FG).pack(side="right", padx=6)

    tree.pack(fill="both", expand=True, padx=12, pady=6)

    def load():
        for i in tree.get_children():
            tree.delete(i)
        rows = db_manager.get_daily_attendance(date_e.get().strip())
        for r in rows:
            vals = [r.get(c, "") for c in COLS]
            vals[9] = "YES" if r.get("review_required") else ""
            # iid keeps the real employee_id string; Treeview "values" coerces
            # "01" to the int 1, which breaks DB lookups on retrieval
            tree.insert("", "end", iid=r["employee_id"], values=vals,
                        tags=(r.get("status", ""),))
        n = {"PRESENT": 0, "REVIEW": 0, "ABSENT": 0}
        for r in rows:
            n[r.get("status", "ABSENT")] = n.get(r.get("status", "ABSENT"), 0) + 1
        summary.set(f"Registered {len(rows)} | Present {n['PRESENT']} | "
                    f"Review {n['REVIEW']} | Absent {n['ABSENT']}")

    def _resolve(status):
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Select row", "Select an employee row first.")
            return
        eid = sel[0]
        db_manager.set_attendance_status(eid, date_e.get().strip(), status)
        load()

    def export():
        from reports import excel_report
        d = date_e.get().strip()
        try:
            path = excel_report.export(d, db_manager.get_daily_attendance(d),
                                       db_manager.get_events(d))
            messagebox.showinfo("Exported", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    tk.Button(top, text="⟳ Load", command=load).pack(side="left", padx=4)
    tk.Button(top, text="✔ Mark selected PRESENT", command=lambda: _resolve("PRESENT"),
              bg="#1f7a35", fg="white").pack(side="left", padx=4)
    tk.Button(top, text="✘ Mark selected ABSENT", command=lambda: _resolve("ABSENT"),
              bg="#7a1f1f", fg="white").pack(side="left", padx=4)
    tk.Button(top, text="📊 Export Excel", command=export,
              bg="#2f6fd0", fg="white").pack(side="left", padx=12)

    load()
    return frame
