"""gui/registration_tab.py — register/edit employees in the database."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox

BG, FG, SUB = "#0f1626", "#cfe0ff", "#9fb7d8"


def build_registration_tab(parent):
    from database import db_manager, db_setup
    db_setup.init_db()

    frame = tk.Frame(parent, bg=BG)

    left = tk.Frame(frame, bg=BG); left.pack(side="left", fill="y", padx=12, pady=12)
    tk.Label(left, text="Register Employee", bg=BG, fg=FG,
             font=("Verdana", 12, "bold")).pack(anchor="w", pady=(0, 8))

    entries = {}
    for lbl in ("Employee ID", "Name", "Department", "Designation"):
        tk.Label(left, text=lbl, bg=BG, fg=SUB).pack(anchor="w")
        e = tk.Entry(left, width=28); e.pack(anchor="w", pady=(0, 6))
        entries[lbl] = e

    cols = ("employee_id", "name", "department", "designation")
    right = tk.Frame(frame, bg=BG); right.pack(side="left", fill="both", expand=True, padx=12, pady=12)
    header = tk.Frame(right, bg=BG); header.pack(fill="x")
    tk.Label(header, text="Registered Employees", bg=BG, fg=FG,
             font=("Verdana", 12, "bold")).pack(side="left")
    tree = ttk.Treeview(right, columns=cols, show="headings", height=16)
    for c, w in zip(cols, (100, 180, 130, 130)):
        tree.heading(c, text=c); tree.column(c, width=w)
    tree.pack(fill="both", expand=True, pady=6)

    def refresh():
        for i in tree.get_children():
            tree.delete(i)
        for emp in db_manager.get_employees():
            # iid keeps the real employee_id string; Treeview "values" coerces
            # "01" to the int 1, which breaks DB lookups on retrieval
            tree.insert("", "end", iid=emp["employee_id"],
                        values=[emp[c] for c in cols])

    def save():
        eid = entries["Employee ID"].get().strip()
        name = entries["Name"].get().strip()
        if not eid or not name:
            messagebox.showwarning("Missing data", "Employee ID and Name are required.")
            return
        db_manager.add_employee(eid, name, entries["Department"].get(),
                                entries["Designation"].get())
        refresh()
        messagebox.showinfo("Saved", f"{eid} — {name} registered.\n\n"
                            "Next: enroll their face on the Enrollment tab.")

    def remove():
        sel = tree.selection()
        if not sel:
            return
        eid = sel[0]
        if messagebox.askyesno("Delete", f"Delete {eid} and their attendance rows?"):
            db_manager.delete_employee(str(eid))
            from core.embedding_gallery import get_gallery
            get_gallery().remove_employee(str(eid))
            refresh()

    tk.Button(left, text="💾 Save / Update", command=save,
              bg="#2f6fd0", fg="white").pack(anchor="w", pady=8)
    tk.Button(header, text="🗑 Delete selected", command=remove,
              bg="#7a1f1f", fg="white").pack(side="right")

    refresh()
    return frame
