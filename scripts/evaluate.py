"""
scripts/evaluate.py — score the system against ground truth (Phase 3).

Ground truth CSV (you write this by hand — who was ACTUALLY in the office):
    date,employee_id
    2026-07-11,01
    2026-07-11,02
    2026-07-11,06

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py truth.csv
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py truth.csv --date 2026-07-11

Reports per date and overall:
    found      truly present AND system said PRESENT
    review     truly present, system said REVIEW (one click from correct)
    missed     truly present, system said ABSENT
    FALSE MARK truly absent,  system said PRESENT  <- the must-be-zero number
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 1
    truth_path = sys.argv[1]
    only_date = None
    if "--date" in sys.argv:
        only_date = sys.argv[sys.argv.index("--date") + 1]

    truth = defaultdict(set)          # date -> {employee_id}
    with open(truth_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = row["date"].strip()
            if not only_date or d == only_date:
                truth[d].add(row["employee_id"].strip())
    if not truth:
        print("No ground-truth rows matched."); return 1

    from database import db_manager, db_setup
    db_setup.init_db()

    tot = defaultdict(int)
    print(f"{'date':<12} {'found':>6} {'review':>7} {'missed':>7} "
          f"{'FALSE MARK':>11}   details")
    print("-" * 78)
    for d in sorted(truth):
        rows = {r["employee_id"]: r for r in db_manager.get_daily_attendance(d)}
        found, review, missed, false_mark = [], [], [], []
        for emp in sorted(truth[d]):
            st = rows.get(emp, {}).get("status", "ABSENT")
            (found if st == "PRESENT" else
             review if st == "REVIEW" else missed).append(emp)
        for emp, r in rows.items():
            if r["status"] == "PRESENT" and emp not in truth[d]:
                false_mark.append(emp)
        det = []
        if missed:
            det.append("missed: " + ",".join(missed))
        if false_mark:
            det.append("FALSE: " + ",".join(false_mark))
        if review:
            det.append("review: " + ",".join(review))
        print(f"{d:<12} {len(found):>6} {len(review):>7} {len(missed):>7} "
              f"{len(false_mark):>11}   {'; '.join(det)}")
        tot["found"] += len(found); tot["review"] += len(review)
        tot["missed"] += len(missed); tot["false"] += len(false_mark)

    n_truth = sum(len(v) for v in truth.values())
    print("-" * 78)
    print(f"TOTAL truly-present: {n_truth} | auto-found: {tot['found']} "
          f"({100*tot['found']/max(n_truth,1):.0f}%) | review: {tot['review']} | "
          f"missed: {tot['missed']} | FALSE MARKS: {tot['false']}")
    if tot["false"]:
        print("\n!! FALSE MARKS present — this violates the #1 rule. "
              "Send this output for calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
