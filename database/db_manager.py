"""
database/db_manager.py
All reads/writes. save_daily_record() UPSERTs on (employee_id, attendance_date):
re-processing the same date, or adding a second video later, UPDATES the single
existing row — it never creates a duplicate attendance row.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from core.attendance_engine import DailyRecord, GateEvent
from database.db_setup import connect, init_db


# ── employees ──────────────────────────────────────────────────────────────
def add_employee(employee_id: str, name: str, department: str = "",
                 designation: str = "") -> None:
    con = connect()
    try:
        con.execute(
            """INSERT INTO employees (employee_id, name, department, designation)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(employee_id) DO UPDATE SET
                 name=excluded.name, department=excluded.department,
                 designation=excluded.designation""",
            (employee_id.strip(), name.strip(), department.strip(), designation.strip()))
        con.commit()
    finally:
        con.close()


def get_employees() -> List[dict]:
    con = connect()
    try:
        rows = con.execute(
            "SELECT * FROM employees ORDER BY employee_id").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def get_employee(employee_id: str) -> Optional[dict]:
    con = connect()
    try:
        r = con.execute("SELECT * FROM employees WHERE employee_id=?",
                        (employee_id,)).fetchone()
        return dict(r) if r else None
    finally:
        con.close()


def delete_employee(employee_id: str) -> None:
    con = connect()
    try:
        con.execute("DELETE FROM attendance WHERE employee_id=?", (employee_id,))
        con.execute("DELETE FROM employees WHERE employee_id=?", (employee_id,))
        con.commit()
    finally:
        con.close()


# ── attendance ─────────────────────────────────────────────────────────────
def save_daily_record(rec: DailyRecord) -> None:
    """UPSERT: one row per (employee, date), always."""
    con = connect()
    try:
        con.execute(
            """INSERT INTO attendance
                 (employee_id, attendance_date, status, confidence, evidence_type,
                  face_evidence_count, body_evidence_count, videos_seen, n_tracks,
                  first_seen, last_seen, review_required, review_reason, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
               ON CONFLICT(employee_id, attendance_date) DO UPDATE SET
                 status=excluded.status,
                 confidence=MAX(attendance.confidence, excluded.confidence),
                 evidence_type=excluded.evidence_type,
                 face_evidence_count=excluded.face_evidence_count,
                 body_evidence_count=excluded.body_evidence_count,
                 videos_seen=excluded.videos_seen,
                 n_tracks=excluded.n_tracks,
                 first_seen=excluded.first_seen,
                 last_seen=excluded.last_seen,
                 review_required=excluded.review_required,
                 review_reason=excluded.review_reason,
                 updated_at=datetime('now')""",
            (rec.employee_id, rec.date, rec.attendance_status, rec.confidence,
             rec.evidence_type, rec.face_evidence_count, rec.body_evidence_count,
             "; ".join(rec.videos_seen), rec.n_tracks,
             rec.first_seen.isoformat(sep=" ") if rec.first_seen else None,
             rec.last_seen.isoformat(sep=" ") if rec.last_seen else None,
             1 if rec.review_required else 0, rec.review_reason))
        con.commit()
    finally:
        con.close()


def set_attendance_status(employee_id: str, date: str, status: str,
                          reason: str = "manual review decision") -> None:
    """Used by the review screen to resolve REVIEW rows by hand."""
    con = connect()
    try:
        con.execute(
            """UPDATE attendance SET status=?, review_required=0, review_reason=?,
                      updated_at=datetime('now')
               WHERE employee_id=? AND attendance_date=?""",
            (status, reason, employee_id, date))
        con.commit()
    finally:
        con.close()


def get_daily_attendance(date: str) -> List[dict]:
    """Joined roster view: EVERY registered employee, exactly one row.
    Employees with no attendance row for the date come back as ABSENT."""
    con = connect()
    try:
        rows = con.execute(
            """SELECT e.employee_id, e.name, e.department, e.designation,
                      COALESCE(a.status, 'ABSENT')        AS status,
                      COALESCE(a.confidence, 0)           AS confidence,
                      COALESCE(a.evidence_type, '')       AS evidence_type,
                      COALESCE(a.face_evidence_count, 0)  AS face_evidence_count,
                      COALESCE(a.body_evidence_count, 0)  AS body_evidence_count,
                      COALESCE(a.videos_seen, '')         AS videos_seen,
                      COALESCE(a.review_required, 0)      AS review_required,
                      a.first_seen, a.last_seen, ? AS attendance_date
               FROM employees e
               LEFT JOIN attendance a
                 ON a.employee_id = e.employee_id AND a.attendance_date = ?
               ORDER BY e.employee_id""", (date, date)).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


# ── gate events (diagnostic only) ─────────────────────────────────────────
def save_gate_events(date: str, events: List[GateEvent]) -> None:
    if not events:
        return
    con = connect()
    try:
        con.executemany(
            """INSERT INTO gate_events
               (employee_id, event_date, timestamp, event_type, camera_id,
                video_name, confidence) VALUES (?,?,?,?,?,?,?)""",
            [(e.employee_id, date, e.timestamp.isoformat(sep=" "), e.event_type,
              e.camera_id, e.video_name, e.confidence) for e in events])
        con.commit()
    finally:
        con.close()


def get_events(date: str) -> List[dict]:
    con = connect()
    try:
        rows = con.execute(
            "SELECT * FROM gate_events WHERE event_date=? ORDER BY timestamp",
            (date,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()
