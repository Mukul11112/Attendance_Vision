"""
database/db_setup.py
SQLite schema. The critical rule lives here:

    UNIQUE(employee_id, attendance_date)

so even if application logic ever tried to insert an employee twice for one
date, the database itself refuses the duplicate; writes use UPSERT.
"""
from __future__ import annotations
import sqlite3

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    department    TEXT DEFAULT '',
    designation   TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS attendance (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id          TEXT NOT NULL REFERENCES employees(employee_id),
    attendance_date      TEXT NOT NULL,               -- YYYY-MM-DD
    status               TEXT NOT NULL,               -- PRESENT / REVIEW
    confidence           REAL DEFAULT 0,
    evidence_type        TEXT DEFAULT 'face',
    face_evidence_count  INTEGER DEFAULT 0,
    body_evidence_count  INTEGER DEFAULT 0,
    videos_seen          TEXT DEFAULT '',
    n_tracks             INTEGER DEFAULT 0,
    first_seen           TEXT,
    last_seen            TEXT,
    review_required      INTEGER DEFAULT 0,
    review_reason        TEXT DEFAULT '',
    updated_at           TEXT DEFAULT (datetime('now')),
    UNIQUE (employee_id, attendance_date)
);

CREATE TABLE IF NOT EXISTS gate_events (               -- diagnostic only
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id   TEXT NOT NULL,
    event_date    TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    camera_id     TEXT DEFAULT '',
    video_name    TEXT DEFAULT '',
    confidence    REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(attendance_date);
CREATE INDEX IF NOT EXISTS idx_events_date     ON gate_events(event_date);
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(settings.DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    con = connect()
    try:
        con.executescript(SCHEMA)
        con.commit()
    finally:
        con.close()
