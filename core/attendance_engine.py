"""
core/attendance_engine.py
Turns per-track identity evidence (possibly from MANY videos of the same
date) into exactly ONE attendance record per employee per date.

This module is pure logic (no models, no DB) so it is unit tested.

Attendance policy (presence-only — no entry/exit, no working hours):

    present ids = set()
    every CONFIRMED/LOCKED track whose fused confidence clears
    PRESENT_MIN_CONFIDENCE adds its employee_id to the set — once.
    Evidence for the same employee from multiple tracks/videos is MERGED
    into the single record (it strengthens confidence; it never duplicates).

GateEvent is kept only because the pipeline can emit crossings for gate
cameras; it is IGNORED for the attendance decision by design.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config import settings
from core.identity_evidence import CONFIRMED, LOCKED

PRESENT = "PRESENT"
REVIEW = "REVIEW"
ABSENT = "ABSENT"


@dataclass
class TrackEvidence:
    """One identified track's contribution, emitted by the pipeline."""
    employee_id: str
    status: str                    # CONFIRMED / LOCKED / CANDIDATE / ...
    confidence: float              # 0..1 fused track confidence
    first_seen: datetime
    last_seen: datetime
    camera_id: str
    camera_location: str
    camera_type: str
    video_name: str
    n_accepted: int                # accepted face observations on the track
    n_body: int = 0                # accepted body observations (Phase 2)


@dataclass
class GateEvent:                   # emitted for gate cameras; NOT used for attendance
    employee_id: str
    timestamp: datetime
    event_type: str                # "IN" / "OUT"
    camera_id: str
    video_name: str
    confidence: float


@dataclass
class DailyRecord:
    employee_id: str
    date: str
    attendance_status: str         # PRESENT / REVIEW
    confidence: float
    evidence_type: str             # "face" / "face+body"
    face_evidence_count: int = 0
    body_evidence_count: int = 0
    videos_seen: List[str] = field(default_factory=list)
    n_tracks: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    review_required: bool = False
    review_reason: str = ""
    has_confirmed: bool = False


def fuse_day(date: str, evidences: List[TrackEvidence],
             gate_events: Optional[List[GateEvent]] = None) -> Dict[str, DailyRecord]:
    """Merge all track evidence for one date into one record per employee.

    A set-based present decision: an employee is added at most once no matter
    how many frames, tracks, or videos they appear in.
    """
    records: Dict[str, DailyRecord] = {}

    for ev in evidences:
        confirmed = ev.status in (CONFIRMED, LOCKED)
        candidate_review = (not confirmed
                            and ev.n_accepted >= settings.REVIEW_MIN_ACCEPTED)
        # confirmed tracks drive attendance; sustained candidates (repeated
        # accepted matches that never met the strong-confirmation standard)
        # surface as REVIEW rows only — visible to a human, never auto-PRESENT
        if not (confirmed or candidate_review):
            continue
        rec = records.get(ev.employee_id)
        if rec is None:
            rec = DailyRecord(employee_id=ev.employee_id, date=date,
                              attendance_status=REVIEW, confidence=0.0,
                              evidence_type="face")
            records[ev.employee_id] = rec
        if confirmed:
            rec.has_confirmed = True

        rec.n_tracks += 1
        rec.face_evidence_count += max(ev.n_accepted, 0)
        rec.body_evidence_count += max(ev.n_body, 0)
        if ev.video_name and ev.video_name not in rec.videos_seen:
            rec.videos_seen.append(ev.video_name)
        rec.first_seen = min(rec.first_seen, ev.first_seen) if rec.first_seen else ev.first_seen
        rec.last_seen = max(rec.last_seen, ev.last_seen) if rec.last_seen else ev.last_seen
        # multiple independent confirmations strengthen, never duplicate:
        # keep the best track confidence, nudged up slightly per extra track
        rec.confidence = max(rec.confidence, ev.confidence)
        if rec.n_tracks > 1:
            rec.confidence = min(1.0, rec.confidence + 0.02)
        if rec.body_evidence_count > 0:
            rec.evidence_type = "face+body"

    # final status per employee (exactly one record each — dict keyed by id)
    for rec in records.values():
        if not rec.has_confirmed:
            rec.attendance_status = REVIEW
            rec.review_required = True
            rec.review_reason = ("repeated face matches but none met the "
                                 "strong-confirmation standard — verify manually")
            rec.confidence = round(float(min(rec.confidence, 0.49)), 4)
            continue
        if rec.confidence >= settings.PRESENT_MIN_CONFIDENCE:
            rec.attendance_status = PRESENT
            rec.review_required = False
        elif rec.confidence >= settings.REVIEW_MIN_CONFIDENCE:
            rec.attendance_status = REVIEW
            rec.review_required = True
            rec.review_reason = (f"fused confidence {rec.confidence:.2f} below "
                                 f"PRESENT threshold {settings.PRESENT_MIN_CONFIDENCE}")
        else:
            rec.attendance_status = REVIEW
            rec.review_required = True
            rec.review_reason = "identity evidence too weak"
        rec.confidence = round(float(rec.confidence), 4)

    return records


def full_roster_view(date: str, records: Dict[str, DailyRecord],
                     all_employee_ids: List[str]) -> List[dict]:
    """Present/Review/Absent rows for the WHOLE registered roster.
    Employees never seen get ABSENT. One row per employee, guaranteed."""
    rows = []
    for emp in sorted(set(all_employee_ids)):
        rec = records.get(emp)
        if rec is None:
            rows.append({"employee_id": emp, "date": date, "status": ABSENT,
                         "confidence": 0.0, "evidence_type": "",
                         "face_evidence_count": 0, "body_evidence_count": 0,
                         "videos_seen": "", "review_required": False})
        else:
            rows.append({"employee_id": emp, "date": date,
                         "status": rec.attendance_status,
                         "confidence": rec.confidence,
                         "evidence_type": rec.evidence_type,
                         "face_evidence_count": rec.face_evidence_count,
                         "body_evidence_count": rec.body_evidence_count,
                         "videos_seen": "; ".join(rec.videos_seen),
                         "review_required": rec.review_required})
    return rows
