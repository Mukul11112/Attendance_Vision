"""
core/door.py
Main-door exit watch: who walked out, and when.

The rule (chosen deliberately on 31 Jul 2026):
    a track crossing the door line is named from the DAY'S body memory —
    clothes, shoulders, build harvested after a face lock earlier today —
    but ONLY if that person was face-confirmed present today.

That is what makes it safe. Someone walking out turned away from the camera
still gets named, because we already know what they are wearing today; but the
matcher can never invent a person who was never seen, which is the failure that
collapsed identities in July. Every alert records HOW it was identified, so a
body-matched exit is never mistaken for a face-confirmed one.
"""
from __future__ import annotations
import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Dict, List, Optional

from config import settings

log = logging.getLogger("door")

CONFIG_PATH = os.path.join(settings.BASE_DIR, "door_config.json")

DEFAULTS = {
    "camera": 5,                 # NVR channel 5 is named "Main-door"
    # The door line is DIAGONAL across the doorway. Traced from the site photo
    # of 31 Jul 2026 (green line in front of the glass door): it runs from the
    # lower-left (0.150, 0.591) to the upper-right (0.366, 0.372) of the frame.
    # Stored normalized 0-1 so it survives any frame size.
    "line_points": [0.150, 0.591, 0.366, 0.372],
    "line_fraction": None,       # legacy horizontal line, kept for old configs
    # Same photo: the red IN arrow points down-right INTO the room, the red OUT
    # arrow points up-left towards the glass door. With this line's orientation
    # the room side is the negative side, so leaving = "down" in the engine's
    # legacy naming. Verified by test_door_matches_site_photo.
    "out_direction": "down",
    "enabled": True,
    "min_body_similarity": 0.62,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("door_config.json unreadable (%s) — using defaults", e)
    return cfg


def save_config(cfg: dict) -> None:
    merged = dict(DEFAULTS)
    merged.update(cfg)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)


@dataclass
class ExitEvent:
    employee_id: str
    name: str
    at: str                      # HH:MM:SS
    camera_id: str
    how: str                     # "face" | "body"
    confidence: float

    def headline(self) -> str:
        return f"{self.name or self.employee_id} left the office at {self.at}"


class DoorWatch:
    """Decides whether a crossing is a real exit worth alerting on.

    Kept separate from the pipeline so the rule is testable and so the GUI can
    ask questions ("who is out?") without touching recognition code.
    """

    def __init__(self, day: Optional[str] = None) -> None:
        self.day = day or date.today().isoformat()
        self._lock = threading.Lock()
        self._exited: Dict[str, ExitEvent] = {}
        self._entered: Dict[str, ExitEvent] = {}
        self.cfg = load_config()

    # ── who is eligible to be named at the door ────────────────────────────
    def face_confirmed_today(self) -> Dict[str, str]:
        """{employee_id: name} for people already marked PRESENT today.

        This is the gate: the door may only put a name to someone in here.
        """
        try:
            from database import db_manager
            # get_daily_attendance returns EVERY employee with status
            # PRESENT / REVIEW / ABSENT — only PRESENT counts as confirmed.
            return {str(r["employee_id"]): (r.get("name") or "")
                    for r in db_manager.get_daily_attendance(self.day)
                    if str(r.get("status", "")).upper() == "PRESENT"}
        except Exception as e:
            log.warning("could not read today's present list: %s", e)
            return {}

    def is_door_camera(self, camera_id: str) -> bool:
        """Only the configured main-door camera may raise door alerts.

        Belt and braces: today only that camera is given a line, so only it can
        emit crossings — but if a line is ever configured elsewhere by mistake,
        its crossings must not be able to announce that someone left the office.
        """
        want = str(self.cfg.get("camera", 5))
        digits = "".join(ch for ch in str(camera_id) if ch.isdigit())
        return digits == want

    def identify(self, employee_id: Optional[str], how: str,
                 confidence: float) -> Optional[str]:
        """Return the employee_id to alert for, or None to stay silent."""
        if not employee_id:
            return None
        eligible = self.face_confirmed_today()
        if employee_id not in eligible:
            # Body matched someone who was never confirmed present today —
            # exactly the case we refuse to guess at.
            log.info("door: ignoring %s crossing (%s) — not face-confirmed today",
                     employee_id, how)
            return None
        return employee_id

    def record_exit(self, employee_id: str, how: str, confidence: float,
                    camera_id: str, when: Optional[datetime] = None
                    ) -> Optional[ExitEvent]:
        """Register an exit. Returns the event, or None if it is a duplicate."""
        if not self.is_door_camera(camera_id):
            log.info("door: ignoring OUT from %s — not the main-door camera",
                     camera_id)
            return None
        eid = self.identify(employee_id, how, confidence)
        if eid is None:
            return None
        with self._lock:
            if eid in self._exited:          # already gone; don't nag
                return None
            name = self.face_confirmed_today().get(eid, "")
            ev = ExitEvent(employee_id=eid, name=name,
                           at=(when or datetime.now()).strftime("%H:%M:%S"),
                           camera_id=camera_id, how=how, confidence=confidence)
            self._exited[eid] = ev
        self._persist(ev, when)
        return ev

    def record_entry(self, employee_id: str, how: str, confidence: float,
                     camera_id: str, when: Optional[datetime] = None
                     ) -> Optional[ExitEvent]:
        """Someone walked IN through the main door.

        Unlike an exit, an entry needs no "was already present" gate — walking
        in through the door is what makes you present. It clears any recorded
        exit so a later departure alerts again.
        """
        if not employee_id or not self.is_door_camera(camera_id):
            return None
        eid = str(employee_id)
        with self._lock:
            self._exited.pop(eid, None)
        try:
            from database import db_manager
            name = {str(e["employee_id"]): (e.get("name") or "")
                    for e in db_manager.get_employees()}.get(eid, "")
        except Exception:
            name = ""
        ev = ExitEvent(employee_id=eid, name=name,
                       at=(when or datetime.now()).strftime("%H:%M:%S"),
                       camera_id=camera_id, how=how, confidence=confidence)
        with self._lock:
            self._entered[eid] = ev
        self._persist(ev, when, "IN")
        return ev

    def is_in_office(self, employee_id: str) -> bool:
        with self._lock:
            return str(employee_id) not in self._exited

    def re_entered(self, employee_id: str) -> None:
        """Someone came back in — allow a later exit to alert again."""
        with self._lock:
            self._exited.pop(str(employee_id), None)

    def exits_today(self) -> List[ExitEvent]:
        with self._lock:
            return sorted(self._exited.values(), key=lambda e: e.at)

    def entries_today(self) -> List[ExitEvent]:
        with self._lock:
            return sorted(self._entered.values(), key=lambda e: e.at)

    def _persist(self, ev: ExitEvent, when: Optional[datetime],
                 kind: str = "OUT") -> None:
        try:
            from core.attendance_engine import GateEvent
            from database import db_manager
            db_manager.save_gate_events(self.day, [GateEvent(
                employee_id=ev.employee_id, timestamp=(when or datetime.now()),
                event_type=kind, camera_id=ev.camera_id,
                video_name=f"door:{ev.how}", confidence=ev.confidence)])
        except Exception as e:
            log.warning("could not persist %s event: %s", kind, e)


_watch: Optional[DoorWatch] = None
_w_lock = threading.Lock()


def get_door_watch(day: Optional[str] = None) -> DoorWatch:
    global _watch
    with _w_lock:
        today = day or date.today().isoformat()
        if _watch is None or _watch.day != today:
            _watch = DoorWatch(today)
        return _watch
