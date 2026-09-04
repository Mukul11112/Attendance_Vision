"""
core/live_camera.py
Continuous LIVE attendance from an RTSP camera (Hikvision).

Reuses the batch VideoPipeline unchanged: the pipeline's frame loop already
runs forever on a live stream (no total-frame count), accumulates a set of
CONFIRMED/LOCKED employees, and emits it via progress_cb. We simply:
  • build the RTSP URL from camera_config.json (credentials URL-encoded),
  • run the pipeline in this thread,
  • mark each newly-CONFIRMED employee PRESENT in the DB the moment they appear,
  • on stop, run the normal fuse_day() so stored confidences become the real
    per-track numbers (the UPSERT keeps the maximum).

Latency note: on a CPU-only box the pipeline processes slower than realtime, so
the live view lags the wall clock. That does NOT affect attendance correctness
(presence is a set — a person seen at any point is PRESENT); it only means the
preview/roster trails the true time. Use the sub-stream (channel 102) to reduce
the gap.
"""
from __future__ import annotations
import json
import os
import threading
from datetime import date, datetime
from typing import Callable, Optional
from urllib.parse import quote

from config import settings
from core import door
from core.pipeline import VideoPipeline, VideoJob, Progress
from core import attendance_engine as ae
from core.attendance_engine import DailyRecord
from database import db_manager, db_setup

CONFIG_PATH = os.path.join(settings.BASE_DIR, "camera_config.json")

# RTSP over TCP is far more reliable than UDP on office LANs (no torn frames);
# must be set before OpenCV's FFmpeg backend opens the stream.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_rtsp_url(cfg: dict, channel: Optional[str] = None) -> str:
    ip = cfg["ip"]
    port = cfg.get("port", 554)
    user = quote(str(cfg["username"]), safe="")
    pw = quote(str(cfg["password"]), safe="")
    ch = str(channel or cfg.get("channel", "101"))
    return f"rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/{ch}"


def build_playback_url(cfg: dict, channel: str, day: str,
                       start_clock: str = "00:00:00",
                       end_clock: str = "23:59:59") -> str:
    """RTSP URL for RECORDED footage already on the NVR — the same thing the
    Hikvision web UI's Playback tab plays.

    `day` is YYYY-MM-DD; clocks are HH:MM:SS in the NVR's own timezone. The
    /Streaming/tracks/ endpoint streams stored video as fast as the link allows
    (measured ~2.5x realtime here), which is what makes marking a past day
    practical. Verified working against this site's NVR on 31 Jul 2026.
    """
    ip = cfg["ip"]
    port = cfg.get("port", 554)
    user = quote(str(cfg["username"]), safe="")
    pw = quote(str(cfg["password"]), safe="")
    d = day.replace("-", "")
    s = start_clock.replace(":", "")
    e = end_clock.replace(":", "")
    return (f"rtsp://{user}:{pw}@{ip}:{port}/Streaming/tracks/{channel}"
            f"?starttime={d}T{s}Z&endtime={d}T{e}Z")


def redact(url: str, cfg: dict) -> str:
    return url.replace(quote(str(cfg["password"]), safe=""), "****")


class LiveAttendance:
    """Runs continuous live recognition until stop() is called.

    on_update(info: dict) is called on every pipeline progress tick with:
      {frame preview (np BGR), n_tracks, present_ids (sorted list),
       elapsed_s, fps}.
    """

    def __init__(self, camera_id: str = "LIVE-01", location: str = "office",
                 camera_type: str = "office_room", mode: str = "BALANCED",
                 channel: Optional[str] = None, want_preview=None,
                 url: Optional[str] = None, day: Optional[str] = None,
                 start_clock: Optional[str] = None,
                 line_fraction: Optional[float] = None,
                 out_is_down: Optional[bool] = None,
                 line_points: Optional[tuple] = None) -> None:
        # A line turns this camera into a gate: crossings become IN/OUT events.
        # Only the main-door camera sets one; everything else stays
        # presence-only, exactly as before.
        self.line_fraction = line_fraction
        self.line_points = line_points
        self.out_is_down = out_is_down
        self.want_preview = want_preview      # callable -> bool, or None = always
        self.cfg = load_config()
        # url/day/start_clock are set by NVR playback of a PAST day; live use
        # leaves them None and gets today's live stream as before.
        self.url = url or build_rtsp_url(self.cfg, channel)
        self.camera_id = camera_id
        self.location = location
        self.camera_type = camera_type
        self.mode = mode
        self.date = day or date.today().isoformat()
        self.start_clock = start_clock
        self.is_playback = url is not None

        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self.seen_present: set = set()
        self.error: Optional[str] = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, on_update: Optional[Callable[[dict], None]] = None) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(on_update,),
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def redact_url(self) -> str:
        return redact(self.url, self.cfg)

    # ── worker ─────────────────────────────────────────────────────────────
    def _mark_present(self, employee_id: str) -> None:
        """Provisional PRESENT the instant a track is CONFIRMED/LOCKED live.
        Overwritten with real fused confidence on stop() (UPSERT keeps max)."""
        # On NVR playback the wall clock is NOT the footage's clock — anchor the
        # provisional stamps to the day being replayed so an interrupted run
        # can't stamp yesterday's record with today's time. fuse_day() replaces
        # these with the real per-track numbers when the run completes.
        stamp = datetime.now()
        if self.is_playback:
            stamp = datetime.strptime(f"{self.date} {self.start_clock or '00:00:00'}",
                                      "%Y-%m-%d %H:%M:%S")
        tag = ("NVR" if self.is_playback else "LIVE") + f":{self.camera_id}"
        rec = DailyRecord(
            employee_id=employee_id, date=self.date,
            attendance_status="PRESENT", confidence=0.85,
            evidence_type="face", face_evidence_count=0,
            videos_seen=[tag], n_tracks=1,
            first_seen=stamp, last_seen=stamp,
            has_confirmed=True, review_required=False)
        db_manager.save_daily_record(rec)

    def _run(self, on_update: Optional[Callable[[dict], None]]) -> None:
        import time
        db_setup.init_db()
        self._t0 = time.time()
        job = VideoJob(
            path=self.url, date=self.date,
            start_time=(self.start_clock or datetime.now().strftime("%H:%M:%S")),
            camera_id=self.camera_id, camera_location=self.location,
            camera_type=self.camera_type, line_fraction=self.line_fraction,
            line_points=self.line_points, out_is_down=self.out_is_down)
        def _gate(ev) -> None:
            """A crossing on the door line — mark IN as present, OUT as gone."""
            watch = door.get_door_watch(self.date)
            when = ev.timestamp if isinstance(ev.timestamp, datetime) else None

            if ev.event_type == "IN":
                # Walking in through the main door IS the attendance mark.
                if ev.employee_id not in self.seen_present:
                    self._mark_present(ev.employee_id)
                    self.seen_present.add(ev.employee_id)
                entry = watch.record_entry(
                    employee_id=ev.employee_id, how="face",
                    confidence=float(ev.confidence or 0.0),
                    camera_id=self.camera_id, when=when)
                if entry and on_update:
                    on_update({"entry": entry, "preview": None, "n_tracks": 0,
                               "present_ids": sorted(self.seen_present),
                               "elapsed_s": 0.0, "fps": 0.0})
                return

            exit_ev = watch.record_exit(
                employee_id=ev.employee_id,
                how="face",                 # crossing carries a real identity
                confidence=float(ev.confidence or 0.0),
                camera_id=self.camera_id, when=when)
            if exit_ev and on_update:
                on_update({"exit": exit_ev, "preview": None, "n_tracks": 0,
                           "present_ids": sorted(self.seen_present),
                           "elapsed_s": 0.0, "fps": 0.0})

        has_line = (self.line_points is not None
                    or self.line_fraction is not None)
        pipe = VideoPipeline(mode=self.mode, want_preview=self.want_preview,
                             on_gate=_gate if has_line else None)

        def cb(pr: Progress) -> None:
            new = set(pr.present_ids) - self.seen_present
            for emp in sorted(new):
                self._mark_present(emp)
            self.seen_present.update(new)
            if on_update:
                on_update({
                    "preview": pr.preview,
                    "n_tracks": pr.n_tracks,
                    "present_ids": sorted(self.seen_present),
                    "elapsed_s": time.time() - self._t0,
                    "fps": pr.fps,
                })

        # RECONNECT LOOP. The NVR refuses a session under load ("453 Not Enough
        # Bandwidth") and an RTSP stream can drop at any time. Previously one
        # such refusal ended the thread for good: the camera's tile stayed black
        # until the whole run was restarted by hand — this is what happened to
        # camera 6 on 31 Jul 2026. A live camera should keep trying.
        import logging
        log_ = logging.getLogger("live_camera")
        attempt = 0
        max_attempts = None if not self.is_playback else 3   # live = forever
        while not self._stop:
            attempt += 1
            try:
                result = pipe.process(job, progress_cb=cb,
                                      cancel=lambda: self._stop)
                # final, honest fuse: replace provisional confidences with the
                # real per-track numbers the pipeline computed
                records = ae.fuse_day(self.date, result.evidences,
                                      result.gate_events)
                for rec in records.values():
                    db_manager.save_daily_record(rec)
                self.error = None
                return                                   # finished cleanly
            except Exception as e:
                self.error = str(e)
                log_.warning("%s attempt %d failed: %s", self.camera_id, attempt, e)
                if self._stop or (max_attempts and attempt >= max_attempts):
                    log_.exception("live attendance failed")
                    return
                delay = min(30.0, 2.0 * attempt)         # 2s, 4s, 6s … cap 30s
                if on_update:
                    on_update({"reconnecting": f"reconnecting in {delay:.0f}s "
                                               f"(attempt {attempt})",
                               "preview": None, "n_tracks": 0,
                               "present_ids": sorted(self.seen_present),
                               "elapsed_s": 0.0, "fps": 0.0})
                waited = 0.0
                while waited < delay and not self._stop:  # stay responsive to Stop
                    time.sleep(0.25)
                    waited += 0.25


class MultiCameraLive:
    """Run several NVR cameras at once for live attendance.

    Each camera gets its own LiveAttendance (own tracker/identity state) but they
    SHARE the model singletons (get_gallery/get_embedder/...), so memory stays
    flat no matter how many cameras. The CPU is the shared bottleneck: on a
    CPU-only box, N cameras each run ~1/N as often as one would — fine for
    attendance (presence is a set fused across all cameras by employee).

    stream="sub" (channel x02) is the sensible default; "main" (x01) is full HD
    and will crawl with many cameras on CPU.
    """

    def __init__(self, cams=None, mode: str = "BALANCED", stream: str = "sub",
                 location: str = "office", want_preview=None) -> None:
        cams = list(cams) if cams else [1, 2, 3, 4, 5, 6, 7]
        digit = "2" if stream == "sub" else "1"
        dcfg = door.load_config()
        self.runners = []
        for n in cams:
            # Only the configured main-door camera gets a line; the rest stay
            # presence-only so a stray crossing elsewhere cannot claim an exit.
            is_door = (dcfg.get("enabled") and n == int(dcfg.get("camera", 5)))
            pts = dcfg.get("line_points") if is_door else None
            self.runners.append(LiveAttendance(
                camera_id=f"CAM-{n}", location=location, mode=mode,
                channel=f"{n}0{digit}", want_preview=want_preview,
                line_points=tuple(pts) if pts else None,
                line_fraction=dcfg.get("line_fraction") if is_door else None,
                out_is_down=(dcfg.get("out_direction", "down") == "down")
                if is_door else None))

    def start(self, on_update=None) -> None:
        # on_update(cam_index, info) — bind index per runner
        for i, r in enumerate(self.runners):
            cb = (lambda info, i=i: on_update(i, info)) if on_update else None
            r.start(on_update=cb)

    def stop(self) -> None:
        for r in self.runners:
            r.stop()

    def is_running(self) -> bool:
        return any(r.is_running() for r in self.runners)

    def all_present(self):
        s: set = set()
        for r in self.runners:
            s |= r.seen_present
        return sorted(s)


class MultiCameraPlayback(MultiCameraLive):
    """Mark attendance for a PAST day from the NVR's own recordings.

    Same fan-out as MultiCameraLive — one thread per camera, shared models,
    presence fused as a set — but every runner reads a /Streaming/tracks/
    playback URL for `day` instead of the live stream, and writes its records
    against `day` rather than today. This is the equivalent of scrubbing the
    Hikvision web UI's Playback tab for each camera, only automated.
    """

    def __init__(self, day: str, cams=None, mode: str = "BALANCED",
                 stream: str = "sub", location: str = "office",
                 start_clock: str = "00:00:00", end_clock: str = "23:59:59",
                 want_preview=None) -> None:
        cams = list(cams) if cams else [1, 2, 3, 4, 5, 6, 7]
        digit = "2" if stream == "sub" else "1"
        cfg = load_config()
        self.day = day
        self.start_clock = start_clock
        self.end_clock = end_clock
        self.runners = []
        for n in cams:
            channel = f"{n}0{digit}"
            self.runners.append(LiveAttendance(
                camera_id=f"CAM-{n}", location=location, mode=mode,
                want_preview=want_preview, day=day, start_clock=start_clock,
                url=build_playback_url(cfg, channel, day,
                                       start_clock, end_clock)))
