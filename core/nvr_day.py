"""
core/nvr_day.py
Mark a whole past day from the NVR — by DOWNLOADING its recordings and running
them through the ordinary batch pipeline.

Why download instead of stream: RTSP playback is rate-limited by the NVR to
about 1.1x realtime, so a 24h day would take ~24h per camera. The ISAPI
download endpoint moves the same footage at ~37 MB/s (measured), and a
downloaded file can be SEEKED — which is what lets the batch pipeline sample
every 0.5s instead of decoding every frame. That is the difference between
"all day, all cameras" being impossible and being routine.

Disk is the constraint, not bandwidth: a day is ~18 GB per camera, ~126 GB for
seven. So segments are handled in WAVES — download up to a byte budget, process
that wave with process_batch (the same multiprocess path the Process Videos tab
uses), delete, move on. Peak disk stays at roughly one wave.
"""
from __future__ import annotations
import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional

from config import settings
from core import attendance_engine as ae
from core import nvr
from core.live_camera import load_config
from core.pipeline import VideoJob, process_batch
from database import db_manager, db_setup

log = logging.getLogger("nvr_day")


class NvrDayJob:
    """Download + process one calendar day across several cameras.

    on_status(dict) is called from the worker thread with:
      {phase, camera, done, total, downloaded_mb, present_ids, message}
    """

    def __init__(self, day: str, cams: List[int], mode: str = "BALANCED",
                 cache_dir: Optional[str] = None,
                 wave_bytes: Optional[int] = None,
                 start_clock: str = "00:00:00",
                 end_clock: str = "23:59:59") -> None:
        self.day = day
        self.cams = list(cams)
        self.mode = mode
        self.start_clock = start_clock
        self.end_clock = end_clock
        self.cfg = load_config()
        self.cache_dir = cache_dir or getattr(
            settings, "NVR_CACHE_DIR", os.path.join(settings.DATA_DIR, "nvr_cache"))
        self.wave_bytes = wave_bytes or int(
            getattr(settings, "NVR_WAVE_BUDGET_GB", 12) * (1 << 30))

        self.present: set = set()
        self.error: Optional[str] = None
        self.downloaded_bytes = 0
        self.segments_done = 0
        self.segments_total = 0
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, on_status: Optional[Callable[[dict], None]] = None) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(on_status,),
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── helpers ────────────────────────────────────────────────────────────
    def _in_window(self, seg: "nvr.Segment") -> bool:
        """Keep segments that OVERLAP the window, not just those starting in it.

        Segments run ~80 minutes, so a 09:00-09:30 window is usually served by a
        file that started at 08:4x. Testing only the start time silently dropped
        exactly the footage the user asked for.
        """
        end_clock = seg.end[11:19] if len(seg.end) >= 19 else "23:59:59"
        if end_clock < seg.start_clock:          # segment crosses midnight
            end_clock = "23:59:59"
        return seg.start_clock <= self.end_clock and end_clock >= self.start_clock

    def _emit(self, cb, **kw) -> None:
        if not cb:
            return
        payload = {"phase": "", "done": self.segments_done,
                   "total": self.segments_total,
                   "downloaded_mb": self.downloaded_bytes / (1 << 20),
                   "present_ids": sorted(self.present), "message": ""}
        payload.update(kw)
        try:
            cb(payload)
        except Exception:
            log.exception("status callback failed")

    def _plan(self, cb) -> List["nvr.Segment"]:
        """Ask the NVR what exists for this day, for every requested camera."""
        plan: List["nvr.Segment"] = []
        for cam in self.cams:
            if self._stop:
                break
            track = f"{cam}01"                 # main stream; subs are not recorded
            self._emit(cb, phase="listing", camera=cam,
                       message=f"asking NVR for camera {cam}…")
            try:
                segs = [s for s in nvr.search_segments(self.cfg, track, self.day)
                        if self._in_window(s)]
            except Exception as e:
                log.warning("segment search failed for camera %s: %s", cam, e)
                self._emit(cb, phase="listing", camera=cam,
                           message=f"camera {cam}: {e}")
                continue
            plan.extend(segs)
        plan.sort(key=lambda s: s.start)
        return plan

    # ── worker ─────────────────────────────────────────────────────────────
    def _run(self, on_status) -> None:
        try:
            db_setup.init_db()
            os.makedirs(self.cache_dir, exist_ok=True)
            plan = self._plan(on_status)
            self.segments_total = len(plan)
            if not plan:
                self._emit(on_status, phase="done",
                           message="the NVR reported no recordings in that window")
                return
            gb = sum(s.size for s in plan) / (1 << 30)
            self._emit(on_status, phase="planned",
                       message=(f"{len(plan)} segments across {len(self.cams)} "
                                f"cameras" + (f", ~{gb:.0f} GB" if gb else "")))

            wave: List["nvr.Segment"] = []
            wave_bytes = 0
            for seg in plan:
                if self._stop:
                    break
                wave.append(seg)
                wave_bytes += seg.size or (1 << 30)
                if wave_bytes >= self.wave_bytes:
                    self._do_wave(wave, on_status)
                    wave, wave_bytes = [], 0
            if wave and not self._stop:
                self._do_wave(wave, on_status)

            self._emit(on_status, phase="done",
                       message=("stopped early" if self._stop else
                                f"{self.day}: {len(self.present)} marked present"))
        except Exception as e:
            self.error = str(e)
            log.exception("NVR day job failed")
            self._emit(on_status, phase="error", message=str(e))

    def _do_wave(self, wave: List["nvr.Segment"], cb) -> None:
        """Download this wave, process it, then delete it."""
        jobs: List[VideoJob] = []
        files: List[str] = []
        for seg in wave:
            if self._stop:
                break
            cam = int(seg.track[:-2])
            name = f"{self.day}_cam{cam}_{seg.start_clock.replace(':', '')}.mp4"
            dest = os.path.join(self.cache_dir, name)
            self._emit(cb, phase="downloading", camera=cam,
                       message=f"camera {cam} {seg.start_clock} "
                               f"({seg.minutes:.0f} min)")
            base = self.downloaded_bytes
            try:
                nvr.download(self.cfg, seg, dest, cancel=lambda: self._stop,
                             on_bytes=lambda n, b=base: setattr(
                                 self, "downloaded_bytes", b + n))
            except InterruptedError:
                break
            except Exception as e:
                log.warning("download failed (%s cam %s): %s", seg.start_clock, cam, e)
                self._emit(cb, phase="downloading", camera=cam,
                           message=f"camera {cam} {seg.start_clock}: {e}")
                continue
            files.append(dest)
            jobs.append(VideoJob(
                path=dest, date=self.day, start_time=seg.start_clock,
                camera_id=f"CAM-{cam}", camera_location="office",
                camera_type="office_room", line_fraction=None))

        if not jobs:
            self._cleanup(files)
            return

        self._emit(cb, phase="processing",
                   message=f"processing {len(jobs)} segment(s) — "
                           f"same engine as Process Videos")

        def _prog(pr) -> None:
            if pr.present_ids:
                self.present |= set(pr.present_ids)
            self._emit(cb, phase="processing", camera=None,
                       message=f"{pr.video_name} {pr.percent:.0f}% · "
                               f"{pr.fps:.1f} fps · {len(self.present)} present")

        try:
            results = process_batch(jobs, mode=self.mode, progress_cb=_prog,
                                    cancel=lambda: self._stop)
            evidences, gate_events = [], []
            for res in results.values():
                evidences.extend(res.evidences)
                gate_events.extend(res.gate_events)
            records = ae.fuse_day(self.day, evidences, gate_events)
            for rec in records.values():
                db_manager.save_daily_record(rec)
            self.present |= {eid for eid, r in records.items()
                             if getattr(r, "attendance_status", "") == "PRESENT"}
        except Exception as e:
            log.exception("wave processing failed")
            self._emit(cb, phase="processing", message=f"processing failed: {e}")
        finally:
            self.segments_done += len(jobs)
            self._cleanup(files)

    def _cleanup(self, files: List[str]) -> None:
        """Delete the wave's files — 126 GB/day would not fit otherwise."""
        for f in files:
            for attempt in range(3):            # decoders release lazily
                try:
                    if os.path.exists(f):
                        os.remove(f)
                    break
                except OSError:
                    time.sleep(0.5)
