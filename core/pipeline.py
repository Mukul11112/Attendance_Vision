"""
core/pipeline.py
The v2 video-processing pipeline. Implements the architecture:

  ingest -> timestamp normalize -> person detect -> track -> face-in-ROI ->
  quality gate -> align+embed -> gallery match -> multi-frame vote ->
  track lock -> reid support -> per-track evidence -> (gate events).

Runs single-threaded on CPU with adaptive frame sampling and FAST/BALANCED/
ACCURATE modes. Emits progress via a callback so the GUI can run it in a worker
thread and stay responsive. Returns TrackEvidence + GateEvent lists that the
attendance engine fuses across videos.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings
from core import face_quality, reid
from core.byte_track import ByteTracker
from core.embedding_gallery import get_gallery
from core.identity_evidence import (Observation, TrackIdentity, update as ev_update,
                                    recompute as ev_recompute, inherit as ev_inherit,
                                    weaken as ev_weaken, support as ev_support,
                                    CONFIRMED, LOCKED)
from core.attendance_engine import TrackEvidence, GateEvent
from core.recognition_scheduler import RecognitionScheduler

log = logging.getLogger("pipeline")


@dataclass
class VideoJob:
    path: str
    date: str                       # YYYY-MM-DD
    start_time: str                 # HH:MM:SS  (wall-clock of first frame)
    camera_id: str
    camera_location: str
    camera_type: str                # entrance/exit/office_room/meeting_room/corridor
    line_fraction: Optional[float] = None    # for gate cameras; None = presence-only
    line_points: Optional[tuple] = None      # (x1,y1,x2,y2) normalized 0-1; an
    #                                          arbitrary segment, used when the
    #                                          gate line is not horizontal (the
    #                                          main door's line is diagonal).
    #                                          Takes precedence over line_fraction.
    out_is_down: Optional[bool] = None       # door camera overrides the global
    #                                          CROSSING_DIRECTION_MEANS_IN, since
    #                                          "which way is outside" is a property
    #                                          of that doorway, not of the system


@dataclass
class Progress:
    video_name: str
    frame_ts: float
    percent: float
    fps: float
    n_tracks: int
    n_confirmed: int
    n_unknown: int          # UNIDENTIFIED tracks active RIGHT NOW (not cumulative)
    eta_s: float
    n_segments: int = 0     # cumulative track fragments this video (diagnostic only)
    n_present: int = 0      # UNIQUE employees identified so far (only ever grows)
    present_ids: tuple = () # employee IDs identified so far (for the live roster)
    preview: Optional[np.ndarray] = None


ProgressCB = Callable[[Progress], None]


def scene_change_fraction(prev_small, cur_small) -> float:
    """Fraction of pixels that changed between two small grayscale frames."""
    if prev_small is None or prev_small.shape != cur_small.shape:
        return 1.0
    return float((cv2.absdiff(prev_small, cur_small) > 18).mean())


def head_region_from_kpts(kpts, person_box, kpt_conf: float):
    """From COCO keypoints, decide facing direction and a tight head crop.
    Returns (facing: bool, head_box or None). Facing = nose + at least one eye
    visible. Head box is built from visible head points, scaled by shoulder
    width, clipped to the person box. Keypoints steer WHERE to look for a
    face — they are never identity evidence."""
    if kpts is None:
        return True, None                     # no pose info: probe as before
    head = kpts[:5]                           # nose, eyes, ears
    vis = head[head[:, 2] >= kpt_conf]
    nose_ok = kpts[0, 2] >= kpt_conf
    eye_ok = kpts[1, 2] >= kpt_conf or kpts[2, 2] >= kpt_conf
    if len(vis) == 0:
        return False, None                    # back turned: skip face work
    nose_ok, eye_ok = bool(nose_ok), bool(eye_ok)
    cx, cy = float(vis[:, 0].mean()), float(vis[:, 1].mean())
    ls, rs = kpts[5], kpts[6]
    if ls[2] >= kpt_conf and rs[2] >= kpt_conf:
        scale = max(abs(ls[0] - rs[0]), 24.0)
    else:
        scale = max((person_box[2] - person_box[0]) * 0.6, 24.0)
    half = scale * 0.9
    x1 = max(person_box[0], cx - half); x2 = min(person_box[2], cx + half)
    y1 = max(person_box[1], cy - half * 1.1); y2 = min(person_box[3], cy + half * 1.1)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return (nose_ok and eye_ok), None
    return bool((nose_ok and eye_ok) or len(vis) >= 2), (x1, y1, x2, y2)


def probe_faces(face_det, frame, box):
    """Face probe for one person ROI. Native detection first; if nothing is
    found and the ROI is small (far person), retry on a 2x super-sampled copy
    so ~20-30px faces become detectable. Returns (face, src_img, upscaled):
    face landmarks/box are in src_img coordinates — recognition must crop,
    quality-check, and embed from src_img, not from the original frame."""
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2 = min(frame.shape[1], int(box[2])); y2 = min(frame.shape[0], int(box[3]))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    faces = face_det.detect(roi, offset=(0, 0))
    src, upscaled = roi, False
    native_tiny = bool(faces) and min(
        faces[0].box[2] - faces[0].box[0],
        faces[0].box[3] - faces[0].box[1]) < settings.FACE_UPSCALE_IF_SMALLER
    if (not faces or native_tiny) and roi.shape[0] < settings.FACE_UPSCALE_ROI_MAX_H:
        s = settings.FACE_UPSCALE_FACTOR
        src = cv2.resize(roi, (int(roi.shape[1] * s), int(roi.shape[0] * s)),
                         interpolation=cv2.INTER_CUBIC)
        up_faces = face_det.detect(src, offset=(0, 0))
        if up_faces:
            faces, upscaled = up_faces, True
        else:
            src = roi
    if not faces:
        return None
    faces.sort(key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]),
               reverse=True)
    return faces[0], src, upscaled



@dataclass
class VideoResult:
    evidences: List[TrackEvidence] = field(default_factory=list)
    gate_events: List[GateEvent] = field(default_factory=list)
    frames_processed: int = 0
    duration_s: float = 0.0


def _mode_cfg(mode: str) -> dict:
    return settings.PROCESSING_MODES.get(mode, settings.PROCESSING_MODES[settings.DEFAULT_PROCESSING_MODE])


def _abs_time(job: VideoJob, seconds_into: float) -> datetime:
    base = datetime.strptime(f"{job.date} {job.start_time}", "%Y-%m-%d %H:%M:%S")
    return base + timedelta(seconds=seconds_into)


class VideoPipeline:
    def __init__(self, mode: str = None, want_preview=None, on_gate=None,
                 on_sample=None) -> None:
        # on_gate(GateEvent) fires the moment someone crosses a gate line,
        # rather than at the end of the run (see the gate-crossing block).
        # on_sample(dict) fires for every recognition attempt — used by the
        # training-image harvester, off by default.
        self.on_gate = on_gate
        self.on_sample = on_sample
        # want_preview: callable polled on every progress tick. When it returns
        # False the annotated preview frame is not built at all — annotating and
        # queueing a full-HD copy per camera is pure waste when the GUI is not
        # drawing it. Recognition is unaffected either way.
        self.want_preview = want_preview if callable(want_preview) else (lambda: True)
        self.mode = mode or settings.DEFAULT_PROCESSING_MODE
        self.cfg = _mode_cfg(self.mode)
        self.gallery = get_gallery()
        # models loaded lazily/once
        from models.face_detector import get_face_detector
        from models.face_embedder import get_embedder
        from models.person_detector_yolo import get_person_detector
        t0 = time.time()
        self.person_det = get_person_detector()
        self.face_det = get_face_detector()
        self.embedder = get_embedder()
        # Phase 2: OSNet body ReID (optional — graceful fallback to histograms)
        from models.body_embedder import get_body_embedder
        from core.body_gallery import get_body_gallery
        self.body = get_body_embedder()          # None if model not installed
        self.body_gallery = get_body_gallery()
        if self.body is None:
            log.warning("OSNet body model not installed — body ReID disabled, "
                        "using color-histogram linking only. "
                        "Run scripts/download_models.py to enable.")
        log.info("Models ready in %.1fs (mode=%s, body_reid=%s)",
                 time.time() - t0, self.mode, "ON" if self.body else "OFF")

    # ── main ──────────────────────────────────────────────────────────────
    def process(self, job: VideoJob, progress_cb: Optional[ProgressCB] = None,
                cancel: Optional[Callable[[], bool]] = None) -> VideoResult:
        cap = None
        if getattr(settings, "VIDEO_HW_DECODE", False):
            # GPU-accelerated decode (NVDEC/QuickSync via FFmpeg). H.264/H.265
            # decoding is bit-exact by spec, so frames are identical to
            # software decode — this only moves the work off the CPU.
            try:
                cap = cv2.VideoCapture(job.path, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_HW_ACCELERATION,
                                        cv2.VIDEO_ACCELERATION_ANY])
                # isOpened() only means the container parsed — a broken HW
                # decoder (e.g. D3D11VA 0x80070057) still opens but then returns
                # NO frames, which silently produced 0-frame results. Verify a
                # real decode before trusting this capture; else fall back to SW.
                if not cap.isOpened() or not cap.grab():
                    cap.release()
                    cap = None
                    log.warning("HW-accelerated decode unavailable for %s — "
                                "falling back to software decode", job.path)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind the test grab
            except Exception:
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(job.path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {job.path}")   # never fail silently
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_dur = total_frames / fps if total_frames else 0.0

        sample_step = max(int(round(self.cfg["sample_interval_s"] * fps)), 1)
        tracker = ByteTracker()
        reid_mem = reid.ReIDMemory()
        sched = RecognitionScheduler()
        self._diag: Dict[int, dict] = {}     # per-track recognition diagnostics
        self._harvested: Dict[str, int] = {}  # body auto-harvest cap per video
        identities: Dict[int, TrackIdentity] = {}
        track_meta: Dict[int, dict] = {}     # track_id -> first/last seen datetimes, boxes
        present_ever: set = set()            # employees seen CONFIRMED/LOCKED at
                                             # least once — attendance semantics:
                                             # once marked present, never unmarked

        result = VideoResult()
        t_start = time.time()
        frame_idx = -1
        sample_idx = 0
        line_y = None
        prev_small = None
        samples_since_detect = 10**9
        probe_cache: Dict[int, tuple] = {}   # tid -> (probe, size, sample_idx)
        kpts_by_tid: Dict[int, object] = {}  # tid -> latest pose keypoints

        while True:
            if cancel and cancel():
                log.info("Pipeline cancelled by caller")
                break
            grabbed = cap.grab()
            if not grabbed:
                break
            frame_idx += 1
            if frame_idx % sample_step != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                log.warning("Failed to decode frame %d in %s", frame_idx, job.path)
                continue

            frame = self._resize(frame)
            H, W = frame.shape[:2]
            if line_y is None:
                if job.line_points is not None:
                    # normalized (x1,y1,x2,y2) -> pixels at this frame size
                    a, b, c, d = job.line_points
                    line_y = (a * W, b * H, c * W, d * H)
                elif job.line_fraction is not None:
                    line_y = int(H * job.line_fraction)
            ts = frame_idx / fps

            # ---- motion gate: skip static/empty scenes entirely ----
            small = cv2.cvtColor(cv2.resize(frame, (160, max(int(160 * H / W), 1))),
                                 cv2.COLOR_BGR2GRAY)
            changed = scene_change_fraction(prev_small, small)
            prev_small = small
            unresolved_active = any(identities[t.track_id].needs_recognition()
                                    for t in tracker.coast()
                                    if t.track_id in identities)
            if (settings.MOTION_SKIP
                    and changed < settings.MOTION_MIN_CHANGED_FRAC
                    and not unresolved_active
                    and samples_since_detect < settings.MOTION_HEARTBEAT_SAMPLES):
                # nothing moved and nobody on screen needs identifying: an
                # overnight/empty-office sample costs almost nothing now
                sample_idx += 1
                samples_since_detect += 1
                result.frames_processed += 1
                continue

            # ---- person detection + tracking (detection every Nth sample;
            #      seated people barely move, so the tracker coasts between) --
            run_detect = (samples_since_detect + 1 >= self.cfg.get("detect_every_n", 1)
                          or changed >= settings.MOTION_MIN_CHANGED_FRAC * 4)
            if run_detect:
                raw = self.person_det.detect(frame)
                dets = [(p.box, p.score) for p in raw]
                tracks = tracker.update(dets)
                samples_since_detect = 0
                # keypoints per track (pose model only): match det -> track by box
                for tr in tracks:
                    for p in raw:
                        if abs(p.box[0] - tr.box[0]) < 3 and abs(p.box[1] - tr.box[1]) < 3:
                            kpts_by_tid[tr.track_id] = p.kpts
                            break
            else:
                tracks = tracker.coast()
                samples_since_detect += 1
            active_ids = {tr.track_id for tr in tracks}
            if self.on_sample is not None:
                # every person box in THIS frame, so the harvester can reject a
                # body crop that has somebody else standing in it
                self._frame_boxes = tuple((tr.track_id, tuple(tr.box))
                                          for tr in tracks)

            # everyone gets an identity record; recognition FOCUS is separate
            new_ids = {tr.track_id for tr in tracks if tr.track_id not in identities}
            for tr in tracks:
                identities.setdefault(tr.track_id, TrackIdentity(track_id=tr.track_id))

            # face probes: focused/new tracks every sample; the rest staggered
            # (their probe result is cached for the scheduler's priorities)
            face_probe: Dict[int, object] = {}
            face_sizes: Dict[int, float] = {}
            for tr in tracks:
                tid = tr.track_id
                due = (tid in sched.focus or tid in new_ids
                       or (sample_idx + tid) % settings.PROBE_EVERY_N == 0)
                if due:
                    facing, head_box = True, None
                    if getattr(settings, "POSE_HEAD_PROBE", True):
                        facing, head_box = head_region_from_kpts(
                            kpts_by_tid.get(tid), tr.box, settings.KPT_CONF)
                    if not facing:
                        probe_cache.pop(tid, None)   # back turned: no face work
                        continue
                    pf = probe_faces(self.face_det, frame, head_box or tr.box)
                    if pf is not None:
                        fb = pf[0].box
                        probe_cache[tid] = (pf, float(min(fb[2] - fb[0], fb[3] - fb[1])),
                                            sample_idx)
                    else:
                        probe_cache.pop(tid, None)
                cached = probe_cache.get(tid)
                if cached and sample_idx - cached[2] <= settings.PROBE_EVERY_N:
                    if cached[2] == sample_idx:      # fresh probe: usable for
                        face_probe[tid] = cached[0]  # recognition this sample
                    face_sizes[tid] = cached[1]

            # one-person-at-a-time identification: the scheduler picks who gets
            # the expensive embedding+matching work; since cost is capped at
            # RECOGNITION_FOCUS_LIMIT person(s), it can run EVERY sample.
            focus_ids = sched.pick(tracks, identities, sample_idx, face_sizes)

            n_conf = 0
            for tr in tracks:
                tid = tr.track_id
                is_new = tid in new_ids
                ident = identities[tid]
                x1, y1, x2, y2 = [int(v) for v in tr.box]
                x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
                meta = track_meta.setdefault(tid, {"first": _abs_time(job, ts),
                                                    "boxes": []})
                meta["last"] = _abs_time(job, ts)
                meta["boxes"].append(((x1 + x2) / 2, (y1 + y2) / 2, ts))

                # ---- appearance descriptor: OSNet body embedding when installed
                # (staggered to bound CPU), else Phase-1 color histogram ------
                desc = None
                body_turn = (self.body is not None
                             and (is_new or (sample_idx + tid)
                                  % settings.BODY_EVERY_N_SAMPLES == 0))
                if body_turn:
                    desc = self.body.embed(frame[y1:y2, x1:x2])
                elif self.body is None:
                    desc = reid.appearance_descriptor(frame[y1:y2, x1:x2])

                # ---- fragment re-linking (appearance = SUPPORT evidence only) ----
                # A brand-new track that strongly resembles a recently LOST track
                # inherits that track's identity evidence, so recognition votes
                # accumulate instead of restarting from zero on every fragment.
                # Safeguards: donor must be inactive, link must clear the strict
                # transfer threshold, and inherited votes are damped so fresh
                # face evidence always outweighs carried-over appearance links.
                if is_new and desc is not None:
                    link_id, link_sim = reid_mem.best_link(desc, exclude=tid)
                    link_thresh = (settings.BODY_LINK_TRANSFER if self.body
                                   else settings.REID_MIN_LINK_FOR_TRANSFER)
                    if (link_id is not None and link_id not in active_ids
                            and settings.REID_MAX_IDENTITY_TRANSFER
                            and link_sim >= link_thresh):
                        donor = identities.get(link_id)
                        if donor is not None and (donor.n_accepted > 0 or donor.votes):
                            ev_inherit(ident, donor)   # head start; can NEVER
                            #                            confirm without fresh
                            #                            face evidence
                            log.info("track %d inherits capped evidence from lost "
                                     "track %d (body sim=%.2f, id=%s)",
                                     tid, link_id, link_sim,
                                     donor.identity or "partial")

                # appearance memory (supporting evidence only)
                if desc is not None:
                    reid_mem.observe(tid, desc, sample_idx)

                # ---- body gallery: SUPPORT votes + auto-harvest ----
                if self.body is not None and desc is not None:
                    if ident.needs_recognition():
                        bm = self.body_gallery.match(desc)
                        if bm.supported:
                            ev_support(ident, bm.employee_id, bm.similarity)
                    elif (settings.BODY_AUTOHARVEST
                          and ident.status == LOCKED and ident.identity
                          and ident.confidence() >= settings.BODY_AUTOHARVEST_MIN_CONF
                          and self._harvested.get(ident.identity, 0)
                              < settings.BODY_AUTOHARVEST_MAX_PER_VIDEO):
                        # a face-proven person donates body templates, so body
                        # ReID keeps working when their face later disappears
                        if self.body_gallery.add_embedding(ident.identity, desc):
                            self._harvested[ident.identity] = \
                                self._harvested.get(ident.identity, 0) + 1

                # ---- face recognition (focused track only; skip once LOCKED) ----
                if tid in focus_ids:
                    was_conf = ident.status in (CONFIRMED, LOCKED)
                    outcome = self._recognize_track(frame, (x1, y1, x2, y2), ident, ts,
                                                    face=face_probe.get(tid))
                    now_conf = ident.status in (CONFIRMED, LOCKED)
                    sched.report(tid, sample_idx,
                                 progressed=(outcome == "attempted"),
                                 resolved=(now_conf or not ident.needs_recognition()))
                    if now_conf and not was_conf:
                        log.info("IDENTIFIED: track %d -> %s (conf=%.2f) — "
                                 "will be marked PRESENT once; moving to next person",
                                 tid, ident.identity, ident.confidence())

                # ---- gate crossing ----
                if line_y is not None and len(meta["boxes"]) >= settings.LINE_MIN_TRACK_AGE:
                    ev = self._check_crossing(meta["boxes"], line_y, ident, job)
                    if ev:
                        result.gate_events.append(ev)
                        if self.on_gate:
                            # Live door alerts cannot wait for the run to end —
                            # on a live camera it never does.
                            try:
                                self.on_gate(ev)
                            except Exception:
                                log.exception("gate callback failed")

                if ident.status in (CONFIRMED, LOCKED):
                    n_conf += 1

            # ---- identity conflict resolver ----
            # The same employee cannot be in two visible places at once. If
            # multiple ACTIVE tracks are confirmed as the same person, keep the
            # most confident and demote the rest — they must re-prove the
            # identity with fresh face evidence.
            by_emp: Dict[str, list] = {}
            for tr in tracks:
                idn = identities[tr.track_id]
                if idn.status in (CONFIRMED, LOCKED) and idn.identity:
                    by_emp.setdefault(idn.identity, []).append(idn)
            for emp, group in by_emp.items():
                if len(group) > 1:
                    group.sort(key=lambda i: i.confidence(), reverse=True)
                    for loser in group[1:]:
                        ev_weaken(loser, emp)
                        n_conf -= 1
                        log.warning("identity conflict: %s claimed by tracks %s "
                                    "simultaneously — kept track %d, demoted track %d",
                                    emp, [g.track_id for g in group],
                                    group[0].track_id, loser.track_id)

            present_ever.update(i.identity for i in identities.values()
                                if i.identity and i.status in (CONFIRMED, LOCKED))

            reid_mem.prune(sample_idx)
            sample_idx += 1
            result.frames_processed += 1

            # ---- progress ----
            if progress_cb and (sample_idx % 4 == 0):
                elapsed = time.time() - t_start
                proc_fps = result.frames_processed / max(elapsed, 1e-6)
                percent = (frame_idx / total_frames * 100) if total_frames else 0.0
                remaining = (total_frames - frame_idx) / sample_step
                eta = remaining / max(proc_fps, 1e-6)
                n_unknown_now = sum(1 for tr in tracks
                                    if identities.get(tr.track_id) is not None
                                    and identities[tr.track_id].identity is None)
                # unique employees identified so far in this video: attendance
                # semantics (a set) — once in, never out, even if the person's
                # track is currently occluded, fragmented, or later demoted by
                # the conflict resolver (present_ever only ever grows)
                progress_cb(Progress(
                    video_name=job.path.split("/")[-1], frame_ts=ts, percent=percent,
                    fps=proc_fps, n_tracks=len(tracks), n_confirmed=n_conf,
                    n_unknown=n_unknown_now, eta_s=eta, n_segments=len(identities),
                    n_present=len(present_ever),
                    present_ids=tuple(sorted(present_ever)),
                    preview=(self._annotate(frame, tracks, identities, line_y,
                                            focus_ids)
                             if self.want_preview() else None),
                ))

        cap.release()
        result.duration_s = time.time() - t_start

        # ---- emit per-track evidence for identified tracks ----
        for tid, ident in identities.items():
            if ident.identity is None:
                continue
            meta = track_meta.get(tid, {})
            first = meta.get("first"); last = meta.get("last")
            if first is None or last is None:
                continue
            result.evidences.append(TrackEvidence(
                employee_id=ident.identity, status=ident.status,
                confidence=ident.confidence(), first_seen=first, last_seen=last,
                camera_id=job.camera_id, camera_location=job.camera_location,
                camera_type=job.camera_type, video_name=job.path.split("/")[-1],
                n_accepted=ident.n_accepted, n_body=ident.n_body,
            ))
        log.info("Processed %s: %d frames, %d tracks, %d identified in %.1fs",
                 job.path, result.frames_processed, len(identities),
                 len(result.evidences), result.duration_s)
        try:
            self._write_review(job, identities)
        except Exception:
            log.exception("failed to write recognition review report")
        return result

    def _write_review(self, job: VideoJob, identities: Dict[int, "TrackIdentity"]) -> None:
        """Save the best face seen for every UNRECOGNIZED track, stamped with
        the exact reason recognition failed — so 'why wasn't X recognized?'
        becomes something you can see instead of guess."""
        import os
        import re
        # job.path may be an RTSP URL, and NVR playback URLs carry a query
        # string ("...tracks/101?starttime=...&endtime=..."). Neither ? nor &
        # is legal in a Windows directory name, so strip the query and any
        # other reserved character before using this as a folder.
        stem = os.path.splitext(os.path.basename(job.path.split("?")[0]))[0]
        stem = re.sub(r'[<>:"/\\|?*]', "_", stem).strip() or "video"
        out_dir = os.path.join(settings.DATA_DIR, "recognition_review", stem)
        os.makedirs(out_dir, exist_ok=True)
        lines, saved = [], 0
        for tid, ident in sorted(identities.items()):
            d = self._diag.get(tid)
            if ident.identity is not None:
                lines.append(f"track {tid}: IDENTIFIED as {ident.identity} "
                             f"({ident.status}, conf={ident.confidence():.2f})")
                continue
            if not d:
                continue
            best = d.get("best")
            if best is None and d.get("no_face", 0) == 0:
                continue
            if best is None:
                verdict = f"face never detected ({d['no_face']} looks)"
            elif best["kind"] == "match":
                if best.get("ambiguous"):
                    verdict = (f"AMBIGUOUS {best['emp']} vs {best.get('second')} "
                               f"(sim {best['sim']:.2f}, margin {best['margin']:.2f})")
                else:
                    verdict = (f"best sim {best['sim']:.2f} -> {best['emp']} "
                               f"(threshold {settings.FACE_SIMILARITY_ACCEPT})")
            else:
                top = sorted(d["lowq"].items(), key=lambda kv: -kv[1])[:2]
                verdict = "rejected: " + ", ".join(f"{r} x{n}" for r, n in top)
            lines.append(f"track {tid}: NOT RECOGNIZED — {verdict}")
            crop = None if best is None else best.get("crop")
            if crop is not None and crop.size > 0 and saved < 60:
                h, w = crop.shape[:2]
                scale = 160.0 / max(h, 1)
                vis = cv2.resize(crop, (max(int(w * scale), 40), 160))
                strip = np.zeros((26, max(vis.shape[1], 340), 3), dtype=np.uint8)
                canvas = np.zeros((186, strip.shape[1], 3), dtype=np.uint8)
                canvas[:160, :vis.shape[1]] = vis
                cv2.putText(canvas, verdict[:60], (4, 178),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 255), 1,
                            cv2.LINE_AA)
                cv2.imwrite(os.path.join(out_dir, f"track{tid:04d}.jpg"), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1
        with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        log.info("Recognition review written -> %s (%d face images)", out_dir, saved)

    # ── steps ─────────────────────────────────────────────────────────────
    def _recognize_track(self, frame, box, ident: TrackIdentity, ts: float,
                         face=None) -> str:
        """Returns 'no_face' | 'low_quality' | 'attempted' for the scheduler.
        `face` is the pre-probed Face for this track (avoids re-detection).
        Also records per-track diagnostics for the recognition review report."""
        diag = self._diag.setdefault(ident.track_id,
                                     {"no_face": 0, "lowq": {}, "best": None})
        if face is None:
            face = probe_faces(self.face_det, frame, box)
            if face is None:
                diag["no_face"] += 1
                return "no_face"
        # probe result: (Face in src coords, src image, upscaled flag). All
        # cropping/quality/embedding happens on src — for far people src is
        # the 2x super-sampled ROI, which is what makes their face usable.
        f, src, upscaled = face
        fx1, fy1, fx2, fy2 = f.box
        face_crop = src[max(0, fy1):fy2, max(0, fx1):fx2]
        blur_min = (settings.QUALITY_BLUR_MIN / (settings.FACE_UPSCALE_FACTOR ** 2)
                    if upscaled else None)
        q = face_quality.assess(face_crop, f.landmarks, f.score, blur_min=blur_min)
        if not q.passed:
            reason = q.reason.split("(")[0].strip() or "low quality"
            diag["lowq"][reason] = diag["lowq"].get(reason, 0) + 1
            if diag["best"] is None or (diag["best"]["kind"] == "lowq"
                                        and q.score > diag["best"]["q"]):
                diag["best"] = {"kind": "lowq", "q": q.score, "reason": q.reason,
                                "crop": face_crop.copy()}
            ev_update(ident, Observation(None, 0.0, q.score, False, False, False, ts))
            return "low_quality"
        emb = self.embedder.embed(src, f.landmarks)
        m = self.gallery.match(emb)
        if (diag["best"] is None or diag["best"]["kind"] == "lowq"
                or m.similarity > diag["best"].get("sim", -1)):
            diag["best"] = {"kind": "match", "sim": m.similarity,
                            "emp": m.employee_id, "margin": m.margin,
                            "accepted": m.accepted, "ambiguous": m.ambiguous,
                            "second": m.second_id, "crop": face_crop.copy()}
        if self.on_sample is not None:
            # Training-image harvest. Everything the purity rules need is
            # already computed here: the crop, its quality, and how far the
            # match is from the runner-up.
            try:
                self.on_sample({
                    "track_id": ident.track_id,
                    "track_identity": ident.identity,
                    "track_status": ident.status,
                    "employee_id": m.employee_id,
                    "similarity": m.similarity,
                    "margin": m.margin,
                    "second_id": m.second_id,
                    "accepted": m.accepted,
                    "ambiguous": m.ambiguous,
                    "quality": q.score,
                    "embedding": emb,
                    "face_crop": face_crop,
                    "person_box": box,
                    "frame": frame,
                    "other_boxes": getattr(self, "_frame_boxes", ()),
                    "frame_ts": ts,
                })
            except Exception:
                log.exception("harvest sample callback failed")
        ev_update(ident, Observation(
            employee_id=m.employee_id if m.accepted else None,
            similarity=m.similarity, quality=q.score, strong=q.strong,
            accepted=m.accepted, ambiguous=m.ambiguous, frame_ts=ts,
        ))
        return "attempted"

    def _check_crossing(self, boxes, line_y, ident: TrackIdentity,
                        job: VideoJob) -> Optional[GateEvent]:
        """Did this track just cross the gate line, and which way?

        `line_y` is either an int (a horizontal line, the original behaviour)
        or a 4-tuple (x1, y1, x2, y2) describing an arbitrary segment — the
        main door's line runs diagonally across the doorway, so a horizontal
        test could not represent it.

        Direction comes from the SIGN of the cross product: which side of the
        line the centroid sits on. Sign flip = crossing.
        """
        if len(boxes) < 2:
            return None
        hy = settings.LINE_HYSTERESIS_PX

        if isinstance(line_y, (tuple, list)):
            x1, y1, x2, y2 = line_y
            dx, dy = x2 - x1, y2 - y1
            norm = (dx * dx + dy * dy) ** 0.5 or 1.0

            def side(b) -> float:
                # signed distance in px from the line (positive = one side)
                px, py = b[0], b[1]
                return ((px - x1) * dy - (py - y1) * dx) / norm

            prev_s, cur_s = side(boxes[-2]), side(boxes[-1])
            crossed_pos = prev_s < -hy and cur_s > hy     # negative -> positive
            crossed_neg = prev_s > hy and cur_s < -hy     # positive -> negative
            if not (crossed_pos or crossed_neg):
                return None

            # The line is a SEGMENT, not an infinite line. Without this test the
            # doorway's line extends across the whole frame, and someone moving
            # about the room crosses its extension — that is how Deepraj, still
            # sitting in the room, produced an exit alert on 31 Jul 2026.
            p0, p1 = boxes[-2], boxes[-1]
            denom = (prev_s - cur_s) or 1e-9
            f = prev_s / denom                            # 0..1 along p0 -> p1
            cx = p0[0] + f * (p1[0] - p0[0])
            cy = p0[1] + f * (p1[1] - p0[1])
            t = ((cx - x1) * dx + (cy - y1) * dy) / (norm * norm)
            if not (-0.05 <= t <= 1.05):                  # 5% slack at each end
                return None

            crossed_down, crossed_up = crossed_pos, crossed_neg
        else:
            prev_y = boxes[-2][1]; cur_y = boxes[-1][1]
            crossed_down = prev_y < line_y - hy and cur_y > line_y + hy
            crossed_up = prev_y > line_y + hy and cur_y < line_y - hy
            if not (crossed_down or crossed_up):
                return None

        if settings.LINE_REQUIRE_IDENTITY and ident.identity is None:
            return None
        if job.out_is_down is not None:
            is_in = crossed_up if job.out_is_down else crossed_down
        else:
            down_means_in = settings.CROSSING_DIRECTION_MEANS_IN == "top_to_bottom"
            is_in = crossed_down if down_means_in else crossed_up
        if ident.identity is None:
            return None
        return GateEvent(
            employee_id=ident.identity, timestamp=_abs_time(job, boxes[-1][2]),
            event_type="IN" if is_in else "OUT", camera_id=job.camera_id,
            video_name=job.path.split("/")[-1], confidence=ident.confidence(),
        )

    def _resize(self, frame):
        maxw = self.cfg["proc_max_width"]
        h, w = frame.shape[:2]
        if w > maxw:
            s = maxw / w
            frame = cv2.resize(frame, (maxw, int(h * s)))
        return frame

    def _annotate(self, frame, tracks, identities, line_y, focus_ids=frozenset()):
        vis = frame.copy()
        if isinstance(line_y, (tuple, list)):
            x1, y1, x2, y2 = [int(v) for v in line_y]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        elif line_y is not None:
            cv2.line(vis, (0, line_y), (vis.shape[1], line_y), (0, 200, 255), 2)
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.box]
            ident = identities.get(tr.track_id)
            confirmed = ident is not None and ident.status in (CONFIRMED, LOCKED)
            if confirmed:
                name, colour = ident.identity, (0, 220, 0)          # green: done
            elif tr.track_id in focus_ids:
                name, colour = "identifying…", (0, 165, 255)        # orange: current focus
            else:
                name, colour = "…", (200, 200, 200)                 # grey: waiting turn
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(vis, f"#{tr.track_id} {name}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return vis


def _parallel_worker(job_kwargs: dict, mode: str, q, threads: int = 0) -> None:
    """Top-level worker (Windows spawn-safe): one video per process."""
    import cv2 as _cv2
    if threads > 0:
        # split CPU threads fairly between workers so N processes don't each
        # spawn a full thread pool and thrash each other
        settings.ORT_INTRA_THREADS = threads
        try:
            _cv2.setNumThreads(threads)
        except Exception:
            pass
    job = VideoJob(**job_kwargs)
    pipe = VideoPipeline(mode=mode)
    last_jpg = [0.0]

    def cb(pr: Progress) -> None:
        m = {k: getattr(pr, k) for k in
             ("video_name", "frame_ts", "percent", "fps", "n_tracks",
              "n_confirmed", "n_unknown", "eta_s", "n_segments", "n_present",
              "present_ids")}
        if pr.preview is not None and pr.frame_ts - last_jpg[0] >= 2.0:
            ok, buf = _cv2.imencode(".jpg", _cv2.resize(pr.preview, (480, 270)),
                                    [int(_cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                m["jpg"] = buf.tobytes()
                last_jpg[0] = pr.frame_ts
        q.put(("progress", m))

    try:
        res = pipe.process(job, progress_cb=cb)
        q.put(("result", job_kwargs["path"], res))
    except Exception as e:  # never hang the parent
        q.put(("error", job_kwargs["path"], str(e)))


def process_batch(jobs: List[VideoJob], mode: str = None,
                  progress_cb: Optional[ProgressCB] = None,
                  cancel: Optional[Callable[[], bool]] = None,
                  workers: int = None) -> Dict[str, VideoResult]:
    """Process several videos. workers > 1 runs videos on parallel processes —
    the key to multi-camera full-day loads (throughput scales ~linearly)."""
    workers = workers or getattr(settings, "PARALLEL_VIDEOS", 1)
    if workers <= 1 or len(jobs) <= 1:
        pipe = VideoPipeline(mode=mode)
        out: Dict[str, VideoResult] = {}
        carried: set = set()   # employees identified in FINISHED videos, so the
        #                        live "marked present" roster never resets when
        #                        the batch moves on to the next video

        def _carry_cb(pr: Progress) -> None:
            ids = carried | set(pr.present_ids)
            pr.present_ids = tuple(sorted(ids))
            pr.n_present = len(ids)
            if progress_cb:
                progress_cb(pr)

        for job in jobs:
            if cancel and cancel():
                break
            res = pipe.process(job, progress_cb=_carry_cb, cancel=cancel)
            out[job.path] = res
            carried |= {ev.employee_id for ev in res.evidences
                        if ev.status in (CONFIRMED, LOCKED)}
        return out

    import dataclasses
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    pending = [dataclasses.asdict(j) for j in jobs]
    running: Dict[str, mp.Process] = {}
    out: Dict[str, VideoResult] = {}
    import os as _os
    pct: Dict[str, float] = {_os.path.basename(j.path): 0.0 for j in jobs}
    # live "marked present" roster is a UNION across all videos in the batch;
    # without this, progress messages from video B would wipe the names video A
    # had already marked (the roster flicker bug)
    present_by_video: Dict[str, set] = {}
    present_done: set = set()            # from videos that already finished
    threads_per_worker = max(2, ((_os.cpu_count() or 8) - 2)
                             // max(min(workers, len(jobs)), 1))

    def _launch() -> None:
        while pending and len(running) < workers:
            jk = pending.pop(0)
            p = ctx.Process(target=_parallel_worker,
                            args=(jk, mode or "", q, threads_per_worker),
                            daemon=True)
            p.start()
            running[jk["path"]] = p

    _launch()
    import queue as _queue
    while running:
        if cancel and cancel():
            for p in running.values():
                p.terminate()
            log.warning("parallel batch cancelled — %d videos incomplete",
                        len(running) + len(pending))
            break
        try:
            msg = q.get(timeout=0.5)
        except _queue.Empty:
            for path, p in list(running.items()):
                if not p.is_alive():          # crashed without a message
                    running.pop(path)
                    _launch()
            continue
        if msg[0] == "progress":
            m = msg[1]
            pct[m["video_name"]] = m.get("percent", pct.get(m["video_name"], 0.0))
            preview = None
            if "jpg" in m:
                arr = np.frombuffer(m.pop("jpg"), dtype=np.uint8)
                preview = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            overall = sum(pct.values()) / max(len(pct), 1)
            present_by_video[m["video_name"]] = set(m.get("present_ids", ()))
            present_union = set(present_done)
            for s in present_by_video.values():
                present_union |= s
            if progress_cb:
                progress_cb(Progress(
                    video_name=f"{m['video_name']}  (batch {overall:.0f}%)",
                    frame_ts=m["frame_ts"], percent=m["percent"], fps=m["fps"],
                    n_tracks=m["n_tracks"], n_confirmed=m["n_confirmed"],
                    n_unknown=m["n_unknown"], eta_s=m["eta_s"],
                    n_segments=m["n_segments"], n_present=len(present_union),
                    present_ids=tuple(sorted(present_union)),
                    preview=preview))
        elif msg[0] == "result":
            _, path, res = msg
            out[path] = res
            pct[_os.path.basename(path)] = 100.0
            present_done.update(ev.employee_id for ev in res.evidences
                                if ev.status in (CONFIRMED, LOCKED))
            p = running.pop(path, None)
            if p:
                p.join(timeout=5)
            _launch()
        elif msg[0] == "error":
            _, path, err = msg
            log.error("parallel worker failed on %s: %s", path, err)
            running.pop(path, None)
            _launch()
    return out
