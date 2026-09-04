# Office Video Attendance System (v2)

Register employees, upload one or more office videos for a date, and get a
single Present/Review/Absent row per employee, exported to Excel.

Identity is **track-level, not frame-level**: people are detected once, tracked
with persistent IDs, and recognized from evidence accumulated over many frames
and views. A confirmed identity is locked to its track; recognition then stops
on that track. Each confirmed employee enters the day's present set exactly
once, regardless of how many frames, tracks, or videos they appear in.

## Install (Windows, CPU)

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_models.py      # needs internet once; see MODEL_SETUP.md
python scripts\download_models.py --status
```

## Run

```
python app.py
```

1. **Registration** — add Employee ID, name, department, designation.
2. **Enrollment** — per employee, add 5–15 varied face photos (front, slight
   left/right, slight up/down, office lighting) *or* a 10–20 s video of them
   slowly turning their head. Blurry/dark/extreme-pose samples and
   near-duplicate frames are rejected automatically (diversity sampling).
3. **Process Videos** — add all videos for the date, set date/start time and
   camera info, pick FAST / BALANCED / ACCURATE, Start. The GUI never freezes;
   processing runs in a worker thread and streams progress (current video, %,
   FPS, active tracks, confirmed, unknown, ETA, preview).
4. **Attendance** — load the date, resolve any REVIEW rows manually, export
   Excel (`data/exports/attendance_<date>.xlsx`).

Sanity check after enrollment, before processing videos:

```
python scripts\verify_recognition.py path\to\test_photo.jpg
```

## How duplicate attendance is impossible

Three independent layers:

1. **Track lock** (`core/identity_evidence.py`) — a track needs
   `MIN_IDENTITY_VOTES` accepted face observations, a 60% winner vote share,
   and ≥2 high-quality observations before CONFIRMED; more evidence LOCKS the
   identity and recognition stops on that track. One weak match never names a
   person.
2. **Set-based fusion** (`core/attendance_engine.py::fuse_day`) — all track
   evidence from all videos of the date is merged into a dict keyed by
   employee_id: one record per employee, mathematically. Extra tracks/videos
   strengthen confidence; they cannot duplicate.
3. **Database constraint** (`database/db_setup.py`) —
   `UNIQUE(employee_id, attendance_date)` with UPSERT writes. Even re-running
   the whole day updates the one existing row.

## Face + body evidence fusion (current state)

Face embeddings (ArcFace, multi-template per employee, top-K + margin
matching, temporal voting) are the biometric evidence. Body appearance is
**support-only**: it re-links fragmented tracks and lends *damped* evidence
from a lost confirmed track to its continuation — it can never confirm an
employee alone. Phase 2 upgrades the appearance descriptor to OSNet ReID
embeddings with a per-employee body gallery behind the same interface.

## Performance on i5-1335U

Set `ORT_INTRA_THREADS = 4` (default) in `config/settings.py`.
FAST samples 1 frame/s at 640 px — use for long recordings. BALANCED
(default) samples 2 frames/s at 960 px. ACCURATE samples 4 frames/s at
1280 px — use for short or difficult footage. Recognition runs only on
sampled frames, only inside person tracks, only until a track locks.

## Known limitations

- Recognition needs the face to become reasonably visible at some point on a
  track; a person who never faces any camera can only be REVIEW/ABSENT.
- Color-histogram appearance (Phase 1) can be confused by similar clothing;
  that is exactly why it never confirms identity by itself.
- Very distant faces (< ~34 px) are ignored by design rather than guessed.
- Accuracy claims must come from your own validation footage — see the
  evaluation phase; no fixed accuracy number is promised.

## Tests

```
python -m pytest tests -q        # or, without pytest:
python tests\run_tests.py
```

Covers: track persistence and occlusion recovery, no-single-frame-identity,
confirm/lock behavior, conflicting-evidence → review, multi-video present-set
merging, roster ABSENT view, DB uniqueness under repeated saves, raw duplicate
INSERT rejection, and one-row-per-employee Excel export.
