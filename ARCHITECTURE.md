# Architecture (v3)

Video -> sampled frames -> [motion gate] -> YOLOv8n person detection (every
Nth sample; ByteTrack coasts between) -> ByteTrack with center-distance rescue
-> per-track face probe (YuNet; staggered; 2x super-sample for small ROIs) ->
recognition focus scheduler (one person at a time, face-priority, preemption)
-> ArcFace embedding + gallery match (top-K + margin) -> identity evidence
state machine (UNKNOWN -> CANDIDATE -> CONFIRMED -> LOCKED) -> attendance
fusion (set-based, one record per employee per date) -> SQLite UPSERT with
UNIQUE(employee_id, attendance_date) -> Excel export.

Body ReID (OSNet, optional): support votes + fragment re-linking + auto-harvest
from face-LOCKED tracks. Body evidence is capped and can NEVER confirm.

Identity trust bands (calibrated from field data, 11 Jul):
  < 0.38          stranger territory (observed strangers: 0.09-0.14)
  0.38 - 0.50     borderline: may VOTE; sustained -> REVIEW row, never PRESENT
  >= 0.50         confirmation-grade: only these produce STRONG votes;
                  confirmation requires 4 accepted votes incl. 2 strong

Throughput levers (Phase 3a):
  motion gate      static/empty scenes cost ~zero (overnight footage)
  detect cadence   BALANCED detects every 2nd sample, FAST every 3rd
  probe stagger    non-focused tracks probed every 3rd sample (cached)
  parallel videos  N worker processes, one video each (Workers spinbox)

Multi-camera full-day math (i5-1335U):
  per-worker ~3-4x realtime in FAST after gating; x3 workers ~10-12x realtime;
  24h x 12 cameras with ~60% empty/static time -> processable overnight.
