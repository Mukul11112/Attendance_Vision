"""
Central configuration for the Employee Entry/Exit Attendance System.
Every path, threshold, and tunable value in the whole project is read from here.
Do NOT hardcode paths or thresholds anywhere else in the codebase.
"""
import os

# ── Base paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

TRAINING_IMAGE_DIR = os.path.join(DATA_DIR, "TrainingImage")
TRAINER_PATH = os.path.join(DATA_DIR, "TrainingImageLabel", "trainer.yml")  # single canonical name
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")

ICON_PATH = os.path.join(ASSETS_DIR, "AMS.ico")

# ── Branding ──────────────────────────────────────────────────────────────
APP_TITLE = "Attendance Automation System"

# Ensure runtime directories exist on import
for _d in [DATA_DIR, TRAINING_IMAGE_DIR, os.path.dirname(TRAINER_PATH), EXPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Face capture ──────────────────────────────────────────────────────────

# ── Recognition thresholds ────────────────────────────────────────────────
# LBPH distance: LOWER is a better match (this is NOT a percentage/probability).
# anything >= REVIEW threshold is treated as "Unknown"

# Track-level identity confirmation (Phase 5+): don't trust a single frame.
MIN_OBSERVATIONS_FOR_ID = 5            # min recognition samples collected on a track before deciding identity
MIN_VOTE_RATIO = 0.6                   # winning identity must have >=60% of the votes among observations

# ── Video processing / performance ───────────────────────────────────────

# ── Virtual line / direction config ───────────────────────────────────────
# Line is defined as a horizontal or vertical line in the frame, expressed as a
# FRACTION of frame width/height (0.0-1.0) so it's resolution-independent.
LINE_ORIENTATION = "horizontal"   # "horizontal" or "vertical"
LINE_POSITION_FRACTION = 0.5      # 0.5 = middle of frame

# Direction mapping: which side-crossing direction means IN vs OUT.
# For "horizontal" line: "top_to_bottom" or "bottom_to_top"
# For "vertical" line: "left_to_right" or "right_to_left"
CROSSING_DIRECTION_MEANS_IN = "top_to_bottom"

# Minimum displacement (in pixels) across the line required to count as a real
# crossing, to avoid noise/jitter around the line triggering false events.
MIN_CROSSING_DISPLACEMENT = 15

# ── Tracker config (centroid tracker, Phase 5) ────────────────────────────
MAX_DISAPPEARED_FRAMES = 30      # how many frames a track can go undetected before being dropped
MAX_MATCH_DISTANCE = 120         # max pixel distance to associate a detection with an existing track

# ── Working hours ─────────────────────────────────────────────────────────
# If an employee has an unmatched IN (no OUT by end of video / end of day),
# how it should be reported.

# ── Misc ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "DEBUG"
LOG_FILE = os.path.join(DATA_DIR, "system.log")

# ── Meeting-video attendance pipeline ─────────────────────────────────────
REID_MEMORY_FRAMES = 150

# Registration/video quality controls added for difficult real-camera footage

# ── Fast long-recording attendance mode ──────────────────────────────────
# Analyse one frame every N seconds. 5 s gives 8,640 analysed frames for a
# 12-hour recording instead of 1,080,000 frames at 25 FPS.
# Downscale before face detection. Boxes/crops are used on the scaled image;
# LBPH normalizes the final crop to its training size.
# Temporal confirmation settings.
# GUI work is intentionally infrequent; Tk image conversion is expensive.

# ── Main-gate IN/OUT fast processing ─────────────────────────────────────
# 0.5 s = 86,400 analysed frames for 12 hours, instead of 1.08M at 25 FPS.
# Camera-specific calibration for the supplied main-door view: outside is above,
# office interior is below. Adjust LINE_POSITION_FRACTION after an original CCTV test.
LINE_ORIENTATION = "horizontal"
LINE_POSITION_FRACTION = 0.38
CROSSING_DIRECTION_MEANS_IN = "top_to_bottom"
MIN_CROSSING_DISPLACEMENT = 8
MAX_DISAPPEARED_FRAMES = 12
MAX_MATCH_DISTANCE = 300
MIN_OBSERVATIONS_FOR_ID = 3
MIN_VOTE_RATIO = 0.67

# Hybrid main-gate person/body tracking + face recognition

# Performance: expensive HOG fallback is not needed on every sampled frame.


# ══════════════════════════════════════════════════════════════════════════
# MODERN EMBEDDING PIPELINE (v2)  —  added in the upgrade.
# These drive the ONNX-based person detection / face recognition path.
# The legacy LBPH values above are kept so old code keeps working; the v2
# pipeline (core/pipeline.py) reads ONLY the values in this block.
# ══════════════════════════════════════════════════════════════════════════

# ── Model files (downloaded by scripts/download_models.py into models/weights) ─
MODELS_DIR        = os.path.join(BASE_DIR, "models", "weights")

# ── NVR day-replay scratch space ──────────────────────────────────────────
# A day of full-HD recordings is ~18 GB per camera (~126 GB for seven), so the
# cache must NOT sit on C: — it had 9 GB free on 31 Jul 2026. Segments are
# downloaded, processed and deleted in waves, so only NVR_WAVE_BUDGET_GB is
# ever on disk at once. Falls back to data/nvr_cache if E: is not present.
NVR_CACHE_DIR      = (r"E:\nvr_cache" if os.path.isdir("E:\\")
                      else os.path.join(DATA_DIR, "nvr_cache"))
NVR_WAVE_BUDGET_GB = 12
os.makedirs(MODELS_DIR, exist_ok=True)
YUNET_MODEL       = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
ARCFACE_MODEL     = os.path.join(MODELS_DIR, "arcface_w600k_r50.onnx")   # ArcFace r50, 512-d
YOLO_PERSON_MODEL = os.path.join(MODELS_DIR, "yolov8m.onnx")             # person detector
SCRFD_MODEL       = os.path.join(MODELS_DIR, "scrfd_10g_bnkps.onnx")     # SCRFD-10G + 5 kps
ARCFACE_R100_MODEL = os.path.join(MODELS_DIR, "glintr100.onnx")          # ArcFace r100, 512-d

# ── Face stack selection (19 Aug 2026 upgrade) ────────────────────────────
# Both stacks live in the tree; these flags choose between them. Measured by
# scripts/benchmark_face_models.py on 5,778 track-labelled crops from this
# site's own footage (19 Aug 2026) — rerun it rather than trusting these notes.
#
# SCRFD-10G is the better DETECTOR by a wide margin, but not for the obvious
# reason. It finds FEWER faces than YuNet's 2x-upscale path (26.8% of person
# ROIs vs 45.9%) — yet feeding those crops to the same embedder gives
# EER 4.21% vs 14.99% and genuine similarity 0.588 vs 0.454. YuNet's extra
# detections are mostly bad landmarks, and bad landmarks are what put junk
# templates in the gallery in the first place (see clean_gallery.py).
#
# WHY IT IS STILL OFF: alignment is not interchangeable. Every template in
# data/face_embeddings was aligned from YuNet landmarks; probing those with
# SCRFD-aligned crops costs -0.091 on genuine similarity (0.604 -> 0.513),
# which is larger than the entire margin between ACCEPT (0.45) and STRONG
# (0.50). Flipping this flag WITHOUT rebuilding the gallery would stop people
# being recognised at all. Same applies to FACE_EMBEDDER: a different backbone
# is a different vector space and invalidates every stored template.
#
# Upgrade path, in order: assemble per-employee enrollment images -> rebuild
# the gallery with the new stack -> rerun benchmark_face_models.py -> flip.
FACE_DETECTOR     = "yunet"      # "yunet" (matches current gallery) | "scrfd"
FACE_EMBEDDER     = "r50"        # "r50"   (matches current gallery) | "r100"
SCRFD_INPUT_SIZE  = 320          # letterbox side; 320 covers a person ROI well
MIN_FACE_SIZE_SCRFD = 20         # SCRFD is reliable well below YuNet's 34 px
                                 # floor, which is the point of the upgrade

# ── ONNX Runtime ──────────────────────────────────────────────────────────
def _register_nvidia_dlls():
    """Put pip-installed CUDA/cuDNN DLLs (nvidia-* wheels) on the DLL search
    path. cuDNN 9 loads its sub-libraries (cudnn_engines_*.dll) through PATH,
    so os.add_dll_directory alone is not enough on Windows."""
    import sysconfig
    root = os.path.join(sysconfig.get_paths().get("purelib", ""), "nvidia")
    if not os.path.isdir(root):
        return
    for sub in sorted(os.listdir(root)):
        b = os.path.join(root, sub, "bin")
        if os.path.isdir(b):
            os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(b)
            except (OSError, AttributeError):
                pass


def _pick_ort_providers():
    """Fastest available inference backend, automatically:
    CUDA (NVIDIA GPU) > DirectML (any Windows GPU incl. Intel iGPU) > CPU.
    Enable GPU by installing the matching runtime into the venv:
        pip install onnxruntime-gpu[cuda,cudnn]   (NVIDIA; DLLs come via pip)
        pip install onnxruntime-directml   (any Windows GPU; uninstall
                                            plain onnxruntime first)
    No code changes needed — this picks it up on next launch."""
    _register_nvidia_dlls()
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        # use_tf32=0: full-fp32 convolutions so results match the CPU path
        # bit-for-bit in practice (TF32 would trade a little precision for
        # speed we don't need — the GPU is already >7x faster than CPU here)
        return [("CUDAExecutionProvider", {"use_tf32": "0"}),
                "CPUExecutionProvider"]
    for pref in ("DmlExecutionProvider",
                 "CoreMLExecutionProvider"):   # macOS Apple Silicon
        if pref in avail:
            return [pref, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


ORT_PROVIDERS     = _pick_ort_providers()
ORT_INTRA_THREADS = 8                          # per-process CPU threads for ORT;
                                               # process_batch lowers this per
                                               # worker when running many videos

# ── Person detection (YOLO) ───────────────────────────────────────────────
                                     # TRACKER decides what is real (high band
                                     # starts tracks, low band sustains them)
PERSON_NMS_IOU              = 0.50
PERSON_MIN_BOX_W           = 16     # px — far people are small; keep gates loose
PERSON_MIN_BOX_H           = 32     # px    and let temporal confirmation filter
PERSON_MIN_AREA_FRACTION   = 0.0008 # box area / frame area
PERSON_MIN_ASPECT          = 0.20   # w/h
PERSON_MAX_ASPECT          = 1.60   # seated people behind desks are wide
YOLO_INPUT_SIZE            = 960    # 640 misses far people on large CCTV frames

# ── Face detection (YuNet) ────────────────────────────────────────────────
FACE_DETECTION_CONFIDENCE  = 0.62
FACE_NMS_IOU               = 0.30
MIN_FACE_SIZE              = 34     # px, min(w,h) of the face box inside a person ROI (34 suits wide office cams; quality+similarity gates still protect precision)

# ── Face quality gate (core/face_quality.py) ──────────────────────────────
FACE_QUALITY_MIN           = 0.42   # 0..1 combined score; below -> observation ignored for locking
FACE_QUALITY_STRONG        = 0.62   # >= this counts as a "high quality" observation
QUALITY_BLUR_MIN           = 45.0   # variance-of-Laplacian; below = too blurry
QUALITY_BRIGHT_MIN         = 40     # mean luma
QUALITY_BRIGHT_MAX         = 215
QUALITY_MAX_YAW_DEG        = 50.0   # calibrated 11 Jul: 157 over-shoulder
                                    # rejections at 40; landmark yaw estimate
                                    # is harsh, and votes/margin still gate
QUALITY_MAX_PITCH_DEG      = 34.0

# ── small-face boost (Phase 2d): far-seated people ─────────────────────────
# Person ROIs shorter than this get 2x super-sampled before face detection,
# so ~20-30px faces (too small natively) become detectable and embeddable.
FACE_UPSCALE_ROI_MAX_H     = 260
FACE_UPSCALE_FACTOR        = 2.0

# ── Face recognition / gallery matching ───────────────────────────────────
FACE_EMBED_DIM             = 512
FACE_SIMILARITY_ACCEPT     = 0.45   # borderline band start: may VOTE only.
                                    # 31 Jul 2026: raised from 0.38 after the
                                    # gallery purge (scripts/clean_gallery.py).
                                    # Measured on the CLEANED gallery, the
                                    # closest two DIFFERENT employees sit at
                                    # 0.395 (14 Prakhar vs 22 Sidharth), so
                                    # 0.38 sat below the confusion point and
                                    # let strangers cast votes. Keep this
                                    # above 0.40 and below FACE_SIM_STRONG.
FACE_SIM_STRONG            = 0.50   # confirmation-grade similarity: strong
                                    # votes (needed to CONFIRM) require this.
                                    # Added 11 Jul after a lookalike stranger
                                    # reached confirmation at ~0.4x sims.
REVIEW_MIN_ACCEPTED        = 3      # candidate tracks with >= this many
                                    # accepted votes surface as REVIEW rows
                                    # (never auto-PRESENT)  # calibrated 11 Jul from field data:
                                    # true-match near-miss at 0.42 (emp 01),
                                    # strangers score 0.09-0.14 -> wide gap
AMBIGUITY_MARGIN           = 0.06   # best - second_best must exceed this, else AMBIGUOUS
GALLERY_TOPK               = 3      # average of top-K template sims per employee
EMB_PER_EMPLOYEE_MAX       = 40     # cap stored embeddings/employee (incremental gallery)
ENROLL_DUPLICATE_SIM       = 0.94   # reject near-identical enrollment embeddings above this

# ── Multi-frame voting / track identity locking ───────────────────────────
MIN_IDENTITY_VOTES         = 4      # min accepted face observations before CONFIRMED
MIN_STRONG_VOTES           = 2      # of which at least this many must be high-quality
IDENTITY_VOTE_RATIO        = 0.60   # winner share of votes to CONFIRM
IDENTITY_LOCK_THRESHOLD    = 6      # accepted votes to LOCK (stop re-recognising)
IDENTITY_LOCK_MEAN_SIM     = 0.50   # + mean similarity must be at least this to LOCK
LOCK_OVERRIDE_MARGIN       = 0.12   # contradictory evidence must beat locked id by this to flip

# ── Tracking (ByteTrack, core/byte_track.py) ──────────────────────────────
TRACK_HIGH_THRESH          = 0.50
TRACK_LOW_THRESH           = 0.20

# ── one-person-at-a-time identification (recognition focus scheduler) ──────
                                  # one by one; raise to 2-3 on faster machines)
                                  # scheduler rotates to the next person
                                  # being focused again (they stay tracked)
TRACK_MATCH_IOU            = 0.30
TRACK_MAX_AGE              = 80     # SAMPLES a lost track is kept for re-association (sampled pipeline: 80 samples @0.5s = 40s survival)
TRACK_MIN_HITS             = 3      # detections before a track is confirmed/emitted

# ── Body Re-ID (supporting evidence only, core/reid.py) ───────────────────
REID_MEMORY_FRAMES         = 300
REID_MAX_IDENTITY_TRANSFER = True   # face-confirmed track may lend id to a strongly linked track
REID_MIN_LINK_FOR_TRANSFER = 0.92   # but only above this appearance similarity
                                    # (0.80 was reachable between DIFFERENT
                                    # people; the false-ID incident of 10 Jul)

# ── Phase 2: OSNet body ReID ────────────────────────────────────────────────
# Body evidence SUPPORTS identities and re-links fragments; it can never
# confirm an identity by itself (face remains the only biometric).
# ENABLED 31 Jul 2026. It was disabled after OSNet over-linked different people
# into one identity (body sim 0.78 collapsed everyone into Aman/01, so Shilpi
# never surfaced). Re-enabled together with the STRICT BODY_LINK_TRANSFER 0.90
# that the disable note called for — restoring this path WITHOUT that raise
# reproduces the 10 Jul false-ID incident.
OSNET_MODEL                 = os.path.join(MODELS_DIR, "osnet_x0_25.onnx")

# ── YOLOv8 pose upgrade (optional): if this file exists it is preferred over
#    YOLO_PERSON_MODEL. Pose keypoints steer face recognition (head location,
#    facing direction) — they are NEVER used as identity.
#    DISABLED: the pose skip-gate reduced face recall on far/angled CCTV faces,
#    so we run plain yolov8m.onnx (probes every detected person). To re-enable
#    pose steering, restore the path below to the yolov8m_pose.onnx file.
YOLO_POSE_MODEL             = ""   # was: os.path.join(MODELS_DIR, "yolov8m_pose.onnx")
KPT_CONF                    = 0.30  # keypoint visibility threshold
POSE_HEAD_PROBE             = True  # probe faces on the head crop, not the
                                    # whole person box; skip back-turned people
BODY_EMBED_DIM              = 512
BODY_SIMILARITY_SUPPORT     = 0.62  # body match >= this lends a support vote
BODY_AMBIGUITY_MARGIN       = 0.05
BODY_SUPPORT_VOTE           = 0.25  # fraction of one weak face vote
BODY_SUPPORT_CAP            = 2.0   # max total body vote mass per track+employee
BODY_EVERY_N_SAMPLES        = 4     # body embedding cadence per track (staggered)
BODY_EMB_PER_EMPLOYEE_MAX   = 24
BODY_ENROLL_DUPLICATE_SIM   = 0.985
BODY_AUTOHARVEST            = True  # face-LOCKED tracks donate body templates
BODY_AUTOHARVEST_MIN_CONF   = 0.85
BODY_AUTOHARVEST_MAX_PER_VIDEO = 10
BODY_LINK_TRANSFER          = 0.90  # OSNet fragment-link threshold (replaces
                                    # the 0.92 color-histogram rule when the
                                    # body model is installed). 0.78 was the
                                    # value that collapsed distinct people into
                                    # one identity — do not lower it without
                                    # re-checking that Shilpi/03 still surfaces.

# ── Entry / exit line engine ──────────────────────────────────────────────
LINE_HYSTERESIS_PX         = 12     # dead-zone around the line
LINE_MIN_TRACK_AGE         = 4      # track must have this many centroids before a crossing counts
LINE_REQUIRE_IDENTITY      = False  # if True, only identified tracks generate IN/OUT events

# ── Attendance decision engine (core/attendance_engine.py) ────────────────
PRESENT_MIN_CONFIDENCE     = 0.50   # fused confidence to mark PRESENT (else REVIEW)
REVIEW_MIN_CONFIDENCE      = 0.35   # below PRESENT but above this -> REVIEW; below -> not recorded

# ── Processing modes ──────────────────────────────────────────────────────
# Each mode overrides sampling cadence. core/pipeline.py applies the chosen one.
PROCESSING_MODES = {
    # Modes differ by SAMPLING RATE, not detection quality: shrinking frames
    # (old FAST proc_max_width=640) blinded the detector to far people and
    # caused massive track fragmentation. recognize_every_n is legacy: the
    # focus scheduler caps recognition cost per sample, so it runs every sample.
    "FAST":     {"sample_interval_s": 1.0,  "detect_every_n": 3, "recognize_every_n": 1, "proc_max_width": 960},
    "BALANCED": {"sample_interval_s": 0.5,  "detect_every_n": 2, "recognize_every_n": 1, "proc_max_width": 960},
    "ACCURATE": {"sample_interval_s": 0.25, "detect_every_n": 1, "recognize_every_n": 1, "proc_max_width": 1280},
}
DEFAULT_PROCESSING_MODE = "BALANCED"

# ── Camera types (used in the batch video screen) ─────────────────────────
CAMERA_TYPES = ["entrance", "exit", "office_room", "meeting_room", "corridor"]

# ── Phase 3a: throughput for multi-camera full-day loads ───────────────────
                                   # on mains power; each worker loads models)
PROBE_EVERY_N              = 3     # face-probe cadence for non-focused tracks
MOTION_SKIP                = True  # skip static/empty scenes entirely
MOTION_MIN_CHANGED_FRAC    = 0.004 # fraction of pixels that must change
MOTION_HEARTBEAT_SAMPLES   = 40    # full detection at least every N samples
FACE_UPSCALE_IF_SMALLER    = 40    # px: re-detect on 2x even if a tiny native
                                   # face was found (better far embeddings)

# ── restored 13 Jul: these were referenced only via getattr("NAME") strings,
#    which the 3a dead-settings scan could not see, so it deleted them.
#    Their getattr defaults masked the loss until the pose decode incident.
PERSON_DETECTION_FLOOR     = 0.18
PARALLEL_VIDEOS            = 4     # videos processed concurrently (each worker
                                   # is its own process with its own models;
                                   # GPU is shared). i9-14900K + RTX 4000 Ada
                                   # handles 4-6 comfortably.
VIDEO_HW_DECODE            = False # HW decode (D3D11VA) fails on this machine —
                                   # FFmpeg opens the container but returns no
                                   # frames (0x80070057), so the pipeline read 0
                                   # frames from every video. Software decode
                                   # works fine here. Set True only on a machine
                                   # with a working NVDEC/QuickSync decoder.
RECOGNITION_FOCUS_LIMIT    = 3    # recognize up to 3 people per sample (was 1;
                                  # 1 starved everyone but the first-locked person)
RECOGNITION_FOCUS_PATIENCE = 10
RECOGNITION_FOCUS_COOLDOWN = 24
