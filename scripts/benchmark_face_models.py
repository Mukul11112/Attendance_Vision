"""
scripts/benchmark_face_models.py
Compare face DETECTORS and EMBEDDERS on this site's own footage.

Why not just trust published benchmarks: LFW/IJB numbers are measured on
portrait-grade faces. This office runs 20-90 px faces on wide CCTV, which is a
different problem, and the whole point of the 19 Aug 2026 upgrade was to find
out whether the newer models actually help HERE.

Ground truth without labelling anything by hand
-----------------------------------------------
Identity labels come from TRACKING, never from a face model:

    genuine pair  = two crops from the SAME ByteTrack track
                    (one track = one person, by temporal continuity)
    impostor pair = crops from DIFFERENT tracks seen in the SAME frame
                    (two people on screen at once are provably not one person)

This matters. The obvious alternative — scoring against data/training_images —
would be rigged: core/harvest.py filters those crops by re-matching them with
ArcFace and deleting whatever ArcFace disagrees with, so any face a challenger
model would have got right has already been thrown away. Track-derived labels
owe nothing to any embedder.

Caveat kept in the open: same-track pairs are close in time, so they share pose
and lighting and are EASIER than a true cross-session match. That inflates the
absolute numbers — equally for every model, so the comparison holds even though
the absolute TAR does not transfer to "will it recognise him tomorrow".
--min-gap reports the same metrics on temporally distant pairs only.

    python scripts/benchmark_face_models.py <bench_dir> [--min-gap 25]

<bench_dir> is produced by the extractor and holds crops/ + meta.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings                                    # noqa: E402

MAX_GENUINE = 60000
MAX_IMPOSTOR = 200000


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_pairs(records, min_gap: int = 0, rng=None, boxes=None):
    """(genuine, impostor) index pairs from track structure alone.

    Two tracks that are visible in the SAME frame are different people, so any
    crop of one may be paired against any crop of the other — not just the two
    crops from that shared frame. That is what makes FAR=1e-3 measurable.

    The one way this can lie: YOLO occasionally detects a single person twice,
    giving two tracks that are really one human. Those pairs would be scored as
    impostors while looking exactly alike, which poisons precisely the
    high-similarity tail the low-FAR metrics are read from. When boxes are
    available, any track pair whose boxes ever overlap (IoU >= DUP_IOU) is
    dropped as a suspected double-detection.
    """
    DUP_IOU = 0.3
    rng = rng or np.random.default_rng(0)
    by_track = defaultdict(list)
    by_frame = defaultdict(list)
    for i, r in enumerate(records):
        by_track[r["track"]].append(i)
        by_frame[r["frame"]].append(i)

    genuine = []
    for idxs in by_track.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if abs(records[ia]["frame"] - records[ib]["frame"]) >= min_gap:
                    genuine.append((ia, ib))

    # which track pairs are provably different people
    distinct, suspect = set(), set()
    for frame, idxs in by_frame.items():
        tracks = {records[i]["track"] for i in idxs}
        for ta in tracks:
            for tb in tracks:
                if ta >= tb:
                    continue
                key = (ta, tb)
                if boxes is not None:
                    ba = boxes.get(f"{ta}_{frame}"); bb = boxes.get(f"{tb}_{frame}")
                    if ba and bb and _iou(ba, bb) >= DUP_IOU:
                        suspect.add(key); continue
                distinct.add(key)
    distinct -= suspect
    if boxes is not None:
        print(f"  track pairs: {len(distinct)} distinct, "
              f"{len(suspect)} dropped as suspected double-detections")

    impostor = []
    for ta, tb in distinct:
        for ia in by_track[ta]:
            for ib in by_track[tb]:
                impostor.append((ia, ib))

    if len(genuine) > MAX_GENUINE:
        genuine = [genuine[i] for i in rng.choice(len(genuine), MAX_GENUINE, replace=False)]
    if len(impostor) > MAX_IMPOSTOR:
        impostor = [impostor[i] for i in rng.choice(len(impostor), MAX_IMPOSTOR, replace=False)]
    return genuine, impostor


def embed_all(model_path: str, crops: np.ndarray, batch: int = 64) -> np.ndarray:
    """L2-normalized embeddings for every crop. All three models share the
    ArcFace convention (RGB, [-1,1], 112x112) — verified against the CVLFace
    README for AdaFace — so preprocessing is not a confound here."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = settings.ORT_INTRA_THREADS
    sess = ort.InferenceSession(model_path, sess_options=so,
                                providers=settings.ORT_PROVIDERS)
    name = sess.get_inputs()[0].name
    out = []
    t0 = time.time()
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch].astype(np.float32)
        chunk = (chunk - 127.5) / 127.5
        blob = chunk.transpose(0, 3, 1, 2)
        out.append(sess.run(None, {name: blob})[0])
    E = np.concatenate(out).astype(np.float32)
    E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    return E, (time.time() - t0) / max(len(crops), 1) * 1000.0


def metrics(gen: np.ndarray, imp: np.ndarray) -> dict:
    """ROC-AUC, EER and TAR at fixed FAR — computed from the score arrays
    directly so there is no sklearn dependency."""
    lo = min(gen.min(), imp.min()); hi = max(gen.max(), imp.max())
    ths = np.linspace(lo, hi, 2000)
    tar = np.array([(gen >= t).mean() for t in ths])
    far = np.array([(imp >= t).mean() for t in ths])
    order = np.argsort(far)
    auc = float(np.trapezoid(tar[order], far[order])) if hasattr(np, "trapezoid") \
        else float(np.trapz(tar[order], far[order]))
    eer_i = int(np.argmin(np.abs(far - (1 - tar))))
    out = {"auc": abs(auc), "eer": float((far[eer_i] + (1 - tar[eer_i])) / 2),
           "gen_mean": float(gen.mean()), "gen_std": float(gen.std()),
           "imp_mean": float(imp.mean()), "imp_std": float(imp.std()),
           "d_prime": float((gen.mean() - imp.mean()) /
                            np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))}
    for target in (1e-2, 1e-3, 1e-4):
        ok = np.where(far <= target)[0]
        out[f"tar@far{target:g}"] = float(tar[ok].max()) if len(ok) else 0.0
        out[f"thr@far{target:g}"] = float(ths[ok].min()) if len(ok) else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--min-gap", type=int, default=0,
                    help="only pair crops at least this many sampled frames apart")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.bench_dir, "meta.json")))
    records = meta["records"]
    print(f"{len(records)} crops from {meta['frames']} frames of "
          f"{os.path.basename(meta['video'])}")

    crops = np.stack([
        cv2.cvtColor(cv2.imread(os.path.join(args.bench_dir, "crops", r["file"])),
                     cv2.COLOR_BGR2RGB) for r in records])
    print(f"loaded crops {crops.shape}")

    bpath = os.path.join(args.bench_dir, "boxes.json")
    boxes = json.load(open(bpath)) if os.path.isfile(bpath) else None
    genuine, impostor = build_pairs(records, args.min_gap, boxes=boxes)
    print(f"pairs: {len(genuine)} genuine / {len(impostor)} impostor "
          f"(min_gap={args.min_gap})")
    if not genuine or not impostor:
        print("not enough pairs — extract more footage"); return 1

    gi = np.array(genuine); ii = np.array(impostor)
    models = [("ArcFace r50 (current)", settings.ARCFACE_MODEL),
              ("ArcFace r100 (glint360k)", settings.ARCFACE_R100_MODEL),
              ("AdaFace IR-101 (WebFace12M)",
               os.path.join(settings.MODELS_DIR, "adaface_ir101_webface12m.onnx"))]

    results = {"meta": {"crops": len(records), "genuine": len(genuine),
                        "impostor": len(impostor), "min_gap": args.min_gap,
                        "video": os.path.basename(meta["video"]),
                        "frames": meta["frames"]},
               "detectors": meta.get("det", {}), "embedders": {}}

    for label, path in models:
        if not os.path.isfile(path):
            print(f"\n{label}: SKIPPED (missing {path})"); continue
        E, ms = embed_all(path, crops)
        g = np.einsum("ij,ij->i", E[gi[:, 0]], E[gi[:, 1]])
        m = np.einsum("ij,ij->i", E[ii[:, 0]], E[ii[:, 1]])
        r = metrics(g, m); r["ms_per_face"] = ms; r["dim"] = int(E.shape[1])
        results["embedders"][label] = r
        print(f"\n{label}   ({ms:.1f} ms/face)")
        print(f"  genuine  {r['gen_mean']:.3f} +/- {r['gen_std']:.3f}")
        print(f"  impostor {r['imp_mean']:.3f} +/- {r['imp_std']:.3f}   d'={r['d_prime']:.2f}")
        print(f"  AUC {r['auc']:.4f}   EER {r['eer']*100:.2f}%   "
              f"TAR@FAR1e-2 {r['tar@far0.01']:.3f}   TAR@FAR1e-3 {r['tar@far0.001']:.3f}")

    out = args.out or os.path.join(args.bench_dir, f"results_gap{args.min_gap}.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
