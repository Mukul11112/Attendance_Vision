"""
scripts/clean_gallery.py
Quarantine face templates that do not belong to the employee they are filed
under.

Why this exists: on 31 Jul 2026 an unregistered visitor was being confirmed as
employee 10. The cause was not the matcher — it was the gallery. Most employee
folders held embeddings of MORE THAN ONE person (employee 10's own templates
were anti-correlated with each other, min sim -0.059), so a stranger only had
to resemble whichever face had been mixed in.

Method: for each employee take the MEDOID (the template most similar to all the
others = the dominant face), keep templates within KEEP_SIM of it, and move the
rest to data/face_embeddings_quarantine/. The dominant cluster is a judgement,
not a certainty, so nothing is deleted and the whole directory is backed up
first.

    python scripts/clean_gallery.py            # dry run, prints what it would do
    python scripts/clean_gallery.py --apply    # back up, then quarantine
"""
from __future__ import annotations
import os
import shutil
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings                                    # noqa: E402

EMB_DIR = os.path.join(settings.DATA_DIR, "face_embeddings")
QUAR_DIR = os.path.join(settings.DATA_DIR, "face_embeddings_quarantine")

KEEP_SIM = 0.40      # template must be this close to the dominant face
MIN_KEEP = 3         # never reduce an employee below this many templates


def _load() -> dict:
    out = {}
    for fn in sorted(os.listdir(EMB_DIR)):
        if fn.endswith(".npz"):
            a = np.load(os.path.join(EMB_DIR, fn))["emb"].astype(np.float32)
            a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
            out[fn[:-4]] = a
    return out


def _names() -> dict:
    try:
        from database import db_manager
        return {str(e["employee_id"]): (e.get("name") or "")
                for e in db_manager.get_employees()}
    except Exception:
        return {}


def plan(A: np.ndarray):
    """-> (keep_idx, drop_idx). Keeps the dominant cluster around the medoid."""
    n = len(A)
    if n <= MIN_KEEP:
        return list(range(n)), []
    S = A @ A.T
    medoid = int(np.argmax(S.sum(axis=1)))
    sims = S[medoid]
    keep = [i for i in range(n) if sims[i] >= KEEP_SIM]
    if len(keep) < MIN_KEEP:                    # too aggressive — take the best
        keep = list(np.argsort(-sims)[:MIN_KEEP])
    keep = sorted(set(keep))
    return keep, [i for i in range(n) if i not in set(keep)]


def main() -> None:
    apply = "--apply" in sys.argv
    mats = _load()
    names = _names()

    print(f"{'ID':<5}{'NAME':<20}{'n':>4}{'keep':>6}{'drop':>6}   "
          f"{'self-med before':>16}{'after':>8}")
    total_drop = 0
    results = {}
    for eid, A in sorted(mats.items(), key=lambda kv: (len(kv[0]), kv[0])):
        keep, drop = plan(A)
        results[eid] = (keep, drop)
        total_drop += len(drop)

        def med(M):
            if len(M) < 2:
                return float("nan")
            S = M @ M.T
            iu = np.triu_indices(len(M), 1)
            return float(np.median(S[iu]))

        print(f"{eid:<5}{names.get(eid, '')[:18]:<20}{len(A):>4}{len(keep):>6}"
              f"{len(drop):>6}   {med(A):>16.3f}{med(A[keep]):>8.3f}")

    print(f"\ntotal templates quarantined: {total_drop}")

    # cross-employee closeness after the purge — this is what sets the threshold
    cleaned = {e: mats[e][results[e][0]] for e in mats}
    worst = 0.0
    pair = ("", "")
    for a in cleaned:
        for b in cleaned:
            if a >= b:
                continue
            m = float((cleaned[a] @ cleaned[b].T).max())
            if m > worst:
                worst, pair = m, (a, b)
    print(f"closest DIFFERENT employees after purge: {worst:.3f} "
          f"({pair[0]} vs {pair[1]})")
    print(f"  -> any accept threshold at or below {worst:.2f} can still confuse them")

    if not apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(settings.DATA_DIR, f"face_embeddings_backup_{stamp}")
    shutil.copytree(EMB_DIR, backup)
    os.makedirs(QUAR_DIR, exist_ok=True)
    print(f"\nbacked up -> {backup}")

    for eid, (keep, drop) in results.items():
        if not drop:
            continue
        A = mats[eid]
        np.savez_compressed(os.path.join(QUAR_DIR, f"{eid}.npz"), emb=A[drop])
        np.savez_compressed(os.path.join(EMB_DIR, f"{eid}.npz"),
                            emb=A[keep].astype(np.float32))
    print(f"quarantined templates -> {QUAR_DIR}")
    print("done — restart the app so the gallery reloads.")


if __name__ == "__main__":
    main()
