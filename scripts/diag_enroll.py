"""Enrollment audit: template counts, per-employee self-consistency,
and cross-employee confusability. Run: python scripts/diag_enroll.py"""
import os, numpy as np
from config import settings

D = os.path.join(settings.DATA_DIR, "face_embeddings")
emb = {}
for fn in sorted(os.listdir(D)):
    if fn.endswith(".npz"):
        emb[fn[:-4]] = np.load(os.path.join(D, fn))["emb"].astype(np.float32)

print(f"{'emp':>4} {'templates':>9} {'self-sim(mean)':>14} {'self-sim(min)':>13}")
print("-" * 46)
for e, m in emb.items():
    if len(m) > 1:
        S = m @ m.T
        iu = np.triu_indices(len(m), 1)
        pair = S[iu]
        print(f"{e:>4} {len(m):>9} {pair.mean():>14.3f} {pair.min():>13.3f}")
    else:
        print(f"{e:>4} {len(m):>9} {'(single tpl)':>14} {'-':>13}")

# Confusability: for employee 01 (Aman), how close is his template centroid
# to every OTHER employee's centroid? High = easy to confuse / weak separation.
print("\nEmployee 01 (Aman) centroid similarity to others:")
if "01" in emb:
    c01 = emb["01"].mean(0); c01 /= np.linalg.norm(c01)
    rows = []
    for e, m in emb.items():
        if e == "01":
            continue
        c = m.mean(0); c /= np.linalg.norm(c)
        rows.append((e, float(c01 @ c)))
    for e, s in sorted(rows, key=lambda x: -x[1]):
        flag = "  <-- confusable" if s > 0.35 else ""
        print(f"   01 vs {e}: {s:+.3f}{flag}")
else:
    print("   NO 01.npz found!")
