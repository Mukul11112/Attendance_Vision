# Troubleshooting

UPDATE RULE: never delete/replace the project folder. Extract update zips to
Downloads, double-click update.bat inside. It protects .venv, data, weights.

.venv not recognized        -> py -3.11 -m venv .venv ; pip install -r requirements.txt
tests count wrong           -> update zip not applied; re-run update.bat
Models MISSING              -> scripts/download_models.py (+ pip install ultralytics
                               for YOLO export; + torchreid gdown for OSNet)
Gallery: empty              -> data/ was wiped; re-enroll (keep backups:
                               xcopy /E /I data\face_embeddings C:\attendance_backup\face_embeddings)
Someone not recognized      -> tab 5 "Why not recognized?": yellow line = the
                               diagnosis. best sim 0.38-0.49 -> add CCTV crops
                               of them to enrollment (raises true sims > 0.50);
                               "too small"/"never detected" -> ACCURATE mode or
                               camera placement; they appear as REVIEW when
                               matches are sustained.
Wrong person marked         -> must never happen (band design). If it does:
                               run scripts/evaluate.py and send the output.
Slow processing             -> Workers spinbox 2-3, FAST mode for bulk days,
                               plugged in, Best Performance, OneDrive paused.
PowerShell path errors      -> always prefix: .\.venv\Scripts\python.exe
