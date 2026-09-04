# QUICK START (VS Code, Windows)

## 0. Extract to a FRESH folder
Example: `C:\projects\attendance_system` — NOT inside OneDrive (sync can lock
the database), and NOT on top of any older project folder (mixed old/new
files cause wrong-file launches).

If you already ran an earlier version elsewhere, copy over only:
- `data\`            (your registered employees + enrollments + attendance)
- `models\weights\`  (already-downloaded ONNX files, saves ~180 MB download)

## 1. One-time setup
Double-click **`setup_windows.bat`**. It creates `.venv` (Python 3.11),
installs requirements, runs the 15 logic tests, and downloads the 3 model
files. Needs internet once. If the model download is blocked by a proxy,
follow MODEL_SETUP.md (manual browser download).

## 2. Open in VS Code
File → Open Folder → the project folder.
- Bottom-right interpreter should show **.venv (3.11.x)**. If it shows any
  other Python: Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv`.
- Press **F5** → "Run Attendance App"  (or double-click `run_app.bat`,
  or in the terminal: `python app.py` — always the ROOT app.py).

## 3. Use the app (in tab order)
1. **Registration** — add employees (ID, name, department). Delete button is
   top-right above the table.
2. **Enrollment** — select employee → "Enroll from photos" (5–15 varied
   shots) or "Enroll from video" (10–20 s slow head-turn). The log shows
   accepted/rejected with reasons.
3. Sanity check in the terminal, with a photo NOT used for enrollment:
   `python scripts\verify_recognition.py C:\path\to\test.jpg`
   → expect `[PASS] ... WOULD count this as <ID>`.
4. **Process Videos** — add the day's video(s), set date/start time, camera
   type `office_room`, leave the line field blank, mode BALANCED → Start.
5. **Attendance** — load the date, resolve REVIEW rows, **Export Excel**
   → `data\exports\attendance_<date>.xlsx`.

## Common issues
- "Models missing" popup → run `python scripts\download_models.py --status`
  and follow MODEL_SETUP.md for anything MISSING.
- `ModuleNotFoundError` → wrong interpreter; re-select `.venv` (step 2).
- App opens but enroll buttons refuse → models missing (see above).
- Full details of any processing run: `data\system.log`.
