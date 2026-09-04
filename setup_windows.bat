@echo off
REM ============================================================
REM  One-time setup: creates .venv, installs requirements,
REM  downloads the three ONNX models. Needs internet ONCE.
REM  Run from the project folder (double-click works).
REM ============================================================
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher 'py' not found. Install Python 3.11 from python.org
    echo         and tick "Add python.exe to PATH" during install.
    pause & exit /b 1
)

py -3.11 -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [WARN] Python 3.11 not found, falling back to default py -3.
    echo        Avoid 3.14 - onnxruntime wheels may be unavailable for it.
    set "PYCMD=py -3"
) else (
    set "PYCMD=py -3.11"
)

if not exist .venv (
    echo Creating virtual environment .venv ...
    %PYCMD% -m venv .venv || (echo [ERROR] venv creation failed & pause & exit /b 1)
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo [ERROR] pip install failed & pause & exit /b 1)

echo.
echo Running logic tests (should say 15 passed) ...
python tests\run_tests.py

echo.
echo Downloading model weights ...
python scripts\download_models.py
echo.
python scripts\download_models.py --status

echo.
echo ============================================================
echo  Setup finished. Start the app with run_app.bat
echo  (or in VS Code: press F5 with "Run Attendance App")
echo ============================================================
pause
