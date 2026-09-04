@echo off
REM ============================================================
REM  update.bat — applies THIS extracted update onto your project
REM  WITHOUT touching .venv, data (enrollments/attendance), or
REM  models\weights. Double-click from inside the extracted
REM  attendance_system folder of any update zip.
REM ============================================================
setlocal
set "DEFAULT=C:\Users\themu\OneDrive\Desktop\attendance_system"
set /p TARGET="Project folder [%DEFAULT%]: "
if "%TARGET%"=="" set "TARGET=%DEFAULT%"

if not exist "%TARGET%\requirements.txt" (
    echo [ERROR] %TARGET% does not look like the project folder.
    pause & exit /b 1
)

echo Copying update from "%~dp0" to "%TARGET%" ...
robocopy "%~dp0." "%TARGET%" /E /XD .venv data weights __pycache__ .git /XF update.bat /NFL /NDL /NJH /NJS
if errorlevel 8 (echo [ERROR] copy failed & pause & exit /b 1)

echo.
echo Running tests with the project's own Python ...
if exist "%TARGET%\.venv\Scripts\python.exe" (
    pushd "%TARGET%"
    ".venv\Scripts\python.exe" tests\run_tests.py
    popd
) else (
    echo [WARN] no .venv found in target — run setup_windows.bat there first.
)
echo.
echo Update applied. Start the app normally.
pause
