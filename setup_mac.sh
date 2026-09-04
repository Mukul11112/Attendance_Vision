#!/bin/bash
# One-time setup on macOS. Run from the project folder:  bash setup_mac.sh
set -e
python3 --version
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python tests/run_tests.py
./.venv/bin/python scripts/download_models.py || true
./.venv/bin/python scripts/download_models.py --status
echo "Run the app with:  ./.venv/bin/python app.py"
