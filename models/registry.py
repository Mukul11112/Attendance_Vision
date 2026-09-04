"""
models/registry.py
Single source of truth for which model weight files the v2 pipeline needs.

The GUI and scripts call missing_required()/status_report() before loading
anything, so the user gets one clear message instead of a stack trace.
"""
from __future__ import annotations
import os
from config import settings

# (friendly name, absolute path, required?)
MODEL_FILES = [
    ("YuNet face detector",      settings.YUNET_MODEL,       True),
    ("ArcFace face embedder",    settings.ARCFACE_MODEL,     True),
    ("YOLOv8m person detector",  settings.YOLO_PERSON_MODEL, True),
    ("OSNet body ReID (Phase 2)", settings.OSNET_MODEL,      False),
    ("SCRFD face detector (alt)", settings.SCRFD_MODEL,      False),
]


def missing_required() -> list:
    """Names of required model files that are not on disk."""
    return [name for name, path, req in MODEL_FILES if req and not os.path.isfile(path)]


def status_report() -> str:
    lines = []
    for name, path, req in MODEL_FILES:
        ok = os.path.isfile(path)
        tag = "OK      " if ok else ("MISSING " if req else "optional")
        lines.append(f"[{tag}] {name}\n          {path}")
    return "\n".join(lines)
