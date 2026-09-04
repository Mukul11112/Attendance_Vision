"""
tests/run_tests.py — runs tests/test_core_logic.py without pytest installed.
Emulates the two fixtures used (tmp_path, monkeypatch). Prefer plain
`python -m pytest tests -q` when pytest is available.
"""
from __future__ import annotations
import inspect
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)


class _RaisesCtx:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"expected {self.exc.__name__} was not raised")
        return issubclass(et, self.exc)


def _install_pytest_shim():
    import types
    if "pytest" in sys.modules:
        return
    shim = types.ModuleType("pytest")
    shim.raises = _RaisesCtx
    sys.modules["pytest"] = shim


def main() -> int:
    _install_pytest_shim()
    from tests import test_core_logic as mod

    tests = [(n, f) for n, f in vars(mod).items()
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        mp = _MonkeyPatch()
        tmp = Path(tempfile.mkdtemp(prefix="att_test_"))
        kwargs = {}
        params = inspect.signature(fn).parameters
        if "tmp_path" in params:
            kwargs["tmp_path"] = tmp
        if "monkeypatch" in params:
            kwargs["monkeypatch"] = mp
        try:
            fn(**kwargs)
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
        finally:
            mp.undo()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
