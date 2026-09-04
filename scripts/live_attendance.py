"""Run continuous live attendance from the console (test / headless use).

  python scripts/live_attendance.py                 # run until Ctrl+C
  python scripts/live_attendance.py --minutes 2     # stop after 2 minutes
  python scripts/live_attendance.py --channel 102   # use the lighter sub-stream
"""
from __future__ import annotations
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.live_camera import LiveAttendance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="0 = run until Ctrl+C")
    ap.add_argument("--channel", default=None, help="101 main / 102 sub")
    ap.add_argument("--camera-id", default="LIVE-01")
    ap.add_argument("--location", default="office")
    args = ap.parse_args()

    live = LiveAttendance(camera_id=args.camera_id, location=args.location,
                          channel=args.channel)
    print("Live camera URL:", live.redact_url() if hasattr(live, "redact_url")
          else "(configured)")

    last = {"line": ""}

    def on_update(info: dict) -> None:
        line = (f"[{int(info['elapsed_s']):>5}s] "
                f"tracks={info['n_tracks']:>2} "
                f"proc_fps={info['fps']:.1f} "
                f"PRESENT({len(info['present_ids'])}): {info['present_ids']}")
        if line != last["line"]:
            print(line, flush=True)
            last["line"] = line

    live.start(on_update=on_update)
    limit = args.minutes * 60 if args.minutes > 0 else None
    print("Running… Ctrl+C to stop." if not limit
          else f"Running for {args.minutes} min…")
    try:
        while live.is_running():
            if limit and (time.time() - live._t0) > limit and live._t0 > 0:
                live.stop()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping…")
        live.stop()

    # let the worker finish its final DB fuse
    for _ in range(10):
        if not live.is_running():
            break
        time.sleep(1)
    if live.error:
        print("ERROR:", live.error)
    print("Final PRESENT:", sorted(live.seen_present))


if __name__ == "__main__":
    main()
