"""Headless test: run all NVR cameras at once for a short window."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.live_camera import MultiCameraLive

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 90
stream = sys.argv[2] if len(sys.argv) > 2 else "sub"

multi = MultiCameraLive(cams=[1, 2, 3, 4, 5, 6, 7], mode="BALANCED", stream=stream)
last = {}

def on_update(i, info):
    cam = i + 1
    key = (cam, info.get("n_tracks"), len(info.get("present_ids", [])))
    if last.get(cam) != key:
        last[cam] = key
        print(f"cam{cam}: tracks={info.get('n_tracks')} "
              f"fps={info.get('fps',0):.1f} present={info.get('present_ids')}", flush=True)

print(f"Starting 7 cameras ({stream} stream) for {SECONDS:.0f}s…")
multi.start(on_update=on_update)
t0 = time.time()
while time.time() - t0 < SECONDS:
    time.sleep(2)
multi.stop()
for _ in range(15):
    if not multi.is_running():
        break
    time.sleep(1)
print("\nALL-CAMERA present today:", multi.all_present())
errs = [(i + 1, r.error) for i, r in enumerate(multi.runners) if r.error]
if errs:
    print("errors:", errs)
