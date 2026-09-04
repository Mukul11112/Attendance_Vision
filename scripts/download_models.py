"""
scripts/download_models.py
Downloads the three required ONNX models into models/weights/.
Run ONCE on a machine with internet access:

    python scripts/download_models.py

If a corporate proxy blocks a URL, download the file in a browser and place
it at the exact path printed by --status (see MODEL_SETUP.md for mirrors).
"""
from __future__ import annotations
import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings  # noqa: E402

DOWNLOADS = [
    # (target path, primary URL)
    (settings.YUNET_MODEL,
     "https://github.com/opencv/opencv_zoo/raw/main/models/"
     "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    (settings.ARCFACE_MODEL,
     # InsightFace buffalo_l recognition model (w600k_r50), ONNX, 512-d
     "https://github.com/deepinsight/insightface/releases/download/v0.7/"
     "buffalo_l.zip"),
    (settings.YOLO_PERSON_MODEL,
     # Exported YOLOv8n; alternatively export locally with ultralytics:
     #   from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=12)
     "LOCAL_EXPORT"),
]


def _fetch(url: str, dst: str) -> None:
    print(f"Downloading {url}\n        -> {dst}")
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    print(f"        done ({os.path.getsize(dst)/1e6:.1f} MB)")


def main() -> int:
    if "--status" in sys.argv:
        from models.registry import status_report
        print(status_report())
        return 0

    os.makedirs(settings.MODELS_DIR, exist_ok=True)

    # 1) YuNet — direct download
    if not os.path.isfile(settings.YUNET_MODEL):
        _fetch(DOWNLOADS[0][1], settings.YUNET_MODEL)
    else:
        print("YuNet already present.")

    # 2) ArcFace — comes inside insightface's buffalo_l.zip
    if not os.path.isfile(settings.ARCFACE_MODEL):
        import io, zipfile
        url = DOWNLOADS[1][1]
        print(f"Downloading {url} (contains w600k_r50.onnx)…")
        data = urllib.request.urlopen(url).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(n for n in z.namelist() if n.endswith("w600k_r50.onnx"))
            with z.open(member) as src, open(settings.ARCFACE_MODEL, "wb") as out:
                out.write(src.read())
        print(f"Extracted ArcFace -> {settings.ARCFACE_MODEL}")
    else:
        print("ArcFace already present.")

    # 3) YOLOv8n — export locally via ultralytics (most reliable path)
    if not os.path.isfile(settings.YOLO_PERSON_MODEL):
        try:
            from ultralytics import YOLO
            print("Exporting yolov8n.pt -> ONNX via ultralytics…")
            m = YOLO("yolov8n.pt")            # auto-downloads the .pt
            out = m.export(format="onnx", opset=12, imgsz=settings.YOLO_INPUT_SIZE)
            os.replace(out, settings.YOLO_PERSON_MODEL)
            print(f"Exported -> {settings.YOLO_PERSON_MODEL}")
        except ImportError:
            print("\n[ACTION NEEDED] ultralytics is not installed. Either:\n"
                  "  pip install ultralytics    (one-time, only for the export)\n"
                  "then re-run this script, or place a yolov8n.onnx manually at:\n"
                  f"  {settings.YOLO_PERSON_MODEL}")
            return 1
    else:
        print("YOLOv8n already present.")

    # 4) OSNet body ReID (Phase 2, optional) — export via torchreid
    if not os.path.isfile(settings.OSNET_MODEL):
        try:
            import torch, torchreid
            print("Exporting OSNet x0_25 -> ONNX via torchreid…")
            model = torchreid.models.build_model("osnet_x0_25", num_classes=1,
                                                 pretrained=True)
            model.eval()
            dummy = torch.randn(1, 3, 256, 128)
            torch.onnx.export(model, dummy, settings.OSNET_MODEL, opset_version=11,
                              input_names=["input"], output_names=["output"])
            print(f"Exported -> {settings.OSNET_MODEL}")
        except ImportError:
            print("\n[OPTIONAL] Body ReID model not installed. To enable it:\n"
                  "  pip install torchreid gdown\n"
                  "then re-run this script. The system works without it\n"
                  "(face-only), but body ReID greatly improves continuity.")
        except Exception as e:
            print(f"[WARN] OSNet export failed: {e} — body ReID stays disabled.")

    from models.registry import missing_required, status_report
    print("\n" + status_report())
    return 1 if missing_required() else 0


if __name__ == "__main__":
    raise SystemExit(main())
