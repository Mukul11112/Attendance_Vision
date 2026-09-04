# Model Setup

Three ONNX files must exist in `models/weights/`:

| File | Role | Size |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | Face detection (runs via OpenCV, no ORT needed) | ~0.3 MB |
| `arcface_w600k_r50.onnx` | Face embeddings, 512-d (ONNX Runtime CPU) | ~166 MB |
| `yolov8n.onnx` | Person detection, class 0 only (ONNX Runtime CPU) | ~12 MB |

## Automatic

```
python scripts\download_models.py
python scripts\download_models.py --status
```

The script downloads YuNet from the OpenCV Zoo, extracts `w600k_r50.onnx`
from InsightFace's `buffalo_l.zip` release, and exports `yolov8n.onnx` via
`ultralytics` (install it once with `pip install ultralytics` if prompted —
it can be uninstalled after the export).

## Manual (proxy-blocked office network)

1. **YuNet** — download `face_detection_yunet_2023mar.onnx` from
   `github.com/opencv/opencv_zoo` (models/face_detection_yunet) and save as
   `models/weights/face_detection_yunet_2023mar.onnx`.
2. **ArcFace** — download `buffalo_l.zip` from the InsightFace v0.7 GitHub
   release, open the zip, copy `w600k_r50.onnx`, save as
   `models/weights/arcface_w600k_r50.onnx`.
3. **YOLOv8n** — on any machine with internet:
   `pip install ultralytics` then
   `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=12, imgsz=640)"`
   and copy the produced `yolov8n.onnx` to `models/weights/yolov8n.onnx`.

Re-run `--status` until all three show `OK`.

Note: any ArcFace-compatible 512-d recognition ONNX with 112×112 input and
(1,3,112,112) NCHW RGB [-1,1] preprocessing works at the same path. If you
swap the model, delete `data/face_embeddings/` and re-enroll — embeddings
from different models are not comparable.
