from fastapi import FastAPI, UploadFile, File, Query
from typing import List
from ultralytics import YOLO
from PIL import Image
import io

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Pretrained YOLOv8 Auto-Label API (Supervisely)")

# ---------------------------------------------------------
# Lazy-loaded YOLO model (CPU-only)
# ---------------------------------------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # COCO pretrained
    return _model


# ---------------------------------------------------------
# YOLO → Supervisely annotation converter
# ---------------------------------------------------------
def yolo_to_supervisely(results, image_name: str, img_w: int, img_h: int):
    objects = []

    for b in results.boxes:
        cls_id = int(b.cls[0])
        label = get_model().names[cls_id]

        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        # Clamp to image size
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))

        objects.append({
            "id": None,
            "classId": None,
            "objectId": None,
            "description": "",
            "geometryType": "rectangle",
            "labelerLogin": None,
            "createdAt": None,
            "updatedAt": None,
            "tags": [],
            "classTitle": label,
            "points": {
                "exterior": [[x1, y1], [x2, y2]],
                "interior": []
            }
        })

    return {
        "imageId": None,
        "imageName": image_name,
        "createdAt": None,
        "updatedAt": None,
        "link": None,
        "annotation": {
            "description": "",
            "tags": [],
            "size": {
                "height": img_h,
                "width": img_w
            },
            "objects": objects
        }
    }



# ---------------------------------------------------------
# API endpoint
# ---------------------------------------------------------
@app.post("/auto-label/bbox")
async def auto_label_bbox(
    files: List[UploadFile] = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.7, ge=0.0, le=1.0),
):
    """
    Auto-label images using pretrained YOLOv8 and return
    Supervisely-compatible annotation JSON.
    """
    outputs = []
    model = get_model()

    for f in files:
        # Read image
        data = await f.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_w, img_h = img.size

        # Run inference (CPU)
        results = model.predict(
            img,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]

        # Convert to Supervisely format
        ann = yolo_to_supervisely(
            results,
            image_name=f.filename,
            img_w=img_w,
            img_h=img_h
        )

        outputs.append(ann)

    return {
        "status": "success",
        "count": len(outputs),
        "results": outputs
    }


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}
