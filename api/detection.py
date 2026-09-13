from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import RTDETR
from PIL import Image
import io
import cv2
import numpy as np

router = APIRouter()
model = None

def load_model(model_path: str):
    global model
    model = RTDETR(model_path)

@router.post("/json")
async def detect_json(file: UploadFile = File(...)):
    from api.human_detector import detect_human
    
    image_bytes = await file.read()  # read ONCE
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    results = model(image, conf=0.80)[0]
    human_result = detect_human(image)  # reuse same image object

    detections = []
    for box in results.boxes:
        detections.append({
            "class": results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 2),
                "y1": round(float(box.xyxy[0][1]), 2),
                "x2": round(float(box.xyxy[0][2]), 2),
                "y2": round(float(box.xyxy[0][3]), 2),
            }
        })

    for h in human_result["boxes"]:
        detections.append({
            "class": "human",
            "confidence": h["confidence"],
            "bbox": {"x1": h["x1"], "y1": h["y1"], "x2": h["x2"], "y2": h["y2"]}
        })

    return JSONResponse({
        "total_detections": len(detections),
        "detections": detections,
        "human_detected": human_result["human_detected"],
        "human_count": human_result["count"]
    })

@router.post("/visual")
async def detect_visual(file: UploadFile = File(...)):
    from api.human_detector import detect_human
    import cv2

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image, conf=0.80)[0]

    # Draw RT-DETR boxes
    annotated = results.plot()

    # Draw MediaPipe human boxes on top
    human_result = detect_human(image)
    for h in human_result["boxes"]:
        cv2.rectangle(annotated, (h["x1"], h["y1"]), (h["x2"], h["y2"]), (0, 255, 0), 2)
        cv2.putText(annotated, f"human {h['confidence']}", (h["x1"], h["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

async def detect_json_internal(image: Image.Image) -> list:
    results = model(image, conf=0.80)[0]
    detections = []
    for box in results.boxes:
        detections.append({
            "class": results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 2),
                "y1": round(float(box.xyxy[0][1]), 2),
                "x2": round(float(box.xyxy[0][2]), 2),
                "y2": round(float(box.xyxy[0][3]), 2),
            }
        })
    return detections