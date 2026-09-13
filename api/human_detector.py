import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import urllib.request
import os

MODEL_PATH = "human_detector_model.tflite"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=["person"]
)
detector = vision.ObjectDetector.create_from_options(options)

def detect_human(image: Image.Image) -> dict:
    img_array = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    result = detector.detect(mp_image)

    humans = []
    for detection in result.detections:
        if detection.categories[0].category_name == "person":
            bbox = detection.bounding_box
            humans.append({
                "x1": bbox.origin_x,
                "y1": bbox.origin_y,
                "x2": bbox.origin_x + bbox.width,
                "y2": bbox.origin_y + bbox.height,
                "confidence": round(detection.categories[0].score, 3)
            })

    return {
        "human_detected": len(humans) > 0,
        "count": len(humans),
        "boxes": humans
    }