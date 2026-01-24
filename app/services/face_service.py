import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

def decode_image(base64_str: str):
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def extract_embedding(image):
    faces = face_app.get(image)
    if len(faces) != 1:
        raise ValueError("Exactly one face required")
    return faces[0].embedding
