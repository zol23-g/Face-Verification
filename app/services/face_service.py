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

def extract_embedding(frames):
    """
    frames: List[np.ndarray]
    returns: np.ndarray (embedding)
    """

    # 🛑 HARD GUARD — prevents this bug forever
    if not isinstance(frames, list):
        raise TypeError("extract_embedding expects a list of frames")

    if len(frames) == 0:
        raise ValueError("No frames provided")

    # ✅ ALWAYS select ONE frame
    image = frames[len(frames) // 2]

    if not isinstance(image, np.ndarray):
        raise TypeError("Selected frame is not a numpy array")

    faces = face_app.get(image)

    if not faces:
        raise ValueError("No face detected for embedding")

    if len(faces) > 1:
        raise ValueError("Multiple faces detected")

    return faces[0].embedding
