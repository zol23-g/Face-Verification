# app/utils/decode_frames.py
import base64
from typing import Dict, List
import cv2
import numpy as np
from app.services.face_service import face_app

def decode_frames(frames_base64: list[str]) -> list[np.ndarray]:
    frames = []

    for frame_b64 in frames_base64:
        img_bytes = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is not None:
            frames.append(frame)

    return frames

def calculate_embedding_quality(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Calculate quality metrics from frames
    Returns: {"detection_confidence": float, "embedding_quality": float}
    """
    if not frames:
        return {"detection_confidence": 0.0, "embedding_quality": 0.0}
    
    total_confidence = 0.0
    valid_frames = 0
    
    for frame in frames:
        faces = face_app.get(frame)
        if faces and len(faces) > 0:
            total_confidence += faces[0].det_score
            valid_frames += 1
    
    if valid_frames == 0:
        return {"detection_confidence": 0.0, "embedding_quality": 0.0}
    
    avg_confidence = total_confidence / valid_frames
    embedding_quality = min(1.0, avg_confidence)  # Scale to 0-1
    
    return {
        "detection_confidence": avg_confidence,
        "embedding_quality": embedding_quality
    }