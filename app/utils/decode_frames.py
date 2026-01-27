# app/utils/decode_frames.py
import base64
import cv2
import numpy as np

def decode_frames(frames_base64: list[str]) -> list[np.ndarray]:
    frames = []

    for frame_b64 in frames_base64:
        img_bytes = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is not None:
            frames.append(frame)

    return frames
