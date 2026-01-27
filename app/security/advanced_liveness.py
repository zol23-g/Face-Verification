import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
import os

# Path to the model file
MODEL_PATH = 'face_landmarker.task'

# Landmark indices for eyes (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Global variable to cache the landmarker instance for performance
_landmarker_cache = None

def get_landmarker():
    """
    Initialize and return the Face Landmarker task, using a cached instance if available.
    """
    global _landmarker_cache
    if _landmarker_cache is not None:
        return _landmarker_cache
        
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
        
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    _landmarker_cache = vision.FaceLandmarker.create_from_options(options)
    return _landmarker_cache

def eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculate the Eye Aspect Ratio (EAR) to detect blinks.
    """
    p = [landmarks[i] for i in eye_indices]
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def detect_blink(frames):
    blink_count = 0
    prev_ear = None
    landmarker = get_landmarker()

    for frame in frames:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            continue

        face_landmarks = result.face_landmarks[0]
        landmarks_array = np.array([(lm.x, lm.y) for lm in face_landmarks])

        left_ear = eye_aspect_ratio(landmarks_array, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks_array, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2

        # Sensitivity adjustment: using a slightly more lenient threshold
        if prev_ear is not None and prev_ear > 0.22 and ear < 0.19:
            blink_count += 1

        prev_ear = ear

    return blink_count >= 1

def detect_head_turn(frames, direction="left"):
    landmarker = get_landmarker()
    nose_positions = []

    for frame in frames:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            continue

        # Nose tip (1) and additional points for reference (e.g., 4 for nose bridge)
        nose_tip = result.face_landmarks[0][1]
        nose_positions.append(nose_tip.x)

    if len(nose_positions) < 2:
        return False

    # Calculate the total movement relative to the initial position
    # We use the max deviation to capture the peak of the turn
    initial_pos = nose_positions[0]
    
    if direction == "left":
        # Turning left means the nose moves to the LEFT in the image (smaller X)
        # However, if the camera is mirrored, it might be the opposite.
        # We'll check for a significant deviation from the starting point.
        min_pos = min(nose_positions)
        movement = min_pos - initial_pos
        return movement < -0.02  # Reduced threshold from 0.03 to 0.02 for better sensitivity
        
    if direction == "right":
        # Turning right means the nose moves to the RIGHT in the image (larger X)
        max_pos = max(nose_positions)
        movement = max_pos - initial_pos
        return movement > 0.02  # Reduced threshold from 0.03 to 0.02 for better sensitivity

    return False

def generate_challenge():
    return random.choice(["blink", "turn_left", "turn_right"])

def verify_challenge(challenge: str, frames: list) -> bool:
    print(f"[LIVENESS] Challenge: {challenge}")
    print(f"[LIVENESS] Frames received: {len(frames)}")

    if challenge == "blink":
        detected = detect_blink(frames)
        print(f"[LIVENESS] Blink detected: {detected}")
        return detected

    if challenge == "turn_left":
        detected = detect_head_turn(frames, direction="left")
        print(f"[LIVENESS] Head left detected: {detected}")
        return detected

    if challenge == "turn_right":
        detected = detect_head_turn(frames, direction="right")
        print(f"[LIVENESS] Head right detected: {detected}")
        return detected

    return False