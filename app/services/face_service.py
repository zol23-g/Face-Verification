import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace with the high-accuracy Buffalo_L model
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

def decode_image(base64_str: str):
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def extract_embedding(frames):
    """
    Extracts a robust 'Centroid Embedding' by fusing multiple high-quality frames.
    
    Strategy:
    1. Iterate through all provided frames.
    2. Filter for frames with exactly one face and high detection confidence.
    3. Extract embeddings from these valid frames.
    4. Calculate the mean (centroid) embedding to reduce noise and variance.
    5. Normalize the final vector for cosine similarity.
    """
    if not isinstance(frames, list):
        raise TypeError("extract_embedding expects a list of frames")

    if len(frames) == 0:
        raise ValueError("No frames provided")

    valid_embeddings = []
    
    for i, image in enumerate(frames):
        if not isinstance(image, np.ndarray):
            continue
            
        # Get face data
        faces = face_app.get(image)
        
        # Quality Filter:
        # - Exactly one face
        # - Detection score > 0.6 (ensures it's a clear face)
        if len(faces) == 1 and faces[0].det_score > 0.6:
            valid_embeddings.append(faces[0].embedding)
            
    if not valid_embeddings:
        # Fallback: If no high-quality frames found, try the middle frame with lower threshold
        middle_frame = frames[len(frames) // 2]
        faces = face_app.get(middle_frame)
        if not faces:
            raise ValueError("No face detected in any frame for embedding")
        return faces[0].embedding

    # --- ACCURACY BOOST: Multi-Frame Fusion ---
    # Convert list to numpy array [N, 512]
    embeddings_stack = np.vstack(valid_embeddings)
    
    # Calculate the mean embedding (Centroid)
    # This averages out noise from lighting, blur, or slight expression changes
    centroid_embedding = np.mean(embeddings_stack, axis=0)
    
    # Normalize the final vector (Crucial for Cosine Similarity in Qdrant)
    norm = np.linalg.norm(centroid_embedding)
    if norm > 0:
        centroid_embedding = centroid_embedding / norm
        
    print(f"[FACE SERVICE] Fused embedding from {len(valid_embeddings)} valid frames.")
    return centroid_embedding