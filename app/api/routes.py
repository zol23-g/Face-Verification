import uuid
import hashlib
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas import FaceRequest, FaceResponse
from app.database import SessionLocal
from app.security.advanced_liveness import verify_challenge
from app.services.face_service import extract_embedding
from app.services.vector_service import search_face, insert_face
from app.services.fraud_service import compute_fraud
from app.models import VerificationLog
from app.utils.decode_frames import decode_frames

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/verify", response_model=FaceResponse)
def verify_face(payload: FaceRequest, db: Session = Depends(get_db)):
    # 1️⃣ Decode frames
    frames = decode_frames(payload.frames_base64)
    if len(frames) < 5:
        raise HTTPException(400, "Not enough frames for verification")

    # 2️⃣ Liveness check (Using the optimized MediaPipe Tasks implementation)
    if not verify_challenge(payload.challenge, frames):
        # Log failed liveness attempt if necessary
        raise HTTPException(400, f"Liveness verification failed for challenge: {payload.challenge}")

    # 3️⃣ Extract embedding
    # Note: Ensure extract_embedding is optimized to pick the clearest frame
    embedding = extract_embedding(frames)

    # 4️⃣ Search Vector DB (Qdrant)
    matches = search_face(embedding)
    fraud_score = compute_fraud(matches)

    # 5️⃣ Prevent replay & Generate ID
    embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
    point_id = str(uuid.uuid4())

    # 6️⃣ Store embedding if legit (Fraud threshold < 0.7)
    is_verified = fraud_score < 0.7
    if is_verified:
        insert_face(
            point_id=point_id,
            embedding=embedding,
            payload={
                "user_id": payload.user_id,
                "embedding_hash": embedding_hash
            }
        )

    # 7️⃣ Log verification to SQL DB
    matched_user = matches[0].payload.get("user_id") if matches else None
    similarity = matches[0].score if matches else 0.0
    
    log = VerificationLog(
        user_id=payload.user_id,
        matched_user=matched_user,
        similarity=similarity,
        fraud_score=fraud_score
    )
    db.add(log)
    db.commit()

    return FaceResponse(
        verified=is_verified,
        fraud_score=fraud_score
    )
