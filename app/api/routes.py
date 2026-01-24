from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas import FaceRequest, FaceResponse
from app.database import SessionLocal
from app.services.face_service import decode_image, extract_embedding
from app.security.liveness import basic_liveness_check
from app.services.vector_service import search_face, insert_face
from app.services.fraud_service import compute_fraud
from app.models import VerificationLog
import hashlib

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/verify", response_model=FaceResponse)
def verify_face(payload: FaceRequest, db: Session = Depends(get_db)):
    image = decode_image(payload.image_base64)

    if not basic_liveness_check(image):
        raise HTTPException(400, "Liveness check failed")

    embedding = extract_embedding(image)

    matches = search_face(embedding)
    fraud_score = compute_fraud(matches)

    embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()

    if fraud_score < 0.7:
        insert_face(embedding_hash, embedding)

    log = VerificationLog(
        user_id=payload.user_id,
        matched_user=matches[0].id if matches else None,
        similarity=matches[0].score if matches else 0.0,
        fraud_score=fraud_score
    )
    db.add(log)
    db.commit()

    return FaceResponse(
        verified=fraud_score < 0.7,
        fraud_score=fraud_score
    )
