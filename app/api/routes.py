import uuid
from fastapi import HTTPException, Depends, status
from fastapi import APIRouter
from sqlalchemy.orm import Session

from app.schemas import (
    FaceVerifyRequest,
    FaceRegisterRequest,
    FaceResponse,
    RegisterResponse
)
from app.database import SessionLocal
from app.security.advanced_liveness import verify_challenge
from app.services.face_service import extract_embedding,face_app
from app.services.vector_service import init_collection, insert_face, search_face, search_face_with_threshold, update_face_vector
from app.services.fraud_service import compute_fraud
from app.models import UserFace, VerificationLog
from app.utils.database_helper import create_new_user, update_existing_user
from app.utils.decode_frames import calculate_embedding_quality, decode_frames
from sqlalchemy.orm import Session
from datetime import datetime


router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/verify", response_model=FaceResponse)
def verify_face(payload: FaceVerifyRequest, db: Session = Depends(get_db)):
    print(f"[ROUTES] Verifying face for User ID: {payload.user_id}")
    user = db.query(UserFace).filter(UserFace.user_id == payload.user_id).first()
    if not user:
        raise HTTPException(404, f"User ID '{payload.user_id}' not found.")

    frames = decode_frames(payload.frames_base64)
    if len(frames) < 5:
        raise HTTPException(400, "Not enough frames")

    # if not verify_challenge(payload.challenge, frames):
    #     raise HTTPException(400, "Liveness verification failed")

    # --- ACCURACY BOOST: Multi-Frame Extraction ---
    embedding = extract_embedding(frames)

    matches = search_face(embedding)
    fraud_score = compute_fraud(matches)

    best_match = matches[0] if matches else None
    similarity = best_match.score if best_match else 0.0
    matched_user_id = best_match.payload.get("user_id") if best_match else None
    print (f"[ROUTES] Best match: {matched_user_id} with similarity {similarity:.4f} and fraud score {fraud_score:.4f}")
    
    # Threshold for verification (0.75 is standard for Buffalo_L)
    is_verified = (
        best_match is not None
        and similarity >= 0.75
        and matched_user_id == payload.user_id
        # and fraud_score < 0.7
    )

    if is_verified:
        user.is_verified = True
        db.commit()
        message = "Verification successful"
    else:
        message = "Verification failed: Identity mismatch"

    log = VerificationLog(
        user_id=payload.user_id,
        matched_user=matched_user_id,
        similarity=similarity,
        fraud_score=fraud_score
    )
    db.add(log)
    db.commit()

    return FaceResponse(
        verified=is_verified,
        message=message,
        fraud_score=fraud_score,
        similarity=similarity
    )



@router.post("/register", response_model=RegisterResponse)
def register_face(payload: FaceRegisterRequest, db: Session = Depends(get_db)):
    """
    Register or update face data.
    - If user_id doesn't exist: Create new record (ALWAYS, even with similar faces)
    - If user_id exists: Update existing record
    """
    try:
        # Initialize collection
        init_collection()
        
        # Decode frames
        frames = decode_frames(payload.frames_base64)
        if len(frames) < 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough frames"
            )
        
        # Extract embedding
        embedding = extract_embedding(frames)
        
        # Check for existing user with SAME user_id
        existing_user = db.query(UserFace).filter(UserFace.user_id == payload.user_id).first()
        
        # Search for similar faces (different user_ids)
        SIMILARITY_THRESHOLD = 0.70
        matches = search_face_with_threshold(embedding, threshold=SIMILARITY_THRESHOLD)
        
        # Process similar matches
        similar_matches = []
        warning_message = None
        fraud_alert = False
        
        for match in matches:
            match_user_id = match["payload"].get("user_id")
            
            # Skip if it's the same user (during update)
            if existing_user and match_user_id == existing_user.user_id:
                continue
                
            similar_matches.append({
                "user_id": match_user_id,
                "score": match["score"],
                "name": match["payload"].get("name"),
                "email": match["payload"].get("email")
            })
            
            # Set warning and fraud alert based on similarity (but don't block)
            if match["score"] >= 0.85:
                warning_message = f"⚠️ HIGH SIMILARITY ALERT: This face is {match['score']:.1%} similar to user '{match_user_id}'"
                fraud_alert = True
            elif match["score"] >= 0.75:
                if not warning_message:  # Only set if no higher warning
                    warning_message = f"⚠️ Similarity Warning: This face is {match['score']:.1%} similar to user '{match_user_id}'"
                fraud_alert = True
        
        # Calculate metrics
        similar_faces_count = len(similar_matches)
        max_similarity = max([match["score"] for match in similar_matches]) if similar_matches else 0.0
        similar_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate quality metrics
        quality_metrics = calculate_embedding_quality(frames)
        
        if existing_user:
            return update_existing_user(
                existing_user=existing_user,
                payload=payload,
                embedding=embedding,
                similar_matches=similar_matches,
                similar_faces_count=similar_faces_count,
                max_similarity=max_similarity,
                quality_metrics=quality_metrics,
                warning_message=warning_message,
                fraud_alert=fraud_alert,
                db=db
            )
        else:
            return create_new_user(
                payload=payload,
                embedding=embedding,
                similar_matches=similar_matches,
                similar_faces_count=similar_faces_count,
                max_similarity=max_similarity,
                quality_metrics=quality_metrics,
                warning_message=warning_message,
                fraud_alert=fraud_alert,
                db=db
            )
            
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        print(f"Registration error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )