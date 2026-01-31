from datetime import datetime
# At the top of your file
from fastapi import HTTPException, Depends, status
import uuid
import uuid
from app.models import UserFace
from app.schemas import RegisterResponse
from app.services.vector_service import insert_face, update_face_vector


def update_existing_user(existing_user, payload, embedding, similar_matches, 
                         similar_faces_count, max_similarity, quality_metrics,
                         warning_message, fraud_alert, db):
    """Handle update of existing user"""
    
    # Update vector database
    update_face_vector(
        point_id=existing_user.id,
        embedding=embedding,
        payload={
            "user_id": payload.user_id,
            "name": payload.name,
            "email": payload.email,
            "updated_at": datetime.utcnow().isoformat()
        }
    )
    
    # Update SQL database
    existing_user.name = payload.name
    existing_user.email = payload.email
    existing_user.age = payload.age
    existing_user.gender = payload.gender
    existing_user.similar_user_ids = [match["user_id"] for match in similar_matches]
    existing_user.similar_faces_count = similar_faces_count
    existing_user.max_similarity = max_similarity
    existing_user.similarity_scores = similar_matches
    existing_user.registration_count += 1
    existing_user.embedding_quality = quality_metrics["embedding_quality"]
    existing_user.detection_confidence = quality_metrics["detection_confidence"]
    existing_user.last_similarity_check = datetime.utcnow()
    existing_user.updated_at = datetime.utcnow()
    
    # Update match history
    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "similar_faces_count": similar_faces_count,
        "max_similarity": max_similarity,
        "action": "update"
    }
    if existing_user.match_history:
        existing_user.match_history.append(history_entry)
    else:
        existing_user.match_history = [history_entry]
    
    # Update flag based on fraud alert
    existing_user.is_flagged = fraud_alert or (
        similar_faces_count > 3 or 
        max_similarity > 0.90 or
        quality_metrics["detection_confidence"] < 0.4
    )
    
    db.commit()
    
    # Prepare response message
    if similar_faces_count == 0:
        message = "User updated successfully"
    else:
        message = f"User updated. Found {similar_faces_count} similar faces."
    
    return RegisterResponse(
        registered=False,
        updated=True,
        user_id=payload.user_id,
        similar_faces_count=similar_faces_count,
        max_similarity=max_similarity,
        similar_users=similar_matches,
        warning_message=warning_message,
        fraud_alert=fraud_alert,
        message=message
    )

def create_new_user(payload, embedding, similar_matches, similar_faces_count,
                    max_similarity, quality_metrics, warning_message, 
                    fraud_alert, db):
    """Handle creation of new user (ALWAYS allows, even with similar faces)"""
    
    # Generate ID
    point_id = str(uuid.uuid4())
    
    # Insert into vector database
    insert_face(
        point_id=point_id,
        embedding=embedding,
        payload={
            "user_id": payload.user_id,
            "name": payload.name,
            "email": payload.email,
            "created_at": datetime.utcnow().isoformat()
        }
    )
    
    # Determine if verification is required
    requires_verification = (max_similarity >= 0.85 or similar_faces_count >= 3)
    
    # Create in SQL database
    user_face = UserFace(
        id=point_id,
        user_id=payload.user_id,
        name=payload.name,
        email=payload.email,
        age=payload.age,
        gender=payload.gender,
        similar_user_ids=[match["user_id"] for match in similar_matches],
        similar_faces_count=similar_faces_count,
        max_similarity=max_similarity,
        similarity_scores=similar_matches,
        embedding_quality=quality_metrics["embedding_quality"],
        detection_confidence=quality_metrics["detection_confidence"],
        last_similarity_check=datetime.utcnow(),
        match_history=[{
            "timestamp": datetime.utcnow().isoformat(),
            "similar_faces_count": similar_faces_count,
            "max_similarity": max_similarity,
            "action": "create",
            "note": "Registered with similar faces" if similar_faces_count > 0 else None
        }],
        is_flagged=fraud_alert,
        is_verified=(not requires_verification)  # Auto-verify if no high similarity
    )
    
    db.add(user_face)
    db.commit()
    
    # Prepare response message
    if similar_faces_count == 0:
        message = "User registered successfully"
    elif fraud_alert:
        message = f"⚠️ User registered with {similar_faces_count} similar faces. Requires verification."
    else:
        message = f"User registered. Found {similar_faces_count} similar faces."
    
    return RegisterResponse(
        registered=True,
        updated=False,
        user_id=payload.user_id,
        similar_faces_count=similar_faces_count,
        max_similarity=max_similarity,
        similar_users=similar_matches,
        warning_message=warning_message,
        fraud_alert=fraud_alert,
        message=message
    )