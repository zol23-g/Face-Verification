from datetime import datetime
# At the top of your file
from fastapi import HTTPException, Depends, status
import uuid
import uuid
from app.models import StudentFace
from app.schemas import RegisterResponse
from app.services.vector_service import insert_face, update_face_vector


def update_existing_user(existing_user, payload, embedding, similar_matches, 
                         similar_faces_count, max_similarity, quality_metrics,
                         warning_message, fraud_alert, db):
    """Handle update of existing user"""
    
    # Update vector database (always update embedding)
    update_face_vector(
        point_id=existing_user.id,
        embedding=embedding,
        payload={
            "user_id": payload.user_id,
            "first_name": payload.first_name,
            "last_name": payload.last_name,
            "middle_name": payload.middle_name,
            "email": payload.email,
            "updated_at": datetime.utcnow().isoformat(),
            "comparison_enabled": payload.comparison_enabled or existing_user.comparison_enabled or "No"
        }
    )
    
    # Update SQL database with basic fields (always updated)
    existing_user.first_name = payload.first_name
    existing_user.last_name = payload.last_name
    existing_user.middle_name = payload.middle_name
    existing_user.email = payload.email
    existing_user.age = payload.age
    existing_user.token = payload.token
    existing_user.comparison_enabled = payload.comparison_enabled or existing_user.comparison_enabled or "No"
    existing_user.registration_count += 1
    existing_user.embedding_quality = quality_metrics.get("embedding_quality", existing_user.embedding_quality)
    existing_user.detection_confidence = quality_metrics.get("detection_confidence", existing_user.detection_confidence)
    existing_user.updated_at = datetime.utcnow()
    
    # Check if comparison is enabled
    comparison_enabled = payload.comparison_enabled or existing_user.comparison_enabled or "No"
    
    if comparison_enabled and comparison_enabled.lower() == 'yes':
        # Update similarity fields (only when comparison enabled)
        existing_user.similar_user_ids = [match["user_id"] for match in similar_matches]
        existing_user.similar_faces_count = similar_faces_count
        existing_user.max_similarity = max_similarity
        existing_user.similarity_scores = similar_matches
        existing_user.last_similarity_check = datetime.utcnow()
        existing_user.is_flagged = fraud_alert or (
            similar_faces_count > 3 or 
            max_similarity > 0.90 or
            quality_metrics.get("detection_confidence", 0.0) < 0.4
        )
    else:
        # Clear similarity fields when comparison is disabled
        existing_user.similar_user_ids = []
        existing_user.similar_faces_count = 0
        existing_user.max_similarity = 0.0
        existing_user.similarity_scores = []
        existing_user.is_flagged = False
        # Keep last_similarity_check as is (don't update when comparison disabled)
    
    # Update match history
    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": "update",
        "comparison_enabled": comparison_enabled
    }
    
    # Only add similarity details if comparison was enabled
    if comparison_enabled and comparison_enabled.lower() == 'yes':
        history_entry.update({
            "similar_faces_count": similar_faces_count,
            "max_similarity": max_similarity,
            "note": "Updated with similarity check" if similar_faces_count > 0 else "Updated"
        })
    
    if existing_user.match_history:
        existing_user.match_history.append(history_entry)
    else:
        existing_user.match_history = [history_entry]
    
    db.commit()
    
    # Prepare response message
    if not comparison_enabled or comparison_enabled.lower() != 'yes':
        message = "User updated successfully (comparison disabled)"
    elif similar_faces_count == 0:
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
        comparison_enabled=comparison_enabled,
        message=message
    )

def create_new_user(payload, embedding, similar_matches, similar_faces_count,
                    max_similarity, quality_metrics, warning_message, 
                    fraud_alert, db):
    """Handle creation of new user"""
    
    # Generate ID
    point_id = str(uuid.uuid4())
    
    # Insert into vector database (always store embedding)
    insert_face(
        point_id=point_id,
        embedding=embedding,
        payload={
            "user_id": payload.user_id,
            "first_name": payload.first_name,
            "last_name": payload.last_name,
            "middle_name": payload.middle_name,
            "email": payload.email,
            "created_at": datetime.utcnow().isoformat(),
            "comparison_enabled": payload.comparison_enabled or "No"
        }
    )
    
    # Determine if verification is required (only when comparison enabled)
    requires_verification = False
    if payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes':
        requires_verification = (max_similarity >= 0.85 or similar_faces_count >= 3)
    
    # Prepare match history entry
    match_history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": "create",
        "comparison_enabled": payload.comparison_enabled or "No"
    }
    
    # Only add similarity details if comparison was enabled
    if payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes':
        match_history_entry.update({
            "similar_faces_count": similar_faces_count,
            "max_similarity": max_similarity,
            "note": "Registered with similarity check" if similar_faces_count > 0 else "Registered"
        })
    
    # Create in SQL database
    user_face = StudentFace(
        id=point_id,
        user_id=payload.user_id,
        first_name=payload.first_name,
        last_name=payload.last_name,
        middle_name=payload.middle_name,
        email=payload.email,
        age=payload.age,
        token=payload.token,
        comparison_enabled=payload.comparison_enabled or "No",
        # Similarity fields (populated only when comparison enabled)
        similar_user_ids=[match["user_id"] for match in similar_matches] if (payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes' and similar_matches) else [],
        similar_faces_count=similar_faces_count if (payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes') else 0,
        max_similarity=max_similarity if (payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes') else 0.0,
        similarity_scores=similar_matches if (payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes' and similar_matches) else [],
        # Quality metrics (always calculated)
        embedding_quality=quality_metrics.get("embedding_quality", 0.0),
        detection_confidence=quality_metrics.get("detection_confidence", 0.0),
        last_similarity_check=datetime.utcnow() if (payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes') else None,
        match_history=[match_history_entry],
        # Fraud flags (only when comparison enabled)
        is_flagged=fraud_alert if (payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes') else False,
        # is_verified=(not requires_verification) if (payload.comparison_enabled and payload.comparison_enabled.lower() == 'yes') else True
    )
    
    db.add(user_face)
    db.commit()
    
    # Prepare response message
    if not payload.comparison_enabled or payload.comparison_enabled.lower() != 'yes':
        message = "User registered successfully (comparison disabled)"
    elif similar_faces_count == 0:
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
        comparison_enabled=payload.comparison_enabled or "No",
        message=message
    )