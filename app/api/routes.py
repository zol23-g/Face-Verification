import uuid
from fastapi import HTTPException, Depends, Query, status
from fastapi import APIRouter
from pyparsing import Optional
from sqlalchemy.orm import Session
import logging
from app.schemas import (
    FaceRegisterRequest,
    RegisterResponse,
    StudentFaceListResponse,
    StudentFaceResponse,
    StudentFaceCreate,
    StudentFaceUpdate

)
from app.database import SessionLocal
from app.security.advanced_liveness import verify_challenge
from app.services.face_service import extract_embedding,face_app
from app.services.vector_service import init_collection, search_face_with_threshold
from app.models import StudentFace
from app.utils.database_helper import create_new_user, update_existing_user
from app.utils.decode_frames import calculate_embedding_quality, decode_frames
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional


router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(prefix="/api/v1/student-faces", tags=["Student Faces"])
logger = logging.getLogger(__name__)

@router.post("/register", response_model=RegisterResponse)
def register_face(payload: FaceRegisterRequest, db: Session = Depends(get_db)):
    """
    Register or update face data.
    - If user_id doesn't exist: Create new record
    - If user_id exists: Update existing record
    - Only calculate similarity metrics if comparison_enabled='Yes'
    """
    try:
        # Initialize collection
        init_collection()
        print(f"Incoming request payload - User ID: {payload.user_id}, Comparison: {payload.comparison_enabled}")
        
        # Decode frames
        frames = decode_frames(payload.frames_base64)
        if len(frames) < 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough frames"
            )
        
        # Extract embedding (always needed for registration)
        embedding = extract_embedding(frames)
        
        # Check for existing user with SAME user_id
        existing_user = db.query(StudentFace).filter(StudentFace.user_id == payload.user_id).first()
        
        # Initialize variables (will be used based on comparison_enabled)
        similar_matches = []
        warning_message = None
        fraud_alert = False
        similar_faces_count = 0
        max_similarity = 0.0
        quality_metrics = {"detection_confidence": 0.0, "embedding_quality": 0.0}
        
        # Check if comparison is enabled
        comparison_enabled = payload.comparison_enabled or "No"
        
        # Only calculate similarity metrics if comparison_enabled is 'Yes'
        if comparison_enabled and comparison_enabled.lower() == 'yes':
            print("Comparison enabled - performing similarity search")
            
            # Search for similar faces (different user_ids)
            SIMILARITY_THRESHOLD = 0.70
            matches = search_face_with_threshold(embedding, threshold=SIMILARITY_THRESHOLD)
            
            # Process similar matches
            for match in matches:
                match_user_id = match["payload"].get("user_id")
                
                # Skip if it's the same user (during update)
                if existing_user and match_user_id == existing_user.user_id:
                    continue
                    
                similar_matches.append({
                    "user_id": match_user_id,
                    "score": match["score"],
                    "first_name": match["payload"].get("first_name"),
                    "last_name": match["payload"].get("last_name"),
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
            
            # Calculate similarity metrics
            similar_faces_count = len(similar_matches)
            max_similarity = max([match["score"] for match in similar_matches]) if similar_matches else 0.0
            similar_matches.sort(key=lambda x: x["score"], reverse=True)
            
            # Calculate quality metrics
            quality_metrics = calculate_embedding_quality(frames)
        else:
            print("Comparison disabled - skipping similarity calculations")
            # Just calculate basic quality metrics without similarity search
            if frames:
                try:
                    # Simple quality check on first frame
                    faces = face_app.get(frames[0])
                    if faces and len(faces) > 0:
                        confidence = faces[0].det_score
                        quality_metrics = {
                            "detection_confidence": confidence,
                            "embedding_quality": min(1.0, confidence)
                        }
                except Exception as e:
                    print(f"Error calculating basic quality: {e}")
        
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
    












@router.get("/", response_model=StudentFaceListResponse)
def get_all_student_faces(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name or email"),
    verified_only: Optional[bool] = Query(None, description="Filter by verification status"),
    flagged_only: Optional[bool] = Query(None, description="Filter by flagged status"),
    sort_by: Optional[str] = Query("created_at", description="Field to sort by"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$", description="Sort order")
):
    """
    Get all student faces with pagination, filtering, and sorting
    """
    try:
        # Build query
        query = db.query(StudentFace)
        
        # Apply filters
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (StudentFace.first_name.ilike(search_term)) |
                (StudentFace.last_name.ilike(search_term)) |
                (StudentFace.email.ilike(search_term)) |
                (StudentFace.user_id.ilike(search_term))
            )
        
        if verified_only is not None:
            query = query.filter(StudentFace.is_verified == verified_only)
        
        if flagged_only is not None:
            query = query.filter(StudentFace.is_flagged == flagged_only)
        
        # Apply sorting
        sort_field = getattr(StudentFace, sort_by, StudentFace.created_at)
        if sort_order == "desc":
            query = query.order_by(sort_field.desc())
        else:
            query = query.order_by(sort_field.asc())
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        students = query.offset(offset).limit(page_size).all()
        
        return StudentFaceListResponse(
            total=total,
            page=page,
            page_size=page_size,
            students=students
        )
        
    except Exception as e:
        logger.error(f"Error fetching student faces: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch student faces"
        )

@router.get("/{user_id}", response_model=StudentFaceResponse)
def get_student_face(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a single student face by user_id
    """
    try:
        student = db.query(StudentFace).filter(
            StudentFace.user_id == user_id
        ).first()
        
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student face with user_id '{user_id}' not found"
            )
        
        return student
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching student face {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch student face"
        )

@router.post("/", response_model=StudentFaceResponse, status_code=status.HTTP_201_CREATED)
def create_student_face(
    student_data: StudentFaceCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new student face record
    """
    try:
        # Check if user_id already exists
        existing = db.query(StudentFace).filter(
            StudentFace.user_id == student_data.user_id
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Student face with user_id '{student_data.user_id}' already exists"
            )
        
        # Create new student face
        db_student = StudentFace(**student_data.dict())
        db.add(db_student)
        db.commit()
        db.refresh(db_student)
        
        logger.info(f"Created student face with user_id: {student_data.user_id}")
        return db_student
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating student face: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create student face"
        )

@router.put("/{user_id}", response_model=StudentFaceResponse)
def update_student_face(
    user_id: str,
    student_data: StudentFaceUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a student face record
    """
    try:
        # Find student
        student = db.query(StudentFace).filter(
            StudentFace.user_id == user_id
        ).first()
        
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student face with user_id '{user_id}' not found"
            )
        
        # Update fields
        update_data = student_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(student, field, value)
        
        student.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(student)
        
        logger.info(f"Updated student face with user_id: {user_id}")
        return student
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating student face {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update student face"
        )

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student_face(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a student face record
    """
    try:
        # Find student
        student = db.query(StudentFace).filter(
            StudentFace.user_id == user_id
        ).first()
        
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student face with user_id '{user_id}' not found"
            )
        
        # Delete student
        db.delete(student)
        db.commit()
        
        logger.info(f"Deleted student face with user_id: {user_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting student face {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete student face"
        )

@router.get("/{user_id}/fraud-metrics")
def get_student_fraud_metrics(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get fraud detection metrics for a student
    """
    try:
        student = db.query(StudentFace).filter(
            StudentFace.user_id == user_id
        ).first()
        
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student face with user_id '{user_id}' not found"
            )
        
        # Calculate risk level
        if student.max_similarity >= 0.85:
            risk_level = "high"
        elif student.max_similarity >= 0.75:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "user_id": student.user_id,
            "similarity_score": student.max_similarity,
            "similar_users_count": student.similar_faces_count,
            "is_flagged": student.is_flagged,
            "risk_level": risk_level,
            "is_verified": student.is_verified,
            "registration_count": student.registration_count,
            "last_similarity_check": student.last_similarity_check,
            "embedding_quality": student.embedding_quality,
            "detection_confidence": student.detection_confidence
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fraud metrics for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch fraud metrics"
        )