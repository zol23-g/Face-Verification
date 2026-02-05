from sqlalchemy import JSON, Column, String, Integer, Float, DateTime, Boolean
from app.database import Base
from datetime import datetime
import uuid

class StudentFace(Base):
    __tablename__ = "student_faces"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Identity
    user_id = Column(String(64), unique=True, index=True, nullable=False)
    first_name = Column(String(128))
    last_name = Column(String(128))
    middle_name = Column(String(128), nullable=True)
    email = Column(String(128))

    # Biometrics
    age = Column(Integer, nullable=True)
   

    # Similarity & Fraud Intelligence
    similar_user_ids = Column(JSON, default=list)        # ["user_12", "user_88"]
    similar_faces_count = Column(Integer, default=0)     # how many identities collide
    max_similarity = Column(Float, default=0.0)          # best similarity score seen
    similarity_scores = Column(JSON, default=list)       # Store scores for each match: [{"user_id": "user_12", "score": 0.85}, ...]
    
    # Fraud Detection Metrics
    is_verified = Column(Boolean, default=False)
    is_flagged = Column(Boolean, default=False)          # Flag for suspicious activity
    registration_count = Column(Integer, default=1)      # updates over time
    last_similarity_check = Column(DateTime, nullable=True)  # When was similarity last checked
    match_history = Column(JSON, default=list)           # History of matches found
    
    # Face Quality Metrics
    embedding_quality = Column(Float, default=0.0)       # Quality score of face embedding (0-1)
    detection_confidence = Column(Float, default=0.0)    # Average face detection confidence
    token = Column(String(256), nullable=False)  # Optional token for additional verification
    comparison_enabled = Column(String (3), default="No")  # Whether comparison is enabled for this user
    
    # Audit
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

