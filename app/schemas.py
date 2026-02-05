from datetime import datetime
from typing import Dict, List, Optional , Any
from pydantic import BaseModel, Field, validator

class FaceVerifyRequest(BaseModel):
    user_id: str
    frames_base64: List[str]
    challenge: str
    age: Optional[int] = None
    # gender: Optional[str] = None

class FaceResponse(BaseModel):
    verified: bool
    message: str
    fraud_score: float
    similarity: float

class FaceRegisterRequest(BaseModel):
    user_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    email: str
    age: int
    frames_base64: List[str]
    token: Optional[str] = None
    first_name:Optional[str] = None
    last_name:Optional[str] = None
    middle_name:Optional[str] = None
    comparison_enabled:Optional[str] = None
    @validator('comparison_enabled')
    def validate_comparison_enabled(cls, v):
        if v and v.lower() not in ["yes", "no"]:
            raise ValueError('comparison_enabled must be "Yes" or "No" (case-insensitive)')
        return v

class RegisterResponse(BaseModel):
    registered: bool = False
    updated: bool
    user_id: str
    similar_faces_count: int
    max_similarity: float
    similar_users: List[Dict[str, Any]] = Field(default_factory=list)  # Details of similar users
    fraud_alert: bool = False
    requires_verification: bool = False
    comparison_enabled: str = "No"
    warning_message: Optional[str] = None  # Warning if high similarity found
    message: str

    @validator('requires_verification', always=True)
    def set_requires_verification(cls, v, values):
        """Auto-set requires_verification based on similarity"""
        if 'max_similarity' in values and values['max_similarity'] >= 0.85:
            return True
        if 'similar_faces_count' in values and values['similar_faces_count'] >= 3:
            return True
        return False
    
    @validator('fraud_alert', always=True)
    def set_fraud_alert(cls, v, values):
        """Auto-set fraud_alert based on similarity"""
        if 'max_similarity' in values and values['max_similarity'] >= 0.75:
            return True
        return False

class SimilarityMatch(BaseModel):
    user_id: str
    score: float
    name: Optional[str] = None
    email: Optional[str] = None

class FraudMetrics(BaseModel):
    similarity_score: float
    similar_users_count: int
    is_flagged: bool
    risk_level: str  # low, medium, high


    # Request Schemas
class StudentFaceCreate(BaseModel):
    user_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    email: str
    age: Optional[int] = None
    token: Optional[str] = None
    comparison_enabled: Optional[str] = "No"
    is_verified: Optional[bool] = False
    is_flagged: Optional[bool] = False

    @validator('comparison_enabled')
    def validate_comparison_enabled(cls, v):
        if v and v.lower() not in ["yes", "no"]:
            raise ValueError('comparison_enabled must be "Yes" or "No"')
        return v.title() if v else "No"

class StudentFaceUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    token: Optional[str] = None
    comparison_enabled: Optional[str] = None
    is_verified: Optional[bool] = None
    is_flagged: Optional[bool] = None
    embedding_quality: Optional[float] = None
    detection_confidence: Optional[float] = None
    
    @validator('comparison_enabled')
    def validate_comparison_enabled(cls, v):
        if v and v.lower() not in ["yes", "no"]:
            raise ValueError('comparison_enabled must be "Yes" or "No"')
        return v.title() if v else None

# Response Schemas
class SimilarityScore(BaseModel):
    user_id: str
    score: float

class StudentFaceResponse(BaseModel):
    id: str
    user_id: str
    first_name: Optional[str]
    last_name: Optional[str]
    middle_name: Optional[str]
    email: str
    age: Optional[int]
    
    # Similarity & Fraud Intelligence
    similar_user_ids: List[str] = []
    similar_faces_count: int = 0
    max_similarity: float = 0.0
    similarity_scores: List[Dict[str, Any]] = []
    
    # Fraud Detection Metrics
    is_verified: bool = False
    is_flagged: bool = False
    registration_count: int = 1
    last_similarity_check: Optional[datetime] = None
    match_history: List[Dict[str, Any]] = []
    
    # Face Quality Metrics
    embedding_quality: float = 0.0
    detection_confidence: float = 0.0
    token: str
    comparison_enabled: str = "No"
    
    # Audit
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class StudentFaceListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    students: List[StudentFaceResponse]
    
    class Config:
        from_attributes = True

