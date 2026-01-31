from typing import Dict, List, Optional , Any
from pydantic import BaseModel, Field, validator

class FaceVerifyRequest(BaseModel):
    user_id: str
    frames_base64: List[str]
    challenge: str
    age: Optional[int] = None
    gender: Optional[str] = None

class FaceResponse(BaseModel):
    verified: bool
    message: str
    fraud_score: float
    similarity: float

class FaceRegisterRequest(BaseModel):
    user_id: str
    name: str
    email: str
    age: int
    gender: str
    frames_base64: List[str]
    # challenge: str

class RegisterResponse(BaseModel):
    registered: bool = False
    updated: bool
    user_id: str
    similar_faces_count: int
    max_similarity: float
    similar_users: List[Dict[str, Any]] = Field(default_factory=list)  # Details of similar users
    fraud_alert: bool = False
    requires_verification: bool = False
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