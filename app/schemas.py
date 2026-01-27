from typing import List
from pydantic import BaseModel

class FaceVerifyRequest(BaseModel):
    user_id: str
    frames_base64: List[str]
    challenge: str
class FaceResponse(BaseModel):
    verified: bool
    fraud_score: float

class FaceRegisterRequest(BaseModel):
    user_id: str
    frames_base64: List[str]
    challenge: str

class RegisterResponse(BaseModel):
    registered: bool
    message: str
