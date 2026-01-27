from typing import List
from pydantic import BaseModel

class FaceRequest(BaseModel):
    user_id: str
    frames_base64: List[str]
    challenge: str
class FaceResponse(BaseModel):
    verified: bool
    fraud_score: float
