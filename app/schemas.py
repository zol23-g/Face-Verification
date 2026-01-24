from pydantic import BaseModel

class FaceRequest(BaseModel):
    user_id: str
    image_base64: str

class FaceResponse(BaseModel):
    verified: bool
    fraud_score: float
