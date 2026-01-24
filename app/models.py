from sqlalchemy import Column, String, Integer, Float, DateTime
from app.database import Base
from datetime import datetime
import uuid

class UserFace(Base):
    __tablename__ = "user_faces"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(64), index=True)
    embedding_hash = Column(String(64), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class VerificationLog(Base):
    __tablename__ = "verification_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64))
    matched_user = Column(String(64), nullable=True)
    similarity = Column(Float)
    fraud_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
