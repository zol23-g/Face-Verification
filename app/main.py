from fastapi import FastAPI
from app.api.routes import router
from app.database import Base, engine
from app.services.vector_service import init_collection

app = FastAPI(title="Face Verification System")

Base.metadata.create_all(bind=engine)
init_collection()

app.include_router(router)
