from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.database import Base, engine
from app.services.vector_service import init_collection

app = FastAPI(title="Face Verification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
init_collection()

app.include_router(router)
