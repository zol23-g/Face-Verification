from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

def init_collection():
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

def insert_face(face_id, embedding):
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[{"id": face_id, "vector": embedding.tolist()}]
    )

def search_face(embedding, limit=5):
    return client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedding.tolist(),
        limit=limit
    )
