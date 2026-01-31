# services/vector_service.py
from typing import Any, Dict
from qdrant_client import QdrantClient, models
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

# Initialize the client
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

def init_collection():
    """Initialize or recreate the collection"""
    if not client.collection_exists(collection_name=QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=512, 
                distance=models.Distance.COSINE
            )
        )

def search_face_with_threshold(embedding, threshold: float = 0.70, limit: int = 10):
    """
    Search for similar faces with a similarity threshold.
    Returns matches above threshold with their scores and payloads.
    """
    search_result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding.tolist(),
        limit=limit,
        with_payload=True,
        score_threshold=threshold
    )
    
    matches = []
    for point in search_result.points:
        matches.append({
            "point_id": point.id,
            "score": point.score,
            "payload": point.payload
        })
    
    return matches


def insert_face(point_id: str, embedding, payload: dict):
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,                
                vector=embedding.tolist(),
                payload=payload             
            )
        ],
        wait=True
    )
def search_face(embedding, limit=5):
    """
    Searches for similar faces using the new query_points API.
    """
    # In version 1.16.2+, 'search' is replaced by 'query_points'
    search_result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding.tolist(),  # Pass the embedding directly to 'query'
        limit=limit
    )
    return search_result.points  # query_points returns a QueryResponse; .points contains the hits

def update_face_vector(point_id: str, embedding, payload: Dict[str, Any]):
    """Update an existing face vector"""
    # Qdrant's upsert will update if point_id exists
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
        ],
        wait=True
    )

def get_face_by_user_id(user_id: str):
    """
    Search for a face by user_id in payload.
    Returns the point if found.
    """
    search_result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        ),
        limit=1
    )
    
    if search_result[0]:
        point = search_result[0][0]
        return {
            "point_id": point.id,
            "payload": point.payload
        }
    return None