from qdrant_client import QdrantClient, models
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

# Initialize the client
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

def init_collection():
    """
    Creates the collection if it doesn't exist, or recreates it if needed.
    """
    # Check if collection exists first
    if client.collection_exists(collection_name=QDRANT_COLLECTION):
        # If you want to keep the 'recreate' logic (delete and start fresh):
        client.delete_collection(collection_name=QDRANT_COLLECTION)
    
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
    )

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
