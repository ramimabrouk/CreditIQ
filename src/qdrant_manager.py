from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import uuid

class QdrantManager:
    def __init__(self, collection_name="credit_memory"):
        # Initialize in-memory mode for prototype
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.vector_size = 9  # Update this based on SimilarityEngine output size!
                             # [Inc, Exp, Emp, Amt, Term, Cred] + [DocEmb(3)] = 6 + 3 = 9
        
        self._init_collection()

    def _init_collection(self):
        # Check if collection exists (always false for in-memory on init, but good practice)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def add_case(self, case_id, vector, payload):
        """
        Adds a single case to Qdrant.
        Vector must be a list of floats.
        """
        # Ensure vector is list of floats
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
            
        point = PointStruct(
            id=case_id if isinstance(case_id, int) else str(uuid.uuid4()), 
            vector=vector, 
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def search(self, query_vector, k=5, filter_conditions=None):
        """
        Search for nearest neighbors.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=filter_conditions
        )
        
        # Convert to standardized format List[(payload, distance)]
        # Qdrant returns score (cosine sim). Higher is better.
        # But our logic used Euclidean distance (Lower is better).
        # We need to adapt. 
        # Actually cosine distance = 1 - cosine similarity.
        # Let's just return the score and let DecisionEngine handle it?
        # DecisionEngine expected (case, distance). 
        # We will return (payload, score). 
        # Note: DecisionEngine used `1.0 / (dist + 0.01)` for weighting.
        # Use score directly for weighting since it's similarity.
        
        results = []
        for hit in search_result:
            results.append((hit.payload, hit.score))
            
        return results

    def get_count(self):
        info = self.client.get_collection(self.collection_name)
        return info.points_count
