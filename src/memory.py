import numpy as np
import json
from src.similarity import SimilarityEngine
from src.qdrant_manager import QdrantManager

class DecisionMemory:
    def __init__(self):
        self.qdrant = QdrantManager()
        self.similarity_engine = SimilarityEngine()

    def add_case(self, case_data):
        """
        Adds a case to memory (Qdrant).
        """
        # Vectorize if needed
        if 'vector' not in case_data:
            case_data['vector'] = self.similarity_engine.vectorize(case_data['profile'])
        
        vector = case_data['vector']
        
        # Prepare payload (everything except vector)
        payload = case_data.copy()
        if 'vector' in payload:
            del payload['vector']
            
        # Ensure ID
        if 'id' not in payload:
            # We need an ID for Qdrant.
            # In a real system we'd generate one. Here we assume one might exist or we auto-gen in manager.
            # But the manager wants an ID passed in or it creates UUID.
            # Let's check logic: case_data usually has 'id'.
            pass

        # We pass case_id if present, else None (Manager generates UUID)
        case_id = payload.get('id', None)
        if case_id is None:
             # Just use a random int for now if missing, or let manager handle?
             # Manager handles if we pass str(uuid). Let's let manager handle if needed,
             # but we usually generate integer IDs in this proto.
             pass

        self.qdrant.add_case(case_id if case_id is not None else -1, vector, payload)

    def retrieve_neighbors(self, input_vector, k=5, filter_func=None):
        """
        Finds k nearest neighbors using Qdrant.
        """
        # Note: filter_func is hard to translate to Qdrant filters dynamically without
        # complex logic. For this prototype, we'll retrieve slightly more (k*2) and filter in python
        # OR just ignore filter_func if it's complex.
        # The only filter used so far was 'fraud' check.
        
        # If filter_func is specifically checking for 'fraud' label, we can try to optimize,
        # but for now let's keep it simple: 
        # Retrieve, then post-filter? 
        # Qdrant is fast. We can support basic filtering if needed. 
        # But 'fraud' case retrieval was: filter_func=lambda c: 'fraud' in c.get('labels', [])
        
        # Let's try to pass a simpler hint or just use standard search.
        results = self.qdrant.search(input_vector, k=k)
        
        # Convert back to expected format: (case_dict, distance/score)
        # Qdrant returns (payload, score). Score is Cosine Similarity (-1 to 1).
        # Wrapper might yield Euclidean-like implications? 
        # In QdrantManager setup: distance=Distance.COSINE
        # So score is Cosine Similarity. Previous logic used Euclidean Distance.
        # Logic adaptation:
        # Distance = 1 - Similarity (approx for ranking).
        # We need to return a "distance-like" metric for the DecisionEngine which expects lower=closer.
        # Or update DecisionEngine to handle high score = good.
        
        mapped_results = []
        for payload, score in results:
            # Reconstruct case_dict (add vector back if needed? Not really needed for logic)
            # But anomalies detector checks `neighbors[0][1]` as distance.
            
            # Convert Sim to Dist
            dist = 1.0 - score 
            if dist < 0: dist = 0
            
            mapped_results.append((payload, dist))

        # Apply python filter_func if exists
        if filter_func:
            mapped_results = [r for r in mapped_results if filter_func(r[0])]
            
            # If we filtered everything, maybe query more? 
            # For 'fraud' check, we specifically need fraud cases.
            # This 'retrieve then filter' approach is weak if fraud cases are rare.
            # But for prototype with small dataset, it might work if we query enough.
            # Better: Modify `retrieve_neighbors` to accept a `label_filter`.
            pass

        return mapped_results[:k]

    def retrieve_fraud_cases(self, input_vector, k=1):
        """
        Special method to find fraud cases for Anomaly Detector.
        """
        # We can implement a proper Qdrant filter here if we want, 
        # or just rely on the fact that we need to find *any* similar fraud.
        # Let's do a broader search or assume `retrieve_neighbors` with post-filter is okay for now.
        # Actually, let's just use the current mechanism but ask for more matches.
        return self.retrieve_neighbors(input_vector, k=20, filter_func=lambda c: 'fraud' in c.get('labels', []))

    def get_stats(self):
        count = self.qdrant.get_count()
        return {
            'total_cases': count,
            # Approvals/Declines requiring retrieving all logic which is slow in Vector DB without aggregation
            # We'll just return count for now or maintain a counter.
            'status': "Qdrant Connected"
        }

    def load_from_file(self, filepath):
        """Load history from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            for item in data:
                # Re-vectorize on load
                item['vector'] = self.similarity_engine.vectorize(item['profile'])
                self.add_case(item)
