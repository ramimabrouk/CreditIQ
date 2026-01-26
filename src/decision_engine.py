from src.similarity import SimilarityEngine
from src.memory import DecisionMemory
from src.anomaly import AnomalyDetector
import numpy as np

class DecisionEngine:
    def __init__(self):
        self.memory = DecisionMemory()
        self.similarity = SimilarityEngine()
        self.anomaly_detector = AnomalyDetector(self.memory)

    def load_history(self, filepath):
        self.memory.load_from_file(filepath)

    def evaluate_application(self, application_profile):
        """
        Full pipeline: Profile -> Vector -> Anomaly Check -> Memory Retrieval -> Recommendation
        """
        # 1. Vectorize
        vector = self.similarity.vectorize(application_profile)

        # 2. Retrieve Similar Cases
        neighbors = self.memory.retrieve_neighbors(vector, k=5)

        # 3. Detect Anomalies
        anomalies = self.anomaly_detector.analyze(application_profile, vector, neighbors)

        # 4. Formulate Recommendation based on Neighbors
        
        relevant_neighbors = neighbors
        
        approval_score = 0
        if relevant_neighbors:
            # Weighted vote by distance (inverse distance)
            total_weight = 0
            weighted_approve = 0
            
            for case, dist in relevant_neighbors:
                # Dist is 1 - CosineSim.
                # If dist is 0 (identical), weight is high.
                weight = 1.0 / (dist + 0.05) 
                total_weight += weight
                if case['decision'] == 'approve':
                    weighted_approve += weight
            
            if total_weight > 0:
                approval_score = weighted_approve / total_weight
        
        # Decision Logic
        recommendation = "REVIEW"
        confidence = 0.0
        
        if anomalies:
            recommendation = "DECLINE (Anomaly)"
            confidence = 1.0 
        elif approval_score > 0.6:
            recommendation = "APPROVE"
            confidence = approval_score
        elif approval_score < 0.4:
            recommendation = "DECLINE"
            confidence = 1.0 - approval_score
        else:
            recommendation = "MANUAL_REVIEW"
            confidence = 0.5

        # 5. Explainability
        explanation = self._generate_explanation(recommendation, neighbors, anomalies)

        return {
            "recommendation": recommendation,
            "confidence": round(confidence, 2),
            "anomalies": anomalies,
            "explanation": explanation,
            "similar_cases": [
                {
                    "id": casing.get('id', 'Unknown'),
                    "decision": casing.get('decision', 'Unknown'),
                    "distance": round(dist, 3),
                    "match_reason": self._explain_similarity(application_profile, casing.get('profile', {}))
                 } 
                for casing, dist in neighbors
            ]
        }

    def _explain_similarity(self, profile_a, profile_b):
        """Simple text diff of key drivers."""
        reasons = []
        if abs(profile_a.get('income',0) - profile_b.get('income',0)) < 1000:
            reasons.append("Similar Income")
        if profile_a.get('credit_history') == profile_b.get('credit_history'):
            reasons.append("Same Credit Profile")
        if not reasons:
            return "General vector proximity"
        return ", ".join(reasons)

    def _generate_explanation(self, decision, neighbors, anomalies):
        if anomalies:
            return f"Flagged due to anomalies: {', '.join(anomalies)}"
        
        if not neighbors:
            return "No historical precedents found."

        approve_count = sum(1 for c, d in neighbors if c.get('decision') == 'approve')
        total = len(neighbors)
        
        return f"Based on {total} similar past cases ({approve_count} approved). Most similar case was {neighbors[0][0].get('decision')}."

    def learn(self, application_profile, final_decision, actual_outcome=None):
        """
        Adds the completed case to memory.
        """
        case = {
            'profile': application_profile,
            'decision': final_decision,
            'outcome': actual_outcome, 
            'labels': []
        }
        # In Qdrant version, valid ID is preferred.
        # We'll rely on UUID generation in Manager if not provided.
        self.memory.add_case(case)
