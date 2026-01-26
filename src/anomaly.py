import numpy as np

class AnomalyDetector:
    def __init__(self, memory_engine):
        self.memory = memory_engine

    def check_hard_rules(self, profile):
        """Returns list of rule violations."""
        issues = []
        if profile.get('income', 0) < 0:
            issues.append("Negative Income")
        if profile.get('employment_length', 0) > 60: 
            issues.append("Suspicious Employment Length")
        return issues

    def detect_outliers(self, input_vector, neighbors):
        """
        Checks if the input is significantly far from its nearest neighbors.
        """
        if not neighbors:
            return []
            
        # Distance to nearest neighbor
        nearest_dist = neighbors[0][1]
        
        # Threshold need adjustment for Cosine Distance (0 to 2) vs Euclidean.
        # Sim = 1.0 -> Dist = 0.0. Sim = 0.5 -> Dist = 0.5.
        # 'Good' similarity usually > 0.8 => Dist < 0.2.
        # So if Dist > 0.3 or 0.4, it's getting far.
        
        if nearest_dist > 0.35:
            return [f"High Dissimilarity (Nearest Dist: {nearest_dist:.2f})"]
            
        return []

    def check_fraud_similarity(self, input_vector, threshold=0.15):
        """
        Checks if input allows clearly matches known fraud cases using the memory.
        """
        # Use the specific method in memory (which might start using filters later)
        fraud_neighbors = self.memory.retrieve_fraud_cases(input_vector, k=1)
        
        if fraud_neighbors:
            fraud_case, dist = fraud_neighbors[0]
            if dist < threshold:
                return [f"Matches Known Fraud Pattern (Case #{fraud_case.get('id', '?')}, Dist: {dist:.2f})"]
        
        return []

    def analyze(self, profile, input_vector, neighbors):
        anomalies = []
        anomalies.extend(self.check_hard_rules(profile))
        anomalies.extend(self.detect_outliers(input_vector, neighbors))
        anomalies.extend(self.check_fraud_similarity(input_vector))
        return anomalies
