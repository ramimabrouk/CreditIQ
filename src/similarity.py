import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

class SimilarityEngine:
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        # Simple predefined schema for the prototype
        # Features: [Income, Expenses, Employment_Length, Loan_Amount, Loan_Term]
        self.feature_map = {
            'income': 0,
            'expenses': 1,
            'employment_length': 2,
            'loan_amount': 3,
            'loan_term': 4
        }
        # In a real system, we'd fit this on the whole dataset. 
        # Here we'll do a "partial fit" approximation or just simple normalization.
        self.is_fitted = False

    def _extract_numerical_vector(self, profile):
        """Extracts fixed numerical features in specific order."""
        vec = [
            profile.get('income', 0),
            profile.get('expenses', 0),
            profile.get('employment_length', 0),
            profile.get('loan_amount', 0),
            profile.get('loan_term', 0)
        ]
        return np.array(vec, dtype=float).reshape(1, -1)

    def vectorize(self, profile):
        """
        Converts a profile dict into a normalized feature vector.
        For this prototype, we'll use simple min-max scaling loosely based on expected ranges
        to avoid complexity of stateful scalers for the first pass.
        """
        # Manual normalization logic for control/simplicity in prototype
        # ranges: income(0-20k), exp(0-10k), emp(0-20), amt(0-50k), term(12-60)
        
        income_norm = min(profile.get('income', 0) / 20000.0, 1.0)
        exp_norm = min(profile.get('expenses', 0) / 10000.0, 1.0)
        emp_norm = min(profile.get('employment_length', 0) / 20.0, 1.0)
        amt_norm = min(profile.get('loan_amount', 0) / 50000.0, 1.0)
        term_norm = min(profile.get('loan_term', 12) / 60.0, 1.0)
        
        # Credit History Score (0-1 usually, mapping bad->0, good->1)
        # We process this simply: bad=0, fair=0.5, good=1.0
        cred_map = {'bad': 0.0, 'poor': 0.2, 'fair': 0.5, 'good': 0.8, 'excellent': 1.0}
        credit_score = cred_map.get(profile.get('credit_history', 'fair'), 0.5)

        # Vector: [Inc, Exp, Emp, Amt, Term, Credit]
        vector = np.array([income_norm, exp_norm, emp_norm, amt_norm, term_norm, credit_score])
        
        # Add mock document embedding (3 dims)
        # In real world: these come from an embedding model (BERT/ResNet)
        doc_embedding = np.random.normal(0, 0.1, 3) 
        
        return np.concatenate([vector, doc_embedding])

    def calculate_distance(self, vec_a, vec_b):
        """Euclidean distance betweeen two vectors."""
        return np.linalg.norm(vec_a - vec_b)

    def calculate_similarity(self, vec_a, vec_b):
        """Cosine similarity."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)
