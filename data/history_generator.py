import json
import random
import os

def generate_profile(case_id):
    """Generates a realistic-ish loan profile."""
    # Profiles:
    # 1. Safe: High Income, Good Credit, Low Loan
    # 2. Risky: Low Income, Bad Credit, High Loan
    # 3. Average: Mixed
    
    category = random.choices(['safe', 'risky', 'average', 'fraud'], weights=[0.4, 0.3, 0.25, 0.05])[0]
    
    profile = {}
    decision = "decline"
    labels = []
    
    if category == 'safe':
        profile = {
            'income': random.randint(6000, 15000),
            'expenses': random.randint(1000, 3000),
            'employment_length': random.randint(3, 10),
            'loan_amount': random.randint(5000, 20000),
            'loan_term': random.choice([12, 24, 36]),
            'credit_history': 'good'
        }
        decision = "approve"
        
    elif category == 'risky':
        profile = {
            'income': random.randint(2000, 4000),
            'expenses': random.randint(1500, 3000),
            'employment_length': random.randint(0, 2),
            'loan_amount': random.randint(10000, 40000),
            'loan_term': random.choice([36, 48, 60]),
            'credit_history': 'fair' if random.random() > 0.5 else 'bad'
        }
        decision = "decline"

    elif category == 'average':
        profile = {
            'income': random.randint(4000, 7000),
            'expenses': random.randint(2000, 4000),
            'employment_length': random.randint(1, 4),
            'loan_amount': random.randint(5000, 15000),
            'loan_term': 24,
            'credit_history': 'fair'
        }
        # T Toss
        decision = "approve" if profile['income'] > 5000 else "decline"

    elif category == 'fraud':
        # Fraud lookalikes: seemingly ok but specific weird patterns (captured in vector space hopefully)
        profile = {
            'income': random.randint(8000, 12000),
            'expenses': 500, # Unrealistic low expenses
            'employment_length': 0,
            'loan_amount': 45000, # Max out
            'loan_term': 12, # Short term
            'credit_history': 'good' # Stolen profile maybe
        }
        decision = "decline"
        labels = ["fraud"]

    return {
        "id": case_id,
        "profile": profile,
        "decision": decision,
        "outcome": "repaid" if decision == "approve" and random.random() > 0.1 else "default",
        "labels": labels
    }

def main():
    data = []
    for i in range(100):
        data.append(generate_profile(i))
        
    os.makedirs("data", exist_ok=True)
    with open("data/history.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} cases in data/history.json")

if __name__ == "__main__":
    main()
