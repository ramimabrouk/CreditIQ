import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

from src.decision_engine import DecisionEngine

def print_header(text):
    print(f"\n{'='*50}\n{text}\n{'='*50}")

def print_section(text):
    print(f"\n--- {text} ---")

def main():
    print_header("Credit Decision Memory Engine - Prototype Demo")

    # 1. Initialize
    print_section("Initializing System")
    engine = DecisionEngine()
    
    # 2. Load History
    if not os.path.exists("data/history.json"):
        print("Error: data/history.json not found. Run generator first.")
        return

    print("Loading historical data...")
    engine.load_history("data/history.json")
    stats = engine.memory.get_stats()
    print(
    f"Memory Loaded: {stats.get('total_cases', 'N/A')} cases "
    f"({stats.get('approved', stats.get('approvals', 'N/A'))} approved, "
    f"{stats.get('declined', stats.get('declines', 'N/A'))} declined)"
)

    # 3. Simulate New Application (Safe)
    print_header("Scenario 1: Safe Applicant (Similar to past approvals)")
    safe_app = {
        'income': 12000,
        'expenses': 2500,
        'employment_length': 5,
        'loan_amount': 10000,
        'loan_term': 24,
        'credit_history': 'good'
    }
    print(f"Application: {safe_app}")
    
    result = engine.evaluate_application(safe_app)
    print(f"\n>>> Recommendation: {result['recommendation']} (Confidence: {result['confidence']})")
    print(f"Explanation: {result['explanation']}")
    print("Similar Cases:")
    for case in result['similar_cases'][:2]:
        print(f"  - Case {case['id']} ({case['decision']}) [Dist: {case['distance']}]: {case['match_reason']}")

    # 4. Simulate Risky Application
    print_header("Scenario 2: Risky Applicant (Similar to past declines/defaults)")
    risky_app = {
        'income': 2500,
        'expenses': 2000,
        'employment_length': 0,
        'loan_amount': 25000,
        'loan_term': 60,
        'credit_history': 'bad'
    }
    print(f"Application: {risky_app}")
    
    result = engine.evaluate_application(risky_app)
    print(f"\n>>> Recommendation: {result['recommendation']} (Confidence: {result['confidence']})")
    print(f"Explanation: {result['explanation']}")

    # 5. Anomaly / Fraud Attempt
    print_header("Scenario 3: Anomaly / Potential Fraud")
    fraud_app = {
        'income': 10000,
        'expenses': 400, # Unrealistically low
        'employment_length': 0,
        'loan_amount': 50000,
        'loan_term': 12,
        'credit_history': 'good'
    }
    print(f"Application: {fraud_app}")
    result = engine.evaluate_application(fraud_app)
    
    print(f"\n>>> Recommendation: {result['recommendation']}")
    if result['anomalies']:
        print(f"!!! ANOMALIES DETECTED: {result['anomalies']}")
    print(f"Explanation: {result['explanation']}")

    # 6. Learning Loop
    print_header("Scenario 4: Learning Loop (Feedback)")
    print("Simulating manual override: Approving a borderline case and feeding it back...")
    
    borderline_app = {
        'income': 5000,
        'expenses': 4000,
        'employment_length': 2,
        'loan_amount': 5000,
        'loan_term': 12,
        'credit_history': 'fair'
    }
    
    # First check
    print("Initial check:")
    res1 = engine.evaluate_application(borderline_app)
    print(f"System says: {res1['recommendation']}")
    
    # Teach
    print("\n[ACTION] Underwriter approves case manually with 'repaid' outcome.")
    engine.learn(borderline_app, final_decision='approve', actual_outcome='repaid')
    
    # Check again (similar case)
    print("\nChecking a new identical case after learning:")
    res2 = engine.evaluate_application(borderline_app)
    print(f"System says: {res2['recommendation']} (Confidence increased or changed based on neighbor)")
    print(f"Explanation: {res2['explanation']}")

if __name__ == "__main__":
    main()
