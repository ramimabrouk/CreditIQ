import sys
import os
import re
import json

# Add root to path
sys.path.append(os.getcwd())

from src.decision_engine import DecisionEngine

class CreditChatbot:
    def __init__(self):
        print("Initializing Logic Core...")
        self.engine = DecisionEngine()
        self.history_loaded = False
        
        # Simple parsing patterns
        self.key_map = {
            'income': 'income',
            'exp': 'expenses',
            'expense': 'expenses',
            'expenses': 'expenses',
            'emp': 'employment_length',
            'employment': 'employment_length',
            'loan': 'loan_amount',
            'amount': 'loan_amount',
            'term': 'loan_term',
            'credit': 'credit_history',
            'score': 'credit_history'
        }

    def start(self):
        print("\n" + "="*50)
        print("= Credit Decision Memory - Chat Interface")
        print("="*50)
        print("Connected to Qdrant Vector Memory.")
        print("Commands:")
        print(" - 'load': Load historical data to Qdrant")
        print(" - 'assess [details]': specific application (e.g., 'assess income=5000 credit=good')")
        print(" - 'search [id]': look up details of a case")
        print(" - 'quit': exit")
        print("--------------------------------------------------")

        while True:
            try:
                user_input = input("\nUser > ").strip()
                if not user_input:
                    continue
                
                response = self.process_input(user_input)
                print(f"Bot  > {response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def process_input(self, text):
        raw = text.lower()
        
        if raw == 'quit' or raw == 'exit':
            sys.exit(0)
            
        if 'load' in raw:
            return self.load_data()
            
        if raw.startswith('assess') or raw.startswith('eval'):
            return self.assess_application(text)
            
        if raw.startswith('search') or raw.startswith('find'):
            return self.search_case(text)

        return "I didn't understand that. Try 'assess income=5000' or 'load'."

    def load_data(self):
        if not os.path.exists("data/history.json"):
            return "❌ History file not found. Please run history_generator.py first."
        
        self.engine.load_history("data/history.json")
        stats = self.engine.memory.get_stats()
        self.history_loaded = True
        return f"✅ Knowledge Base Loaded. {stats['total_cases']} cases indexed in Qdrant."

    def assess_application(self, text):
        if not self.history_loaded:
            return "⚠️ Please 'load' data first."
            
        # Parse logic
        profile = self._parse_profile(text)
        if not profile:
            return "Could not extract profile data. Use format 'key=value' (e.g., income=5000)."
            
        # Run Engine
        result = self.engine.evaluate_application(profile)
        
        # Format output
        rec = result['recommendation']
        conf = result['confidence']
        expl = result['explanation']
        
        output = [f"Assessment: {rec} ({int(conf*100)}% Confidence)"]
        output.append(f"Reasoning: {expl}")
        
        if result['anomalies']:
            output.append(f"⚠️ Anomalies: {', '.join(result['anomalies'])}")
            
        output.append("\nReference Cases:")
        for c in result['similar_cases'][:2]:
            output.append(f" - [ID {c['id']}] {c['decision'].upper()} (Match: {int((1-c['distance'])*100)}%) - {c['match_reason']}")
            
        return "\n".join(output)

    def search_case(self, text):
        # Implementation for looking up a case by ID isn't directly supported by my Memory class 
        # (which is search-only). 
        # But we can do a mock search or just say "Not implemented".
        # Let's support vector search that finds "Cases like ID X" if we could...
        # For now, just a placeholder.
        return "Search by ID is not yet implemented in this prototype."

    def _parse_profile(self, text):
        """
        Extracts key=value pairs.
        Default keys: income, expenses, employment_length, loan_amount, loan_term, credit_history
        """
        profile = {
            'income': 0, 'expenses': 0, 'employment_length': 0, 
            'loan_amount': 0, 'loan_term': 12, 'credit_history': 'fair' 
        }
        
        # Regex for key=value or key:value
        matches = re.findall(r'(\w+)[=:](\w+)', text)
        
        found_any = False
        for k, v in matches:
            k_norm = self.key_map.get(k.lower())
            if k_norm:
                found_any = True
                if k_norm == 'credit_history':
                    profile[k_norm] = v
                else:
                    try:
                        profile[k_norm] = int(v)
                    except:
                        pass
        
        return profile if found_any else None

if __name__ == "__main__":
    bot = CreditChatbot()
    bot.start()
