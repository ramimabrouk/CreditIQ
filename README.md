# Credit Decision Memory Engine (Prototype)

An explainable, similarity-driven credit decisioning system that uses historical "memory" to assess risk, detect anomalies, and provide evidence-based recommendations.

## 🧠 Memory-Based Decisioning?

Traditional systems use rigid rules or opaque black-box models. This engine works like a human underwriter with a perfect memory:
1. **Recalls** the most similar past applications.
2. **Compares** the current outcome to those historical precedents.
3. **Explains** the decision by citing the specific past cases ("Approved because it looks like Case #124 and #98 which repaid successfully").
4. **Detects Anomalies** if an application looks nothing like what we've seen before.

## 🚀 Key Features

- **Multi-Factor Profiling**: Combines income, expenses, loan details, and credit history into a unified vector.
- **Top-K Retrieval**: instantly finds the nearest historical analogs using cosine/euclidean distance.
- **Anomaly Detection**: Flags applications that are statistical outliers or match known fraud patterns.
- **Audit Trail**: Every decision returns the "Evidence" (IDs of cases used to make the decision).
- **Learning Loop**: The system gets smarter as more decisions are finalized and fed back into memory.

## 🏗️ Architecture
This project uses **Qdrant** as its core memory store.
- **Vectors**: All applications are converted into high-dimensional vectors.
- **Search**: We use Qdrant's similarity search to find the "Nearest Neighbors" in the vector space.
- **Decision**: The decision is a weighted vote of these neighbors.


## 📂 Project Structure

```
src/
├── similarity.py       # Vectorization logic (Income -> normalized vector)
├── memory.py           # In-memory vector store & Top-K retrieval
├── anomaly.py          # Outlier detection & 'Fraud' pattern matching
└── decision_engine.py  # Orchestrator: Profile -> Vector -> Memory -> Decision

data/
└── history.json        # Synthetic historical dataset (generated)

demo/
└── run_demo.py         # End-to-end CLI demonstration script
```

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.8+
- `numpy`, `scikit-learn`

### Installation
```bash
pip install -r requirements.txt
```

### Running the Demo
1. **Generate Data** (Creates synthetic history)
   ```bash
   python data/history_generator.py
   ```

3. **Run the Chatbot (Interactive)**
   ```bash
   python src/chatbot.py
   ```
   *Interact naturally: "Assess income=15000 credit=good"*

## 📊 Example Output

```text
Application: {'income': 12000, 'credit_history': 'good', ...}

>>> Recommendation: APPROVE (Confidence: 1.0)
Explanation: Based on 5 similar past cases (5 approved). Most similar case was approve.
Similar Cases:
  - Case 42 (approve) [Dist: 0.12]: Similar Income, Same Credit Profile
  - Case 15 (approve) [Dist: 0.15]: Similar Income, Same Credit Profile
```

## 🔮 Future Improvements

- **Embeddings Model**: Replace manual vectorization with a trained Neural Network (e.g., TabTransformer) or LLM embeddings for documents.
- **Vector Database**: Migrate from in-memory list to Milvus/Pinecone for million-scale retrieval.
- **Document Analysis**: Integrate OCR to actually parse uploaded bank statements instead of stubbing them.
