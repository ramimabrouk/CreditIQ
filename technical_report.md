# Technical Report: Credit Decision Memory Engine

## 1. Idea and Problem Statement
Traditional credit decisioning systems often rely on rigid, rule-based logic or opaque "black box" machine learning models. While effective, they lack **explainability** and **nuance**. A human underwriter makes decisions based on experience—remembering similar past cases ("This looks like Client X from last year").

**Problem:** How can we automate credit decisions while retaining the "human-like" ability to recall precedents and provide clear, example-based explanations?

**Solution:** The **Credit Decision Memory Engine** is a prototype that uses **Vector Similarity Search** to create a system with "memory". Instead of just predicting a probability, it retrieves the top-k most similar historical applications to justify its decision, offering transparency and auditability.

## 2. Architecture & Qdrant Integration
The system is built as a modular Python application with the following core components:

### Components
*   **Similarity Engine (`similarity.py`)**: Converts raw application data (Income, Expenses, Credit History) into a normalized numerical vector.
*   **Memory Store (`qdrant_manager.py`)**: Uses **Qdrant** (running involved in-memory for this prototype) to store and index historical application vectors.
*   **Decision Engine (`decision_engine.py`)**: The orchestrator that:
    1.  Vectorizes the new application.
    2.  Queries Qdrant for neighbors.
    3.  Aggregates the outcomes of neighbors to form a recommendation.
*   **Anomaly Detector (`anomaly.py`)**: Checks if the new application is too distant from *any* known case, flagging it for manual review.

### Qdrant Integration
**Qdrant** is the core "Hippocampus" of the system.
*   **Vector Config**: Uses Cosine distance to measure similarity between application profiles.
*   **Payloads**: Stores the full metadata (Result: Approve/Reject, Reason, ID) alongside the vectors.
*   **Search**: Performs efficient Approximate Nearest Neighbor (ANN) search to retrieve the "Evidence" for every decision.

## 3. Data Pipeline
The data flow is designed to be continuous, allowing the system to learn from new decisions (in a full production environment).

1.  **Ingestion/Generation**:
    *   Synthethic data is generated (`history_generator.py`) to simulate a history of 1000+ loan applications with varying profiles.
2.  **Vectorization**:
    *   Categorical data (e.g., "Employed", "Self-Employed") is encoded.
    *   Numerical data (Income, Loan Amount) is normalized.
    *   Result: A dense vector representing the applicant's financial DNA.
3.  **Indexing**:
    *   Vectors are upserted into the Qdrant collection `credit_memory`.
4.  **Inference (The Decision)**:
    *   New Application -> Vector -> Qdrant Query -> Top-5 Neighbors -> Weighted Vote -> Decision.

## 4. Project Timeline

### Work Done
*   [x] **Core Logic**: Implemented vector normalization and weighting algorithms.
*   [x] **Qdrant Integration**: Fully integrated `QdrantClient` for vector storage and retrieval.
*   [x] **CLI Demo**: Created an interactive chatbot to demonstrate the "Apply -> Explain" flow.
*   [x] **Anomaly Detection**: Basic distance-based outlier detection.

### Future Work
*   [ ] **Web Interface**: Build a React/Next.js dashboard for loan officers to visualize the "Neighbor Applications" graph.
*   [ ] **Persistent Storage**: Move from Qdrant In-Memory to a persistent Docker/Cloud instance.
*   [ ] **Real Data**: Replace synthetic generators with anonymized real-world credit data.
*   [ ] **LLM Integration**: Use a Local LLM to generate natural language summaries of the decision based on the retrieved neighbors.
