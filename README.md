# LUMINA-DS: Local Instagram Discovery Engine

LUMINA-DS is a completely **100% free and local** Retrieval-Augmented Generation (RAG) system built to query and synthesize vast amounts of Instagram data. It features a modern, glassmorphism-inspired web interface, lightning-fast hybrid search, and local LLM synthesis.

## ✨ Features
- **100% Local**: No API keys, no paid services. Uses open-source local models for embeddings and LLM generation.
- **Hybrid Vector Search**: Combines Dense vectors (`all-MiniLM-L6-v2`) and Sparse vectors (`HashingVectorizer`) using Reciprocal Rank Fusion (RRF) for highly accurate retrieval.
- **Milvus Lite**: Lightweight, serverless local vector database for storing massive amounts of text chunks.
- **Ollama LLM Integration**: Uses `llama3.2:1b` (via Ollama) to read retrieved context and synthesize human-readable, conversational answers.
- **Sleek Frontend UI**: A dark-mode, single-page application built with HTML/CSS/JS, featuring dynamic Instagram image fetching directly from compressed local zip files!

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: Download and install from [ollama.com](https://ollama.com).
- **Node/NPM (Optional)**: If you want to use external tools, but the frontend runs purely on Python's HTTP Server.

### 2. Pull the Local LLM
Ensure Ollama is running, then pull the required Llama 3.2 model:
```bash
ollama pull llama3.2:1b
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Ingestion
You must ingest the raw Instagram data into the Milvus vector database before searching. The system reads directly from the downloaded `.zip` and `.txt` files in `data/raw/`.
```bash
python ingest.py --limit 5000
```
*(Note: You can omit the limit to ingest the full dataset, but it will take time!)*

### 5. Start the Servers
LUMINA-DS requires two servers to run: the FastAPI backend and the static frontend server.

**Start the Backend (API)**:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Start the Frontend (UI)**:
Open a new terminal and run:
```bash
python -m http.server 5500 -d frontend
```

### 6. Use the App!
Navigate to [http://localhost:5500](http://localhost:5500) in your web browser. Type a query (e.g., "healthy dinner recipes"), adjust the minimum likes slider, and watch the local LLM synthesize an answer using real Instagram posts—complete with original images loaded directly from your local disk!

---

## 🛠 Architecture Details

- **Backend**: FastAPI (`app.py`), handles the `/search` and `/image/{post_id}` endpoints.
- **Retrieval Engine**: PyMilvus (`milvus_lumina.db`). The `LuminaRetriever` in `retrieval.py` fetches the top 60 candidates and reranks them locally using `ms-marco-MiniLM-L-6-v2`.
- **Frontend**: A vanilla JavaScript app (`frontend/app.js`) that uses the `Fetch API` to talk to the backend, parses markdown via `marked.js`, and uses dynamic modal overlays for image viewing.

## 🤝 Contributing
Contributions are welcome to optimize vector indexing, improve frontend design, or experiment with different Ollama models!
