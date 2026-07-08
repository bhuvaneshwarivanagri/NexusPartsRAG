# 🔩 NexusPart

**Local RAG System for Industrial Supply Chains**

NexusPart is a fully local, GPU-optional AI system that performs semantic search over a proprietary industrial parts database and uses a locally-hosted LLM (via Ollama) to justify part substitution recommendations in plain English — no cloud APIs, no external data transmission.

## Features

- Semantic search over a parts catalog using HuggingFace sentence embeddings (`all-MiniLM-L6-v2`)
- Dual vector store support: FAISS (in-memory, fast) and ChromaDB (persistent, metadata-filterable)
- RAG pipeline: retrieve candidate parts → build a grounded prompt → generate a justified recommendation via a local LLM (Llama 3 / Mistral via Ollama)
- Automatic handling of missing/sparse part descriptions via synthesis from structured attributes
- EDA and evaluation visualizations (missing data, description coverage, similarity scores, PCA embedding space)
- Optional Streamlit dashboard for interactive search

## Project Structure

```
NexusPart_RAG_System.ipynb   # Main notebook — run top to bottom
Parts.csv                    # Input dataset (semicolon-delimited, you provide this)
Parts_cleaned.csv            # Generated: cleaned dataset with rich_text column
parts_embeddings.npy         # Generated: part embeddings
parts_faiss.index            # Generated: FAISS vector index
chroma_db/                   # Generated: ChromaDB persistent store
nexuspart_app.py             # Generated: Streamlit dashboard script
*.png                        # Generated: EDA and evaluation charts
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- At least one Ollama model pulled, e.g.:
  ```
  ollama pull llama3
  ```
- (Optional) NVIDIA GPU with CUDA, or Apple Silicon, for faster embedding generation. CPU-only works fine, just slower.

## Setup

1. Install dependencies:
   ```
   pip install sentence-transformers faiss-cpu chromadb ollama pandas numpy matplotlib seaborn
   pip install langchain langchain-community langchain-ollama
   pip install scikit-learn tqdm plotly torch streamlit
   ```

2. Start Ollama (if not already running as a background service):
   ```
   ollama serve
   ```

3. Place your parts data as `Parts.csv` in the project directory. Expected format: semicolon-separated, UTF-8, with at minimum an `ID` column and ideally a `DESCRIPTION` column plus attribute columns (`Application`, `Characteristic`, `Material`, `Mounting`, `Rated Current (A)`, `Rated Voltage (V)`, etc.).

## Usage

### Run the notebook

Open `NexusPart_RAG_System.ipynb` and run cells top to bottom:

1. **Phase 1** — installs libraries, checks hardware (CPU/GPU), verifies the Ollama connection and model availability.
2. **Phase 2** — loads and profiles `Parts.csv`, cleans text, synthesizes missing descriptions, generates embeddings, and builds the FAISS/ChromaDB indexes.
3. **Phase 3** — runs example RAG queries, visualizes retrieval quality, and writes the Streamlit dashboard script.

### Query the RAG pipeline directly

```python
result = nexuspart_rag("slow blow ceramic fuse 6.3A 250V for motor circuit", top_k=5)
print(result["recommendation"])
```

### Run the Streamlit dashboard

After the notebook has generated `Parts_cleaned.csv`, `parts_embeddings.npy`, and `parts_faiss.index`:

```
streamlit run nexuspart_app.py
```

This opens an interactive UI with a search tab (query + similarity chart + LLM recommendation) and a dataset explorer tab.

## Configuration

- `MODEL_NAME` (notebook cell 9): the Ollama model used for generation. Default `"llama3"`; change to `"mistral"` or `"llama3.2"` as needed — make sure it's pulled first.
- `EMBEDDING_MODEL` (notebook cell 24): the sentence-transformer model used for embeddings. Default `"all-MiniLM-L6-v2"`; `"BAAI/bge-small-en-v1.5"` is a slightly higher-quality alternative.

## Known Limitations

- Retrieval quality depends on how complete each part's description/attributes are — parts with very sparse data get a generic synthesized description and will retrieve less reliably.
- No re-ranking layer; retrieval relies purely on cosine similarity from the bi-encoder.
- The `ollama` Python client's `list()` response shape has changed across versions (dict-like vs. `Model` objects with a `.model` attribute) — if you see `❌ Ollama not running: 'name'` but Ollama is actually running, this is a client-version mismatch, not a connection failure.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | `torch` not installed in the active environment | `pip install torch`, and confirm your notebook kernel matches the Python environment where you installed it (`sys.executable`) |
| `Error: listen tcp 127.0.0.1:11434: bind: ...` on `ollama serve` | Ollama is already running as a background service | No action needed — check with `curl http://127.0.0.1:11434` |
| `❌ Ollama not running: 'name'` | `ollama.list()` returns `Model` objects (not dicts) in newer client versions | Use `[m.model for m in models.models]` instead of `[m['name'] for m in models.get('models', [])]` |

## Roadmap

- Fine-tune embeddings on domain-specific parts language
- Add a cross-encoder re-ranking layer
- Docker deployment
- User feedback loop for continuous improvement
- Per-row data-completeness scoring surfaced in retrieval and recommendations
