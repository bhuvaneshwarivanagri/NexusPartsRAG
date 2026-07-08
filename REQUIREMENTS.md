# NexusPart — Requirements Document

## 1. Project Overview

NexusPart is a fully local, offline Retrieval-Augmented Generation (RAG) system for industrial supply chain parts management. It performs semantic search over a proprietary parts catalog and uses a locally-hosted LLM to justify part substitution recommendations in natural language, with no cloud APIs or external data transmission.

## 2. Objectives

- Enable semantic (meaning-based) search over an industrial parts database, going beyond keyword matching.
- Generate natural-language, justified part substitution recommendations using a local LLM.
- Keep the entire pipeline — embeddings, vector search, and generation — running on local hardware, for data privacy in a supply chain context.

## 3. Functional Requirements

### 3.1 Data Ingestion & Cleaning
- Load a semicolon-delimited CSV (`Parts.csv`) of parts records.
- Profile missing data (count and percentage per column).
- Normalize and clean free-text description fields (lowercasing, whitespace/character normalization).
- Synthesize a fallback description for parts with no `DESCRIPTION` value, built from structured attribute columns.
- Persist the cleaned dataset (`Parts_cleaned.csv`).

### 3.2 Embedding Generation
- Combine each part's description and key structured attributes into a single "rich text" representation per row.
- Generate dense vector embeddings for each part using a HuggingFace sentence-transformer model (default: `all-MiniLM-L6-v2`).
- Normalize embeddings for cosine-similarity search.
- Persist embeddings to disk (`parts_embeddings.npy`).

### 3.3 Vector Indexing & Retrieval
- Build a FAISS `IndexFlatIP` index over the embeddings for fast top-K similarity search.
- Optionally build an equivalent ChromaDB persistent collection with metadata filtering (Material, Application, Characteristic, Mounting, Size).
- Given a natural-language query, return the top-K most semantically similar parts with similarity scores.

### 3.4 RAG Recommendation Generation
- Construct an augmented prompt combining the user's query with the retrieved candidate parts and their specs.
- Send the prompt to a locally-hosted LLM via Ollama (default model: `llama3`) to produce a structured recommendation: Best Match → Justification → Caveats.
- Gracefully degrade (return a placeholder message) if the local LLM is unavailable, rather than raising an unhandled exception.

### 3.5 Evaluation & Visualization
- Report missing-value statistics and description-coverage statistics.
- Visualize similarity score distributions per query.
- Run a fixed batch of evaluation queries and report mean top-1 / top-5 similarity as a coarse retrieval-quality metric.
- Visualize the embedding space in 2D via PCA, colored by part characteristic.

### 3.6 User Interface
- Provide an interactive Streamlit dashboard (`nexuspart_app.py`) with:
  - A search tab for entering a query and viewing retrieved parts, similarity chart, and LLM recommendation.
  - A dataset explorer tab with summary metrics and filtering by part characteristic.

## 4. Non-Functional Requirements

- **Locality / Privacy:** No part data or query text may be sent to an external/cloud API. All embedding and generation must run on local hardware via Ollama and HuggingFace models.
- **Portability:** Must run on Windows, macOS, and Linux; must detect and use CUDA or Apple MPS acceleration when available, and fall back to CPU otherwise.
- **Reproducibility:** Cleaned data, embeddings, and the FAISS index must be persisted to disk so the pipeline does not need to be fully re-run for each query session.
- **Usability:** Errors (e.g., Ollama not running, model not pulled) must produce a clear, actionable message rather than a raw stack trace where reasonably possible.

## 5. System Requirements

| Component | Requirement |
|---|---|
| OS | Windows 10+, macOS, or Linux |
| Python | 3.11+ |
| Ollama | Installed and running locally (`ollama serve`); at least one model pulled (e.g., `ollama pull llama3`) |
| GPU (optional) | CUDA-capable NVIDIA GPU or Apple Silicon (MPS); CPU-only supported but slower |
| Disk | Sufficient space for the parts CSV, embeddings (`.npy`), FAISS index, and optional ChromaDB persistent store |

## 6. Software Dependencies

```
sentence-transformers
faiss-cpu
chromadb
ollama
pandas
numpy
matplotlib
seaborn
langchain
langchain-community
langchain-ollama
scikit-learn
tqdm
plotly
torch          # required by sentence-transformers; not always auto-installed
streamlit      # required only to run the bonus dashboard
```

## 7. Data Requirements

- Input file: `Parts.csv`, semicolon-separated (`sep=";"`), UTF-8 encoded.
- Expected/used columns include (not all are required, but pipeline quality depends on their completeness):
  - `ID` (unique part identifier — required for indexing/lookup)
  - `DESCRIPTION` (primary free-text field)
  - `Application`, `Characteristic`, `Material`, `Mounting`, `Attribut1`, `Additional Feature`, `Size`
  - `Rated Current (A)`, `Rated Voltage (V)`
- Rows missing all of the above fields still get a minimal synthesized identifier (`"part id: <ID>"`) so no row is dropped from the index, but retrieval quality for such rows will be weak.

## 8. Constraints & Known Limitations

- Retrieval quality is only as good as the completeness of `DESCRIPTION` and the attribute columns feeding `rich_text`; no per-row confidence/completeness score is currently computed.
- No re-ranking (e.g., cross-encoder) stage — retrieval relies solely on bi-encoder cosine similarity.
- No fine-tuning of the embedding model on domain-specific parts language.
- Single-user, single-machine design; no authentication, multi-tenancy, or concurrent-write handling.
- Ollama client library versions differ in API shape (`ollama.list()` output changed between versions) — code should be pinned or defensively written against the installed client version.

## 9. Out of Scope

- Cloud deployment or hosted API access.
- Multi-user access control or authentication.
- Automated procurement/ordering actions (recommendations are advisory only).
- Non-English part descriptions (not tested/handled).

## 10. Future Enhancements

- Fine-tune embeddings on domain-specific industrial parts data.
- Add a cross-encoder re-ranking layer on top of FAISS/ChromaDB retrieval.
- Containerize with Docker for consistent deployment.
- Add a user feedback loop to improve recommendation quality over time.
- Add a per-row data-completeness score surfaced in both retrieval results and the LLM prompt.
