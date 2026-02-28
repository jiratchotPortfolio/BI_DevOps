# Sentinel: RAG-Based Customer Insight Engine

**A production-grade Retrieval-Augmented Generation system for automated customer conversation intelligence.**

---

## Project Overview

Enterprise customer experience teams are routinely overwhelmed by unstructured conversation data. Support tickets, live chat transcripts, and escalation logs accumulate at scale, yet the institutional knowledge embedded within them remains largely inaccessible. Analysts are forced to rely on manual sampling, anecdotal reasoning, and lagged reporting cycles — all of which introduce latency and bias into decision-making.

Sentinel addresses this problem directly. It is an end-to-end data intelligence platform that ingests raw customer conversation logs, structures and persists them in a hybrid storage layer, and exposes a context-aware Q&A interface powered by Retrieval-Augmented Generation (RAG). Rather than relying on a language model's parametric memory — which is inherently static and prone to hallucination — Sentinel grounds every generated answer in retrieved, verifiable source documents from its own data corpus.

The result is a self-improving intelligence layer that enables Customer Experience (CX) teams to query months of conversation history in natural language, identify emerging issue clusters, and extract actionable insight without writing a single SQL query manually.

---

## System Architecture

Sentinel is composed of three discrete layers: Ingestion, Storage, and Logic. Data flows unidirectionally from raw log files through structured and vector storage, ultimately being consumed by the RAG inference engine.

```
Raw Logs (.jsonl / .csv)
        |
        v
+-------------------+
|  Ingestion Layer  |  (Python: data_ingestion.py)
|  - Parsing        |
|  - NLP Cleaning   |
|  - Validation     |
+-------------------+
        |
   +----+----+
   |         |
   v         v
+--------+  +------------------+
|  PgSQL |  |  Vector DB       |
| (Meta) |  |  (FAISS/Chroma)  |
+--------+  +------------------+
   |         |
   +----+----+
        |
        v
+-------------------+
|   Logic Layer     |  (LangChain RAG Chain)
|  - Retriever      |
|  - Prompt Engine  |
|  - LLM (OpenAI)  |
+-------------------+
        |
        v
  Q&A API / CLI Interface
```

### Ingestion Layer

The ingestion layer is implemented as a Python batch pipeline (`src/data_ingestion.py`) that reads raw conversation logs from a configurable source directory. Logs arrive as newline-delimited JSON (`.jsonl`) or CSV exports from CRM systems such as Zendesk or Salesforce.

Each record passes through a pre-processing NLP pipeline before being written to storage. This design decision — to clean prior to embedding — is deliberate: embedding noisy or duplicative text produces low-quality vector representations that degrade retrieval precision. The pipeline handles HTML entity decoding, PII redaction via regex patterns, boilerplate removal (e.g., repeated agent signatures), and language detection for filtering.

### Storage Layer: Hybrid Architecture

Sentinel adopts a hybrid storage strategy that separates structured metadata from semantic content. This mirrors established data warehousing principles for managing heterogeneous data types.

**PostgreSQL (Structured Metadata)**

Relational metadata — conversation IDs, agent IDs, timestamps, resolution status, CSAT scores, and topic classifications — is persisted in a normalized PostgreSQL schema (see `sql/schema.sql`). This enables high-performance analytical queries: filtering by date range, joining on agent performance metrics, or aggregating by issue category. Indexed foreign key relationships allow the retrieval layer to hydrate results with full structured context after a vector search completes.

**FAISS / ChromaDB (Vector Embeddings)**

The semantic content of each conversation turn is embedded using OpenAI's `text-embedding-ada-002` model (or a local HuggingFace equivalent for air-gapped deployments). These dense vector representations are indexed in FAISS for approximate nearest-neighbor (ANN) search, or ChromaDB for environments requiring persistent, metadata-filtered vector retrieval.

FAISS was selected as the primary vector index for its throughput characteristics at scale: sub-millisecond retrieval across millions of vectors using IVF (Inverted File Index) quantization. ChromaDB is offered as an alternative for use cases requiring native metadata filtering without a separate SQL join.

### Logic Layer: LangChain RAG Implementation

The RAG chain is orchestrated using LangChain. At query time, the user's natural language question is embedded and used to retrieve the top-k most semantically similar conversation chunks from the vector index. These chunks, combined with structured metadata retrieved via SQL join, are injected into a prompt template that instructs the LLM to answer strictly from the provided context.

This architecture explicitly reduces hallucination: the LLM is constrained to synthesize from retrieved evidence rather than generating from parametric memory. If the retrieved context does not contain a sufficient answer, the system is configured to return a structured "insufficient evidence" response rather than a fabricated one.

---

## Key Technical Features

### Scalable Data Pipelines

The ingestion pipeline supports both streaming and batch execution modes. In batch mode, `data_ingestion.py` processes log files in configurable chunk sizes using Python generators, preventing memory exhaustion when handling multi-gigabyte log archives. Each batch is committed transactionally to PostgreSQL, ensuring exactly-once delivery semantics on retry. The pipeline exposes Prometheus-compatible metrics for monitoring throughput, error rates, and embedding latency.

### Optimized SQL Queries

The PostgreSQL schema is designed with query performance as a primary constraint. Composite B-tree indexes are defined on `(conversation_date, topic_id)` and `(agent_id, resolution_status)` to support the most frequent analytical access patterns. Full-text search indexes (`GIN` on `tsvector` columns) enable keyword-based conversation search as a complement to semantic vector retrieval.

Complex multi-table joins — such as linking conversation metadata to agent performance records and product taxonomy — are pre-computed as materialized views and refreshed on a configurable schedule, reducing query latency for dashboard consumers from seconds to milliseconds.

### Context-Aware Retrieval

The RAG retrieval step is not a naive top-k vector search. Sentinel implements a two-stage retrieval strategy:

1. **Coarse retrieval**: ANN search over the FAISS index returns the top 20 candidate chunks based on cosine similarity.
2. **Re-ranking**: A cross-encoder re-ranker (e.g., `ms-marco-MiniLM-L-6-v2`) scores each candidate against the original query and selects the final top-5 chunks for context injection.

This approach significantly improves precision over single-stage retrieval, particularly for queries that are semantically broad but topically specific. The SQL metadata layer further filters candidates by date range or product category when structured constraints are present in the query.

---

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Core Language | Python 3.11 | Ecosystem maturity for ML/data pipelines |
| Relational Database | PostgreSQL 15 | ACID compliance, advanced indexing, JSON support |
| Vector Index | FAISS / ChromaDB | High-throughput ANN search; persistent metadata filtering |
| RAG Orchestration | LangChain 0.2 | Modular chain composition; retriever abstraction |
| Embedding Model | OpenAI `text-embedding-ada-002` / HuggingFace `sentence-transformers` | Configurable for cost vs. air-gap requirements |
| LLM Inference | OpenAI GPT-4o / HuggingFace Inference API | Swappable via LangChain LLM interface |
| Containerization | Docker + Docker Compose | Reproducible environments; service isolation |
| Data Validation | Pydantic v2 | Schema enforcement at ingestion boundary |
| Testing | pytest + testcontainers | Integration tests against ephemeral Postgres instances |

---

## Challenges and Solutions

### Challenge 1: Noisy and Inconsistent Conversation Logs

Raw CRM exports contain significant noise: HTML markup from rich-text editors, repeated boilerplate (legal disclaimers, agent signatures), encoding errors, and inconsistent field naming across system versions. Embedding this noise directly produces vector representations that are semantically incoherent, degrading retrieval quality.

**Solution**: A dedicated NLP pre-processing pipeline (`src/preprocessing_pipeline.py`) was implemented as the first stage of ingestion. It applies sequential transformations: HTML stripping via `BeautifulSoup`, regex-based boilerplate removal against a configurable blocklist, Unicode normalization, and language detection via `langdetect` to filter non-English records (configurable). Each transformation step is logged, and records that fail validation are written to a dead-letter queue for manual review rather than silently discarded.

### Challenge 2: Embedding Cost and Latency at Scale

Embedding millions of conversation turns against the OpenAI API incurs both financial cost and network latency. A naive implementation that embeds each record individually would be prohibitively slow.

**Solution**: The embedding pipeline batches records in groups of 512 (the maximum supported by the OpenAI Embeddings API) and uses asynchronous HTTP requests via `httpx` with a bounded concurrency semaphore. Embeddings are cached by a SHA-256 hash of the pre-processed text, ensuring that re-ingestion of unchanged records does not trigger redundant API calls. For cost-sensitive deployments, a local HuggingFace `sentence-transformers` model is offered as a zero-cost alternative with configurable model selection.

### Challenge 3: Context Window Management in RAG Prompts

Injecting multiple retrieved chunks into a prompt risks exceeding the LLM's context window, particularly when source conversations are verbose. Truncating chunks naively at a fixed character limit breaks semantic coherence.

**Solution**: Chunks are split at the sentence boundary using `nltk.sent_tokenize` during ingestion, with a configurable maximum token budget per chunk enforced via the `tiktoken` library. At retrieval time, the prompt builder dynamically calculates the total token count of selected chunks and trims from lowest-ranked to highest-ranked until the budget constraint is satisfied, preserving the most relevant context.

---

## Repository Structure

```
sentinel/
|-- src/
|   |-- data_ingestion.py          # Batch log ingestion pipeline
|   |-- preprocessing_pipeline.py  # NLP cleaning and normalization
|   |-- vector_embedding_manager.py # Embedding generation and FAISS indexing
|   |-- rag_chain.py               # LangChain RAG chain definition
|   |-- db_client.py               # PostgreSQL connection and query layer
|   |-- config.py                  # Environment-based configuration
|-- sql/
|   |-- schema.sql                 # Full relational schema with indexes
|   |-- queries/
|       |-- agent_performance.sql  # Analytical query examples
|       |-- topic_trends.sql
|-- notebooks/
|   |-- 01_eda_conversation_logs.ipynb  # Exploratory data analysis
|   |-- 02_embedding_quality_analysis.ipynb
|-- docker/
|   |-- postgres/
|       |-- init.sql               # DB initialization script
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- .env.example
|-- README.md
```

---

## Setup and Installation

### Prerequisites

- Docker and Docker Compose (v2.x)
- Python 3.11+
- An OpenAI API key (or a local HuggingFace model for offline use)

### Environment Configuration

```bash
cp .env.example .env
# Edit .env to set OPENAI_API_KEY, POSTGRES_* credentials, and VECTOR_DB_PATH
```

### Option A: Docker Compose (Recommended)

This starts PostgreSQL, initializes the schema, and launches the application container.

```bash
docker-compose up --build
```

### Option B: Local Development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the database schema
psql -h localhost -U sentinel_user -d sentinel_db -f sql/schema.sql

# Run the ingestion pipeline against sample data
python src/data_ingestion.py --source data/sample_logs/ --batch-size 512

# Start the Q&A interface
python src/rag_chain.py --interactive
```

### Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## ETL Pipeline Design

Sentinel follows an **ELT** (Extract, Load, Transform) pattern for the relational layer and an **ETL** pattern for the vector layer. This distinction is intentional:

- **ELT for PostgreSQL**: Raw structured fields (IDs, timestamps, status codes) are loaded directly into the database without transformation, allowing SQL-based transformations to be applied later as requirements evolve. This avoids baking business logic into the ingestion layer.

- **ETL for Vector Storage**: Text content must be cleaned and transformed *before* embedding, because the embedding operation itself is the final representation. Post-hoc correction of noisy embeddings requires full re-indexing, making pre-embedding transformation the only cost-effective approach.

This hybrid ETL/ELT design separates concerns cleanly and aligns with the different mutability characteristics of each storage layer.

---

## Future Development

- **Feedback Loop**: Capture user up/down votes on RAG responses and use them to fine-tune the re-ranker via preference learning.
- **Streaming Ingestion**: Replace batch file ingestion with a Kafka consumer for real-time conversation processing.
- **Multi-tenant Isolation**: Introduce row-level security in PostgreSQL and namespace-level isolation in ChromaDB to support multiple business units.
- **Evaluation Framework**: Integrate RAGAS (RAG Assessment) for automated retrieval precision and answer faithfulness scoring on a held-out evaluation set.

---

## License

MIT License. See `LICENSE` for details.
