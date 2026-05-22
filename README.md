# Enterprise-GenAI-Knowledge-Assistant

# LIVE LINK : https://enterprise-genai-rag-knowledge-assistant.streamlit.app/

A production-grade **Retrieval-Augmented Generation (RAG)** system that enables semantic document search and citation-grounded Q&A over private PDF knowledge bases — powered by Groq, LangChain, FAISS, and HuggingFace Transformers.

---

## What It Does

Upload any set of PDF documents and ask questions in natural language. The assistant retrieves the most semantically relevant passages from your documents and generates precise, grounded answers — always citing the source. It never fabricates information outside the provided context.

---

## Architecture

```
User Query
    │
    ▼
HuggingFace Embeddings          ← sentence-transformers/all-MiniLM-L6-v2
(query → dense vector)
    │
    ▼
FAISS Vector Store              ← top-k similarity search (k=4)
(retrieve relevant chunks)
    │
    ▼
LangChain Retrieval Chain       ← context assembly + chat history injection
    │
    ▼
Groq LLM (llama-3.3-70b)       ← ultra-fast inference via Groq Cloud
    │
    ▼
Grounded Answer + Source Citation
```

**Key design decisions:**
- **FAISS** for local, zero-latency vector search — no external vector DB dependency
- **Groq** for near-instant LLM inference (vs. OpenAI latency)
- **Conversational memory** via LangChain `MessagesPlaceholder` — multi-turn context preserved across the session
- **Chunk overlap** of 200 tokens prevents context fragmentation at boundaries
- **Strict source grounding** — system prompt enforces answers only from retrieved context, eliminating hallucination

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Orchestration | LangChain 0.3 |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (CPU) |
| PDF Parsing | pypdf (with strict=False fallback for malformed PDFs) |
| Memory | LangChain `MessagesPlaceholder` (session-scoped) |

---

## Features

- **Semantic search** — finds conceptually relevant content, not just keyword matches
- **Multi-document support** — index and query across multiple PDFs simultaneously
- **Source citations** — every answer is traced back to the originating document
- **Conversational memory** — follow-up questions are understood in context
- **Hallucination prevention** — LLM is constrained to only the retrieved context
- **Resilient PDF parsing** — graceful fallback for corrupted or non-standard PDFs
- **Zero external vector DB** — fully self-contained, no Pinecone/Weaviate setup needed

---

## Getting Started

### Prerequisites

- Python 3.12+
- A [Groq API key](https://console.groq.com) (free tier available)

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
streamlit run app.py
```

---

## RAG Pipeline — Deep Dive

### 1. Document Ingestion
PDFs are loaded via `PyPDFLoader` with a resilient fallback to direct `pypdf.PdfReader(strict=False)` for malformed files. Each page is extracted as a LangChain `Document` with source metadata preserved.

### 2. Chunking
`RecursiveCharacterTextSplitter` splits documents into 1000-token chunks with 200-token overlap. The recursive strategy respects natural text boundaries (paragraphs → sentences → words) before hard-splitting.

### 3. Embedding
Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` — a 384-dimensional model optimised for semantic similarity. Runs locally with no API cost.

### 4. Vector Storage
Embeddings are stored in an in-memory FAISS index using L2 distance for similarity search. On query, the top-4 most relevant chunks are retrieved.

### 5. Generation
Retrieved chunks are injected into a `ChatPromptTemplate` alongside the full conversation history. The Groq-hosted `llama-3.3-70b-versatile` model generates a grounded response constrained to the provided context.

---

## Project Structure

```
├── app.py               # Streamlit UI + full RAG pipeline
├── requirements.txt     # Python dependencies (pinned)
├── packages.txt         # System dependencies
├── .env                 # API keys (not committed)
└── README.md
```

---

## Skills Demonstrated

- **RAG system design** — end-to-end pipeline from ingestion to grounded generation
- **LLM integration** — Groq API with structured prompt engineering
- **Vector search** — FAISS indexing, dense retrieval, similarity search
- **LangChain orchestration** — chains, retrievers, memory, prompt templates
- **HuggingFace Transformers** — local embedding model inference
- **Production resilience** — error handling, fallbacks, empty-state guards
- **Streamlit** — interactive multi-page UI with session state management
