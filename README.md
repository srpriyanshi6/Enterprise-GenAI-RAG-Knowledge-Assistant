# Enterprise-GenAI-Knowledge-Assistant

## LIVE LINK : https://enterprise-genai-rag-knowledge-assistant.streamlit.app/

A production-grade **Retrieval-Augmented Generation (RAG)** system that enables semantic document search and citation-grounded Q&A over private PDF knowledge bases, powered by Groq, LangChain, FAISS, and HuggingFace Transformers.

<img width="1436" height="780" alt="Screenshot 2026-05-22 at 7 41 48 AM" src="https://github.com/user-attachments/assets/24a93862-5ae4-4a17-9a5d-dc1a3f7977ff" />

---

## What It Does

Upload any set of PDF documents and ask questions in natural language. The assistant retrieves the most semantically relevant passages from your documents and generates precise, grounded answers — always citing the source. It never fabricates information outside the provided context.


<img width="1429" height="746" alt="Screenshot 2026-05-22 at 7 45 07 AM" src="https://github.com/user-attachments/assets/d9891af7-c4be-410a-9d48-1f687ad2c9a4" />

Answers question based on document (pdf) provided.

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
- **Strict source grounding** : system prompt enforces answers only from retrieved context, eliminating hallucination

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit (responsive, wide layout) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Orchestration | LangChain 0.3 |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (CPU) |
| PDF Parsing | pypdf (with `strict=False` fallback for malformed PDFs) |
| Memory | LangChain `MessagesPlaceholder` (session-scoped) |
| Deployment | Streamlit Community Cloud |
| Secrets Management | Environment variables via `.env` / Streamlit Secrets |

---

## Frontend & UI Design

The interface is built with **Streamlit's wide layout** for a clean, full-screen experience that works across desktops, laptops, and tablets.

<img width="1440" height="799" alt="Screenshot 2026-05-22 at 7 56 05 AM" src="https://github.com/user-attachments/assets/19262331-7d14-472a-8dfb-ccfcc74e270e" />

<img width="1440" height="810" alt="Screenshot 2026-05-22 at 7 56 34 AM" src="https://github.com/user-attachments/assets/2f148173-349e-4536-a5bd-6dc41c4137c8" />



**UI highlights:**
- **Two-panel layout** — sidebar for document management, main panel for the chat interface
- **Real-time feedback** — spinner animations during document processing and LLM generation
- **Chat-style message rendering** — distinct user and assistant message bubbles with markdown support
- **Collapsible source expanders** — cited source documents are accessible but non-intrusive
- **Session state management** — chat history and vector store persist across interactions without re-processing
- **Responsive design** — adapts fluidly to different screen widths
- **Error and warning surfaces** — clear user-facing messages for edge cases (empty PDFs, corrupt files)
- **Zero-friction file upload** — native drag-and-drop multi-file PDF uploader

---

## Features

- **Semantic search** — finds conceptually relevant content, not just keyword matches
- **Multi-document support** — index and query across multiple PDFs simultaneously
- **Source citations** — every answer is traced back to the originating document
- **Conversational memory** — follow-up questions are understood in context
- **Hallucination prevention** — LLM is constrained to only the retrieved context
- **Resilient PDF parsing** — graceful fallback for corrupted or non-standard PDFs
- **Zero external vector DB** — fully self-contained, no Pinecone/Weaviate setup needed
- **Cloud deployed** — live on Streamlit Community Cloud, accessible from any browser

<img width="307" height="389" alt="Screenshot 2026-05-22 at 7 46 16 AM" src="https://github.com/user-attachments/assets/3a36f897-0347-4374-a19e-71dc2c2ea409" />

Multi-document support

---

## Getting Started

### Prerequisites

- Python 3.11
- A [Groq API key]

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

### Generative AI & LLM Engineering
- **RAG (Retrieval-Augmented Generation)** — end-to-end pipeline design and implementation
- **LLM integration** — Groq API with structured prompt engineering and system-level instructions
- **Prompt engineering** — context injection, role assignment, hallucination mitigation
- **Conversational AI** — multi-turn dialogue with persistent chat history
- **LangChain** — chains, retrievers, memory, prompt templates, document loaders

### Machine Learning & NLP
- **Vector embeddings** — dense semantic representations using transformer-based models
- **Semantic similarity search** — cosine/L2 similarity over high-dimensional embedding space
- **HuggingFace Transformers** — local inference with `sentence-transformers`
- **NLP pipelines** — tokenization, chunking, text splitting strategies
- **Information retrieval** — top-k retrieval, context window management

### Backend & Software Engineering
- **Python** — clean, modular, production-aware code
- **FAISS** — vector indexing and nearest-neighbour search at scale
- **API integration** — RESTful LLM API consumption with environment-based secrets management
- **Error handling & resilience** — multi-level fallbacks, graceful degradation, user-facing error surfaces
- **Session state management** — stateful application design without a traditional backend

### Frontend & Full Stack
- **Streamlit** — interactive, reactive web UI built entirely in Python
- **Responsive UI design** — two-panel layout, wide-screen optimisation, mobile-aware components
- **Real-time UX** — live spinners, streaming-style feedback, progressive rendering
- **Component architecture** — sidebar controls, chat message components, collapsible expanders
- **State-driven UI** — UI behaviour driven by session state, not page reloads

### DevOps & Deployment
- **Cloud deployment** — live production app on Streamlit Community Cloud
- **Dependency management** — pinned `requirements.txt` for reproducible builds
- **Environment secrets** — `.env` locally, Streamlit Secrets in production
- **System dependencies** — `packages.txt` for native library provisioning

---

## Keywords

`Generative AI` · `RAG` · `Retrieval-Augmented Generation` · `LLM` · `Large Language Models` · `LangChain` · `Groq` · `LLaMA` · `HuggingFace` · `Transformers` · `Sentence Transformers` · `FAISS` · `Vector Database` · `Vector Search` · `Semantic Search` · `Embeddings` · `NLP` · `Natural Language Processing` · `Information Retrieval` · `Prompt Engineering` · `Conversational AI` · `Chatbot` · `Document Q&A` · `Knowledge Base` · `Python` · `Streamlit` · `Full Stack` · `Machine Learning` · `Deep Learning` · `AI Engineering` · `MLOps` · `Cloud Deployment` · `REST API` · `Session Management` · `PDF Processing` · `Text Chunking` · `Context Window` · `Hallucination Mitigation` · `Citation` · `Source Grounding`

---
