# Academic Literature RAG

**Academic Literature RAG** is a research-oriented Retrieval-Augmented Generation system for academic papers.

It searches academic sources, persists paper metadata, downloads open-access PDFs, extracts and chunks text, generates embeddings, performs semantic search, and produces grounded answers with citations.

The project is designed as both:

1. A practical engineering MVP for literature-based RAG.
2. A foundation for research experiments on retrieval quality, grounding, citation faithfulness, and scientific question answering.

---

## Current Status

The project currently supports a working end-to-end RAG flow:

```text
paper search
→ metadata persistence
→ PDF registration
→ PDF download
→ PDF text extraction
→ text cleaning
→ chunking
→ embedding generation
→ semantic search
→ evidence prompt building
→ grounded answer generation
→ citations
```

Implemented:

- arXiv retrieval
- Semantic Scholar retrieval
- fallback paper search
- raw API response persistence
- search run persistence
- source paper persistence
- canonical paper linking
- PDF asset tracking
- pending PDF download pipeline
- PDF validation
- page-level PDF text extraction
- text cleaning
- chunking
- embedding persistence
- semantic search over embedded chunks
- grounded answer generation
- citation objects linked to retrieved chunks
- pipeline orchestration service
- deterministic local smoke scripts
- OpenAI generation client
- OpenAI embedding client
- OpenAI-backed smoke scripts
- one end-to-end OpenAI RAG demo script

---

## Why This Project Exists

Academic literature workflows often require more than simple keyword search.

Researchers need to:

- find relevant papers,
- inspect evidence inside PDFs,
- ask questions across papers,
- trace answers back to source passages,
- compare retrieval strategies,
- evaluate grounding quality,
- and build reproducible research pipelines.

This project focuses on building a transparent and testable RAG pipeline where every stage is inspectable:

```text
source response
→ persisted record
→ downloaded PDF
→ extracted pages
→ chunks
→ embeddings
→ retrieved evidence
→ generated answer
→ citations
```

---

## Core Capabilities

### Paper Retrieval

The project retrieves paper candidates from academic sources.

Current sources:

- arXiv
- Semantic Scholar

It also includes a fallback paper search service that can try a primary source first and fall back to another source when the primary source fails or returns no usable results.

---

### Persistence Layer

The project stores intermediate artifacts instead of treating retrieval as a temporary API call.

Persisted entities include:

- search runs
- source papers
- canonical papers
- PDF assets
- page-level extracted text
- text chunks
- chunk embeddings

This makes the pipeline reproducible, debuggable, and easier to evaluate.

---

### PDF Processing

The PDF pipeline supports:

- registering open-access PDF assets,
- downloading pending PDF assets,
- validating downloaded files,
- storing local PDF paths,
- extracting page-level text,
- normalizing extracted text,
- and persisting page text for later chunking.

---

### Chunking

Extracted PDF text is cleaned and converted into retrieval-ready chunks.

The chunking stage keeps page number information, which allows later answers to cite the retrieved evidence with page ranges.

---

### Embeddings

The project supports two embedding modes:

1. **Deterministic smoke embeddings**  
   Used for local development and tests without external API calls.

2. **OpenAI embeddings**  
   Used for real semantic retrieval.

Default OpenAI embedding model:

```text
text-embedding-3-small
```

This can be changed with:

```bash
OPENAI_EMBEDDING_MODEL
```

---

### Semantic Search

The semantic search service:

1. embeds the user query,
2. loads stored chunk embeddings for the same model,
3. computes cosine similarity,
4. ranks chunks,
5. returns the top retrieved evidence chunks.

---

### Grounded Answer Generation

The RAG answer service:

1. retrieves semantically relevant chunks,
2. builds an evidence-only prompt,
3. sends the prompt to a generation client,
4. returns a grounded answer,
5. attaches citations for the retrieved chunks.

Default OpenAI generation model:

```text
gpt-4o-mini
```

This can be changed with:

```bash
OPENAI_GENERATION_MODEL
```

---

## Architecture

High-level architecture:

```text
connectors
  ↓
retrieval services
  ↓
repositories
  ↓
PDF services
  ↓
text services
  ↓
embedding services
  ↓
semantic search
  ↓
prompt builder
  ↓
generation client
  ↓
grounded answer
```

Main project structure:

```text
src/academic_literature_rag/

  connectors/
    arxiv.py
    semantic_scholar.py
    protocols.py

  database/
    models.py
    session.py

  identity/
    matching.py
    normalizers.py

  models/
    canonical_paper.py
    chunk_embedding.py
    paper_candidate.py
    pdf_asset.py
    pdf_page_text.py
    rag_answer.py
    retrieval_result.py
    search_run.py
    semantic_search_result.py
    text_chunk.py

  repositories/
    canonical_paper_repository.py
    chunk_embedding_repository.py
    pdf_asset_repository.py
    pdf_page_text_repository.py
    search_run_repository.py
    source_paper_repository.py
    text_chunk_repository.py

  retrieval/
    vector_math.py

  services/
    chunk_embedding_service.py
    embedding_client.py
    fallback_paper_search_service.py
    generation_client.py
    openai_embedding_client.py
    openai_generation_client.py
    pdf_download_service.py
    pdf_text_extraction_service.py
    pending_pdf_download_service.py
    persisted_retrieval_service.py
    rag_answer_service.py
    rag_pipeline_service.py
    rag_prompt_builder.py
    semantic_search_service.py
    text_chunking_service.py
    text_cleaning_service.py

  storage/
    raw_response_store.py
```

---

## Requirements

- Python 3.12+
- uv
- OpenAI API key for OpenAI-backed scripts
- Optional Semantic Scholar API key

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd academic-literature-rag
```

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run ruff check .
uv run pytest
```

---

## Environment Setup

Create a local environment file:

```bash
cp .env.example .env
```

Then fill in:

```bash
OPENAI_API_KEY=your-key-here
```

Main environment variables:

```bash
OPENAI_API_KEY=

OPENAI_GENERATION_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

SEMANTIC_SCHOLAR_API_KEY=

RAG_DEMO_SEARCH_QUERY=transformer attention
RAG_DEMO_QUESTION=What is attention in transformers?

RAG_DEMO_RETRIEVAL_LIMIT=2
RAG_DEMO_DOWNLOAD_LIMIT=1
RAG_DEMO_EMBEDDING_LIMIT=8
RAG_DEMO_TOP_K=3

RAG_DEMO_DATABASE_PATH=data/db/dev.db
RAG_DEMO_RAW_RESPONSE_DIR=data/raw_responses
RAG_DEMO_PDF_DIR=data/pdfs
```

Never commit real API keys.

---

## Quick Start

### Run the full OpenAI-backed end-to-end demo

```bash
export OPENAI_API_KEY='your-key-here'
uv run python scripts/demo_rag_end_to_end.py
```

Expected output starts with:

```text
Academic Literature RAG end-to-end demo completed.
```

The demo runs:

```text
arXiv retrieval
→ PDF processing
→ OpenAI embedding generation
→ semantic search
→ OpenAI answer generation
→ citation printing
```

---

## Demo Configuration

Change the paper search query:

```bash
export RAG_DEMO_SEARCH_QUERY="medical image retrieval"
```

Change the question:

```bash
export RAG_DEMO_QUESTION="How is retrieval used in medical image analysis?"
```

Change limits:

```bash
export RAG_DEMO_RETRIEVAL_LIMIT=3
export RAG_DEMO_DOWNLOAD_LIMIT=1
export RAG_DEMO_EMBEDDING_LIMIT=10
export RAG_DEMO_TOP_K=5
```

Run the demo:

```bash
uv run python scripts/demo_rag_end_to_end.py
```

---

## Development Smoke Scripts

The repository includes smaller smoke scripts for debugging individual stages.

### Local Ingestion Smoke

Uses arXiv retrieval and deterministic local embeddings.

```bash
uv run python scripts/smoke_rag_pipeline.py
```

Validates:

```text
retrieval
→ PDF registration
→ PDF download
→ text extraction
→ chunking
→ deterministic embedding persistence
```

---

### Local Answer Smoke

Uses deterministic local embeddings and deterministic fake generation.

```bash
uv run python scripts/smoke_rag_answer.py
```

Validates:

```text
semantic search
→ prompt building
→ generation interface
→ grounded answer with citations
```

---

### OpenAI Answer Smoke

Uses OpenAI generation for grounded answer generation.

```bash
export OPENAI_API_KEY='your-key-here'
uv run python scripts/smoke_rag_answer_openai.py
```

Validates:

```text
semantic search
→ prompt building
→ OpenAI generation
→ grounded answer with citations
```

---

### OpenAI Embedding Pipeline Smoke

Uses OpenAI embeddings during ingestion.

```bash
export OPENAI_API_KEY='your-key-here'
uv run python scripts/smoke_rag_pipeline_openai_embeddings.py
```

Validates:

```text
retrieval
→ PDF processing
→ chunking
→ OpenAI embeddings
→ embedding persistence
```

---

## Testing

Run all checks:

```bash
uv run ruff check .
uv run pytest
```

Run a single test file:

```bash
uv run pytest tests/unit/services/test_rag_answer_service.py
```

Run OpenAI client tests:

```bash
uv run pytest tests/unit/services/test_openai_generation_client.py
uv run pytest tests/unit/services/test_openai_embedding_client.py
```

OpenAI client unit tests use fake SDK clients. They do not call the real OpenAI API.

---

## Generated Local Artifacts

The project writes local development artifacts under:

```text
data/db/dev.db
data/raw_responses/
data/pdfs/
```

Temporary command outputs can be stored under:

```text
_tmp/
```

These should not be committed.

Ignored local artifacts should include:

```text
.env
data/
_tmp/
.venv/
__pycache__/
```

---

## Safety and Secrets

Do not commit:

```text
.env
OPENAI_API_KEY
local SQLite databases
downloaded PDFs
raw response dumps
temporary outputs
```

Use `.env.example` for safe configuration documentation.

---

## Current Limitations

The project is a working engineering MVP, but it is not yet a full research evaluation framework.

Current limitations:

- No retrieval benchmark yet
- No citation faithfulness evaluation yet
- No answer faithfulness evaluation yet
- No reranking layer yet
- No hybrid retrieval yet
- No CLI package entry point yet
- No web interface or API server yet
- No experiment tracking yet
- No paper-level formatted bibliography output yet
- No systematic comparison against baseline retrieval methods yet

---

## Roadmap

### Near-term Engineering Milestones

- Add CLI entry point
- Add typed configuration object
- Improve demo output formatting
- Add JSON output option
- Add Markdown report output option
- Add better error messages for missing local artifacts
- Add reusable pipeline factory to reduce script duplication

### Research Milestones

- Build a small evaluation dataset
- Add retrieval precision and recall metrics
- Add citation grounding checks
- Add answer faithfulness evaluation
- Add benchmark questions
- Compare semantic retrieval against keyword retrieval
- Add hybrid retrieval
- Add reranking
- Evaluate chunk size and overlap sensitivity
- Evaluate different embedding models
- Evaluate different generation models

---

## Intended Research Direction

This repository can support experiments around:

- scientific literature RAG,
- grounded academic question answering,
- citation-aware generation,
- retrieval quality evaluation,
- PDF-based evidence extraction,
- chunking strategies,
- embedding model comparison,
- and answer faithfulness analysis.

The long-term goal is to move from a working RAG pipeline to a reproducible research framework for academic literature retrieval and grounded generation.

---

## License

Add a license before public release.

Recommended options:

- MIT for permissive open-source release
- Apache-2.0 for permissive release with explicit patent terms
- Private repository if the project is still under active research development