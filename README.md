# Academic Literature RAG

[![CI](https://github.com/el-gabry/academic-literature-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/el-gabry/academic-literature-rag/actions/workflows/ci.yml)

**Academic Literature RAG** is a production-style Retrieval-Augmented Generation pipeline for academic papers.

It retrieves academic papers, persists metadata, downloads open-access PDFs, extracts and chunks text, generates embeddings, performs semantic search, and produces grounded answers with citations.

The project demonstrates an end-to-end AI engineering workflow for literature-based question answering.

---

## Highlights

- Academic paper retrieval from arXiv and Semantic Scholar
- Fallback retrieval strategy
- Persistent storage for search runs, source papers, canonical papers, PDF assets, extracted text, chunks, and embeddings
- Open-access PDF download and validation
- Page-level PDF text extraction
- Text cleaning and chunking with page range tracking
- OpenAI embedding integration
- Semantic search over embedded chunks
- OpenAI grounded answer generation
- Citation objects linked to retrieved evidence
- Typed environment configuration
- Service factory / dependency composition layer
- Command-line interface
- Text, JSON, and Markdown output formats
- Output file support
- Text and JSON logging
- Unit tests and GitHub Actions CI

---

## System Overview

High-level flow:

```text
paper query
→ paper retrieval
→ metadata persistence
→ PDF registration
→ PDF download
→ PDF text extraction
→ text cleaning
→ chunking
→ embedding generation
→ semantic search
→ prompt construction
→ answer generation
→ grounded citations
```

The pipeline keeps intermediate artifacts so each stage can be inspected, tested, and reused.

---

## Architecture

The project is organized into clear layers:

```text
connectors
  ↓
retrieval services
  ↓
repositories
  ↓
PDF services
  ↓
text processing
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

Main package structure:

```text
src/academic_literature_rag/

  app/
    demo_output.py
    factory.py
    logging_config.py

  connectors/
    arxiv.py
    semantic_scholar.py
    protocols.py

  database/
    models.py
    session.py

  models/
    paper_candidate.py
    pdf_asset.py
    pdf_page_text.py
    text_chunk.py
    chunk_embedding.py
    retrieval_result.py
    rag_answer.py
    semantic_search_result.py

  repositories/
    search_run_repository.py
    source_paper_repository.py
    canonical_paper_repository.py
    pdf_asset_repository.py
    pdf_page_text_repository.py
    text_chunk_repository.py
    chunk_embedding_repository.py

  services/
    persisted_retrieval_service.py
    fallback_paper_search_service.py
    pdf_download_service.py
    pending_pdf_download_service.py
    pdf_text_extraction_service.py
    text_cleaning_service.py
    text_chunking_service.py
    chunk_embedding_service.py
    semantic_search_service.py
    rag_prompt_builder.py
    rag_answer_service.py
    rag_pipeline_service.py
    openai_embedding_client.py
    openai_generation_client.py
```

---

## Engineering Design

### Typed configuration

Environment-driven settings are centralized in:

```text
src/academic_literature_rag/config.py
```

This keeps API keys, model names, storage paths, and demo limits out of individual scripts.

### Service factory

Application wiring is centralized in:

```text
src/academic_literature_rag/app/factory.py
```

The factory builds repositories, clients, pipeline services, and answer services from a single configuration object.

### CLI interface

The project exposes a command-line interface:

```bash
uv run academic-literature-rag --help
```

### Structured output

The demo supports multiple output formats:

```bash
uv run academic-literature-rag demo --format text
uv run academic-literature-rag demo --format json
uv run academic-literature-rag demo --format markdown
```

### Output files

Rendered demo output can be written directly to a file:

```bash
uv run academic-literature-rag demo \
  --format markdown \
  --output-file _tmp/demo_output.md
```

### Structured logging

Logs can be written as text or JSON:

```bash
uv run academic-literature-rag demo \
  --log-level INFO \
  --log-format json
```

Logs are written to `stderr`, while rendered demo output is written to `stdout` or to the selected `--output-file`.

This keeps JSON and Markdown outputs clean for automation and reporting.

---

## Requirements

- Python 3.12+
- uv
- OpenAI API key for OpenAI-backed demos
- Optional Semantic Scholar API key

---

## Installation

Clone the repository:

```bash
git clone https://github.com/el-gabry/academic-literature-rag.git
cd academic-literature-rag
```

Install dependencies:

```bash
uv sync
```

Run checks:

```bash
uv run ruff check .
uv run pytest
```

---

## Environment Setup

Copy the example environment file:

```bash
cp .env.example .env
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY='your-key-here'
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

Run the full OpenAI-backed demo:

```bash
export OPENAI_API_KEY='your-key-here'
uv run academic-literature-rag demo
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

## CLI Usage

Show root help:

```bash
uv run academic-literature-rag --help
```

Show demo help:

```bash
uv run academic-literature-rag demo --help
```

Run the default demo:

```bash
uv run academic-literature-rag demo
```

Run with text output:

```bash
uv run academic-literature-rag demo --format text
```

Run with JSON output:

```bash
uv run academic-literature-rag demo --format json
```

Run with Markdown output:

```bash
uv run academic-literature-rag demo --format markdown
```

Write JSON output directly to a file:

```bash
uv run academic-literature-rag demo \
  --format json \
  --output-file _tmp/demo_output.json
```

Write Markdown output directly to a file:

```bash
uv run academic-literature-rag demo \
  --format markdown \
  --output-file _tmp/demo_output.md
```

Use structured JSON logs while saving Markdown output:

```bash
uv run academic-literature-rag demo \
  --format markdown \
  --output-file _tmp/demo_output.md \
  --log-level INFO \
  --log-format json
```

Logs are written to `stderr`, while rendered demo output is written to `stdout` or to the selected `--output-file`.

---

## Customizing the Demo

Change the paper search query:

```bash
export RAG_DEMO_SEARCH_QUERY="medical image retrieval"
```

Change the question:

```bash
export RAG_DEMO_QUESTION="How is retrieval used in medical image analysis?"
```

Change retrieval limits:

```bash
export RAG_DEMO_RETRIEVAL_LIMIT=3
export RAG_DEMO_DOWNLOAD_LIMIT=1
export RAG_DEMO_EMBEDDING_LIMIT=10
export RAG_DEMO_TOP_K=5
```

Run:

```bash
uv run academic-literature-rag demo --format markdown
```

---

## Development Scripts

The CLI is the preferred interface. Smaller smoke scripts are also available for debugging individual stages.

### Local ingestion smoke

```bash
uv run python scripts/smoke_rag_pipeline.py
```

### Local answer smoke

```bash
uv run python scripts/smoke_rag_answer.py
```

### OpenAI answer smoke

```bash
export OPENAI_API_KEY='your-key-here'
uv run python scripts/smoke_rag_answer_openai.py
```

### OpenAI embedding pipeline smoke

```bash
export OPENAI_API_KEY='your-key-here'
uv run python scripts/smoke_rag_pipeline_openai_embeddings.py
```

### End-to-end demo script

```bash
uv run python scripts/demo_rag_end_to_end.py
```

---

## Testing

Run all checks:

```bash
uv run ruff check .
uv run pytest
```

Run selected tests:

```bash
uv run pytest tests/unit/services/test_rag_answer_service.py
uv run pytest tests/unit/services/test_openai_generation_client.py
uv run pytest tests/unit/services/test_openai_embedding_client.py
uv run pytest tests/unit/app/test_factory.py
uv run pytest tests/unit/app/test_demo_output.py
uv run pytest tests/unit/app/test_logging_config.py
uv run pytest tests/unit/test_cli_output_file.py
uv run pytest tests/integration/test_cli.py
```

OpenAI client tests use fake SDK clients and do not call the real OpenAI API.

---

## CI

GitHub Actions runs automated checks on pushes and pull requests:

```text
uv sync
ruff check .
pytest
```

---

## Generated Local Artifacts

Local development artifacts are written under:

```text
data/db/dev.db
data/raw_responses/
data/pdfs/
```

Temporary outputs can be stored under:

```text
_tmp/
```

These files are ignored by Git and should not be committed.

---

## Safety and Secrets

Do not commit:

```text
.env
OPENAI_API_KEY
local SQLite databases
downloaded PDFs
raw API responses
temporary outputs
```

Use `.env.example` for safe configuration documentation.

---

## Current Status

Implemented:

```text
Retrieval connectors
Fallback retrieval
Persistence layer
PDF asset tracking
PDF download and validation
PDF text extraction
Text cleaning and chunking
Embedding persistence
OpenAI embedding client
Semantic search
Prompt builder
OpenAI generation client
Grounded answer service
Citation objects
Pipeline orchestration
Typed configuration
Service factory
CLI demo command
Text / JSON / Markdown outputs
Output file support
Text / JSON logging
GitHub Actions CI
Unit and integration tests
```

The project is currently a working engineering MVP with a production-style structure.

---

## Skills Demonstrated

This project demonstrates:

```text
Retrieval-Augmented Generation
LLM application engineering
OpenAI API integration
PDF processing
Semantic search
Embedding pipelines
SQL persistence
Repository pattern
Service-layer architecture
Factory pattern
Dependency injection
Typed configuration
CLI development
Structured output design
Structured logging
Unit testing
Integration testing
CI workflow setup
Safe configuration management
```

---

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.
