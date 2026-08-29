# Release Notes

## Current engineering MVP

Academic Literature RAG is currently a working end-to-end engineering MVP for literature-based retrieval-augmented generation.

### Implemented

- Academic paper retrieval from arXiv and Semantic Scholar
- Fallback retrieval strategy
- Metadata persistence
- PDF asset tracking
- Open-access PDF download and validation
- PDF text extraction
- Text cleaning and chunking
- Embedding persistence
- OpenAI embedding client
- Semantic search over embedded chunks
- Prompt construction
- OpenAI grounded answer generation
- Citation objects linked to retrieved evidence
- Typed environment configuration
- Service factory / dependency composition layer
- CLI demo command
- Text, JSON, and Markdown output formats
- Output file support
- Text and JSON logging
- Unit tests
- CLI integration tests
- GitHub Actions CI

### Notes

This repository is intended as a clean AI engineering portfolio project and a reusable foundation for academic literature RAG experimentation.
