# ADR-0001: Use a Modular Monolith with Layered Responsibilities

**Status:** Accepted  
**Date:** 2026-07-02

## Context

Academic Literature RAG will retrieve academic papers from external sources,
preserve raw metadata, process open-access PDFs, build a searchable knowledge
base, and generate citation-grounded answers.

The initial system is being developed by one person and does not currently need
separate deployments, independent scaling, or distributed infrastructure.

## Decision

Use a modular monolith with layered responsibilities.

```text
CLI / Interface
        ↓
Application Services
        ↓
Connectors / Repositories / Storage
        ↓
External APIs / Database / Local Files