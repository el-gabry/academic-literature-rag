"""Compare retrieval quality across dense, hybrid, and hybrid+rerank modes.

Usage:
    uv run python scripts/compare_retrieval_modes.py

Reads labeled queries from tests/fixtures/labeled_queries.json, builds all
three retrievers directly (bypassing RagServiceFactory.create_core_services,
since that method only ever builds whichever single mode
AppConfig.retrieval_mode names -- there is no shared helper we can call
three times with different modes), runs each against every query, and
reports Precision@k / Recall@k / MRR per mode. Writes a per-query breakdown
to retrieval_comparison.csv.

Retrieval modes compared:
    dense           -- SemanticSearchService only
    hybrid          -- HybridSearchService (dense + BM25 via RRF)
    hybrid+rerank   -- HybridSearchService candidates rescored by
                       RerankingSearchService (cross-encoder)

Reranking cannot improve recall over its base retriever -- it only reorders
the candidate_pool_size=20 candidates hybrid already found, then returns the
top_k best of those. A hybrid+rerank run with identical Recall@k but higher
Precision@k/MRR than plain hybrid is the expected "good" outcome, not a bug.

NOTE: The application supports "dense", "hybrid", and "hybrid_rerank"
through AppConfig and RagServiceFactory. This script constructs all three
retrievers directly against the same stored corpus so that their retrieval
quality can be compared in one run.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from academic_literature_rag.config import AppConfig
from academic_literature_rag.database.session import (
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.services.hybrid_search_service import (
    HybridSearchService,
)
from academic_literature_rag.services.openai_embedding_client import (
    OpenAIEmbeddingClient,
)
from academic_literature_rag.services.reranking_search_service import (
    RerankingSearchService,
)
from academic_literature_rag.services.semantic_search_service import (
    SemanticSearchService,
)

FIXTURES_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "labeled_queries.json"
)
OUTPUT_CSV_PATH = Path(__file__).resolve().parent.parent / "retrieval_comparison.csv"
TOP_K = 5


@dataclass(frozen=True)
class LabeledQuery:
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class QueryModeResult:
    mode: str
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    latency_ms: float


def load_labeled_queries(path: Path) -> list[LabeledQuery]:
    raw = json.loads(path.read_text())
    return [
        LabeledQuery(
            query=item["query"],
            relevant_chunk_ids=frozenset(item["relevant_chunk_ids"]),
        )
        for item in raw
    ]


def build_retrievers(config: AppConfig) -> dict[str, object]:
    """Construct dense, hybrid, and hybrid+rerank retrievers directly.

    Mirrors the wiring in RagServiceFactory.create_core_services(), which
    only builds a single configured mode at a time.
    """
    engine = create_sqlite_engine(config.storage.database_path)
    session_factory = create_session_factory(engine)

    chunk_embedding_repository = ChunkEmbeddingRepository(session_factory)
    text_chunk_repository = TextChunkRepository(session_factory)
    embedding_client = OpenAIEmbeddingClient(model_name=config.openai.embedding_model)

    dense_service = SemanticSearchService(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=embedding_client,
    )

    hybrid_service = HybridSearchService(
        semantic_search_service=dense_service,
        text_chunk_repository=text_chunk_repository,
    )
    hybrid_service.refresh_index()

    reranked_service = RerankingSearchService(base_search_service=hybrid_service)

    return {
        "dense": dense_service,
        "hybrid": hybrid_service,
        "hybrid_rerank": reranked_service,
    }


def score_result(
    mode: str,
    query: str,
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: frozenset[str],
    latency_ms: float,
    top_k: int,
) -> QueryModeResult:
    top_k_ids = retrieved_chunk_ids[:top_k]
    hits = [chunk_id for chunk_id in top_k_ids if chunk_id in relevant_chunk_ids]

    precision_at_k = len(hits) / top_k if top_k else 0.0
    recall_at_k = len(hits) / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0

    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(top_k_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            reciprocal_rank = 1.0 / rank
            break

    return QueryModeResult(
        mode=mode,
        query=query,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        latency_ms=latency_ms,
    )


def run_mode(
    mode: str,
    retriever,
    labeled_queries: list[LabeledQuery],
    top_k: int,
) -> list[QueryModeResult]:
    results = []
    for labeled_query in labeled_queries:
        start = time.perf_counter()
        search_results = retriever.search(labeled_query.query, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_chunk_ids = [str(r.text_chunk_id) for r in search_results]
        results.append(
            score_result(
                mode=mode,
                query=labeled_query.query,
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=labeled_query.relevant_chunk_ids,
                latency_ms=latency_ms,
                top_k=top_k,
            )
        )
    return results


def summarize(results: list[QueryModeResult]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "avg_latency_ms": 0.0}
    return {
        "precision_at_k": sum(r.precision_at_k for r in results) / n,
        "recall_at_k": sum(r.recall_at_k for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "avg_latency_ms": sum(r.latency_ms for r in results) / n,
    }


def write_csv(all_results: list[QueryModeResult], path: Path) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["mode", "query", "precision_at_k", "recall_at_k", "reciprocal_rank", "latency_ms"]
        )
        for result in all_results:
            writer.writerow(
                [
                    result.mode,
                    result.query,
                    f"{result.precision_at_k:.4f}",
                    f"{result.recall_at_k:.4f}",
                    f"{result.reciprocal_rank:.4f}",
                    f"{result.latency_ms:.2f}",
                ]
            )


def main() -> None:
    config = AppConfig.from_env()
    labeled_queries = load_labeled_queries(FIXTURES_PATH)
    retrievers = build_retrievers(config)

    all_results: list[QueryModeResult] = []
    summaries: dict[str, dict[str, float]] = {}

    for mode_name, retriever in retrievers.items():
        mode_results = run_mode(mode_name, retriever, labeled_queries, TOP_K)
        all_results.extend(mode_results)
        summaries[mode_name] = summarize(mode_results)

    write_csv(all_results, OUTPUT_CSV_PATH)

    header = (
        f"{'mode':<16}{'P@' + str(TOP_K):>10}{'R@' + str(TOP_K):>10}"
        f"{'MRR':>10}{'avg ms':>12}"
    )
    print(header)
    print("-" * len(header))
    for mode_name, summary in summaries.items():
        print(
            f"{mode_name:<16}"
            f"{summary['precision_at_k']:>10.3f}"
            f"{summary['recall_at_k']:>10.3f}"
            f"{summary['mrr']:>10.3f}"
            f"{summary['avg_latency_ms']:>12.1f}"
        )

    print(f"\nPer-query breakdown written to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
