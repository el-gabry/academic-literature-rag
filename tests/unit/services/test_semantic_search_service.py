from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
    PdfAssetRecord,
    TextChunkRecord,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.chunk_embedding import ChunkEmbedding
from academic_literature_rag.models.text_chunk import TextChunk
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.services.embedding_client import (
    EmbeddingResponse,
)
from academic_literature_rag.services.semantic_search_service import (
    SemanticSearchError,
    SemanticSearchService,
)


class FakeEmbeddingClient:
    """Fake embedding client for semantic search tests."""

    def __init__(
        self,
        *,
        model_name: str = "fake-embedding-model",
        response_model: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._response_model = response_model
        self.embedded_texts: list[str] = []

    @property
    def model_name(
        self,
    ) -> str:
        return self._model_name

    def embed_text(
        self,
        text: str,
    ) -> EmbeddingResponse:
        self.embedded_texts.append(text)

        normalized_text = text.lower()

        if "attention" in normalized_text:
            vector = [1.0, 0.0, 0.0]
        elif "recurrent" in normalized_text:
            vector = [0.0, 1.0, 0.0]
        else:
            vector = [0.0, 0.0, 1.0]

        return EmbeddingResponse(
            model=self._response_model or self._model_name,
            vector=vector,
        )


def build_repositories(
    database_path: Path,
) -> tuple[
    ChunkEmbeddingRepository,
    TextChunkRepository,
    sessionmaker[Session],
]:
    """Create isolated repositories for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return (
        ChunkEmbeddingRepository(session_factory),
        TextChunkRepository(session_factory),
        session_factory,
    )


def create_canonical_paper(
    session_factory: sessionmaker[Session],
) -> UUID:
    """Insert one canonical-paper record directly for service tests."""

    canonical_paper_id = uuid4()

    with session_factory.begin() as session:
        session.add(
            CanonicalPaperRecord(
                id=str(canonical_paper_id),
                title="Example Paper",
                normalized_title="example paper",
                doi="10.1000/example",
                arxiv_id=None,
                authors_json=["A. Researcher"],
                publication_year=2026,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    return canonical_paper_id


def create_pdf_asset(
    session_factory: sessionmaker[Session],
) -> UUID:
    """Insert one downloaded PDF asset directly for service tests."""

    canonical_paper_id = create_canonical_paper(session_factory)
    pdf_asset_id = uuid4()

    with session_factory.begin() as session:
        session.add(
            PdfAssetRecord(
                id=str(pdf_asset_id),
                canonical_paper_id=str(canonical_paper_id),
                source_paper_id=None,
                source_url="https://example.org/paper.pdf",
                download_status="downloaded",
                local_file_path="data/pdfs/example.pdf",
                sha256_checksum="a" * 64,
                content_type="application/pdf",
                file_size_bytes=1024,
                failure_message=None,
                created_at=datetime.now(UTC),
                downloaded_at=datetime.now(UTC),
            )
        )

    return pdf_asset_id


def create_text_chunk(
    *,
    session_factory: sessionmaker[Session],
    pdf_asset_id: UUID | None = None,
    chunk_index: int = 0,
    start_page_number: int = 1,
    end_page_number: int = 1,
    text: str = "Example chunk text.",
) -> TextChunk:
    """Insert one text chunk directly for service tests."""

    resolved_pdf_asset_id = pdf_asset_id or create_pdf_asset(session_factory)
    text_chunk_id = uuid4()
    created_at = datetime.now(UTC)

    with session_factory.begin() as session:
        session.add(
            TextChunkRecord(
                id=str(text_chunk_id),
                pdf_asset_id=str(resolved_pdf_asset_id),
                chunk_index=chunk_index,
                start_page_number=start_page_number,
                end_page_number=end_page_number,
                text=text,
                char_count=len(text),
                created_at=created_at,
            )
        )

    return TextChunk(
        text_chunk_id=text_chunk_id,
        pdf_asset_id=resolved_pdf_asset_id,
        chunk_index=chunk_index,
        start_page_number=start_page_number,
        end_page_number=end_page_number,
        text=text,
        char_count=len(text),
        created_at=created_at,
    )


def save_embedding(
    *,
    chunk_embedding_repository: ChunkEmbeddingRepository,
    text_chunk_id: UUID,
    embedding_model: str,
    vector: list[float],
) -> None:
    """Persist one chunk embedding."""

    chunk_embedding_repository.create_or_replace(
        ChunkEmbedding(
            text_chunk_id=text_chunk_id,
            embedding_model=embedding_model,
            embedding_vector=vector,
            embedding_dimension=len(vector),
        )
    )


def build_service(
    *,
    chunk_embedding_repository: ChunkEmbeddingRepository,
    text_chunk_repository: TextChunkRepository,
    embedding_client: FakeEmbeddingClient,
) -> SemanticSearchService:
    """Create the semantic search service."""

    return SemanticSearchService(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=embedding_client,
    )


def test_search_returns_ranked_chunks_by_similarity(
    tmp_path: Path,
) -> None:
    chunk_embedding_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    attention_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=0,
        start_page_number=2,
        end_page_number=2,
        text="The Transformer uses attention mechanisms.",
    )

    mixed_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=1,
        start_page_number=3,
        end_page_number=3,
        text="This chunk discusses attention and recurrent models.",
    )

    recurrent_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=2,
        start_page_number=4,
        end_page_number=4,
        text="This chunk discusses recurrent neural networks.",
    )

    save_embedding(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_id=attention_chunk.text_chunk_id,
        embedding_model="fake-embedding-model",
        vector=[1.0, 0.0, 0.0],
    )

    save_embedding(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_id=mixed_chunk.text_chunk_id,
        embedding_model="fake-embedding-model",
        vector=[0.7, 0.3, 0.0],
    )

    save_embedding(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_id=recurrent_chunk.text_chunk_id,
        embedding_model="fake-embedding-model",
        vector=[0.0, 1.0, 0.0],
    )

    embedding_client = FakeEmbeddingClient()

    service = build_service(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=embedding_client,
    )

    results = service.search(
        "What is attention?",
        top_k=3,
    )

    assert [result.text_chunk_id for result in results] == [
        attention_chunk.text_chunk_id,
        mixed_chunk.text_chunk_id,
        recurrent_chunk.text_chunk_id,
    ]

    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[0].chunk_index == 0
    assert results[0].start_page_number == 2
    assert results[0].end_page_number == 2
    assert results[0].embedding_model == "fake-embedding-model"
    assert results[0].text == "The Transformer uses attention mechanisms."

    assert embedding_client.embedded_texts == ["What is attention?"]


def test_search_respects_top_k(
    tmp_path: Path,
) -> None:
    chunk_embedding_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    first_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=0,
        text="Attention chunk.",
    )

    second_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=1,
        text="Mixed chunk.",
    )

    save_embedding(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_id=first_chunk.text_chunk_id,
        embedding_model="fake-embedding-model",
        vector=[1.0, 0.0, 0.0],
    )

    save_embedding(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_id=second_chunk.text_chunk_id,
        embedding_model="fake-embedding-model",
        vector=[0.5, 0.5, 0.0],
    )

    service = build_service(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=FakeEmbeddingClient(),
    )

    results = service.search(
        "attention",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].text_chunk_id == first_chunk.text_chunk_id


def test_search_uses_only_embeddings_for_client_model(
    tmp_path: Path,
) -> None:
    chunk_embedding_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    target_chunk = create_text_chunk(
        session_factory=session_factory,
        chunk_index=0,
        text="Target model chunk.",
    )

    other_chunk = create_text_chunk(
        session_factory=session_factory,
        chunk_index=1,
        text="Other model chunk.",
    )

    save_embedding(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_id=target_chunk.text_chunk_id,
        embedding_model="fake-embedding-model",
        vector=[1.0, 0.0, 0.0],
    )

    save_embedding(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_id=other_chunk.text_chunk_id,
        embedding_model="other-model",
        vector=[1.0, 0.0, 0.0],
    )

    service = build_service(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=FakeEmbeddingClient(),
    )

    results = service.search("attention")

    assert len(results) == 1
    assert results[0].text_chunk_id == target_chunk.text_chunk_id


def test_search_returns_empty_list_when_no_embeddings_exist(
    tmp_path: Path,
) -> None:
    chunk_embedding_repository, text_chunk_repository, _session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    service = build_service(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=FakeEmbeddingClient(),
    )

    assert service.search("attention") == []


def test_search_rejects_empty_query(
    tmp_path: Path,
) -> None:
    chunk_embedding_repository, text_chunk_repository, _session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    service = build_service(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=FakeEmbeddingClient(),
    )

    with pytest.raises(SemanticSearchError, match="query cannot be empty"):
        service.search("   ")


def test_search_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    chunk_embedding_repository, text_chunk_repository, _session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    service = build_service(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=FakeEmbeddingClient(),
    )

    with pytest.raises(SemanticSearchError, match="top_k"):
        service.search(
            "attention",
            top_k=0,
        )


def test_search_rejects_query_embedding_model_mismatch(
    tmp_path: Path,
) -> None:
    chunk_embedding_repository, text_chunk_repository, _session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    service = build_service(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=FakeEmbeddingClient(
            model_name="expected-model",
            response_model="different-model",
        ),
    )

    with pytest.raises(SemanticSearchError, match="does not match"):
        service.search("attention")
