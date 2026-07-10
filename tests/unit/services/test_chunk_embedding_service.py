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
from academic_literature_rag.services.chunk_embedding_service import (
    ChunkEmbeddingError,
    ChunkEmbeddingService,
)
from academic_literature_rag.services.embedding_client import (
    EmbeddingResponse,
)


class FakeEmbeddingClient:
    """Fake embedding client for service tests."""

    def __init__(
        self,
        *,
        model_name: str = "fake-embedding-model",
        dimension: int = 3,
        fail_for_text: str | None = None,
        response_model: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._fail_for_text = fail_for_text
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

        if self._fail_for_text is not None and self._fail_for_text in text:
            raise RuntimeError("Fake embedding failure.")

        return EmbeddingResponse(
            model=self._response_model or self._model_name,
            vector=[
                float(len(text)),
                float(self._dimension),
                1.0,
            ],
        )


def build_repository(
    database_path: Path,
) -> tuple[
    ChunkEmbeddingRepository,
    sessionmaker[Session],
]:
    """Create an isolated chunk embedding repository for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return ChunkEmbeddingRepository(session_factory), session_factory


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
                start_page_number=1,
                end_page_number=1,
                text=text,
                char_count=len(text),
                created_at=created_at,
            )
        )

    return TextChunk(
        text_chunk_id=text_chunk_id,
        pdf_asset_id=resolved_pdf_asset_id,
        chunk_index=chunk_index,
        start_page_number=1,
        end_page_number=1,
        text=text,
        char_count=len(text),
        created_at=created_at,
    )


def build_service(
    *,
    repository: ChunkEmbeddingRepository,
    embedding_client: FakeEmbeddingClient,
) -> ChunkEmbeddingService:
    """Create the chunk embedding service."""

    return ChunkEmbeddingService(
        chunk_embedding_repository=repository,
        embedding_client=embedding_client,
    )


def test_embed_text_chunk_generates_and_stores_embedding(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    text_chunk = create_text_chunk(
        session_factory=session_factory,
        text="Transformer attention chunk.",
    )

    embedding_client = FakeEmbeddingClient()

    service = build_service(
        repository=repository,
        embedding_client=embedding_client,
    )

    embedding = service.embed_text_chunk(text_chunk)

    assert embedding.text_chunk_id == text_chunk.text_chunk_id
    assert embedding.embedding_model == "fake-embedding-model"
    assert embedding.embedding_dimension == 3
    assert embedding.embedding_vector == [28.0, 3.0, 1.0]

    loaded_embedding = repository.get_for_text_chunk(
        text_chunk_id=text_chunk.text_chunk_id,
        embedding_model="fake-embedding-model",
    )

    assert loaded_embedding == embedding
    assert embedding_client.embedded_texts == ["Transformer attention chunk."]


def test_embed_missing_chunks_embeds_only_missing_chunks(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    already_embedded_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=0,
        text="Already embedded chunk.",
    )

    missing_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=1,
        text="Missing embedding chunk.",
    )

    repository.create_or_replace(
        ChunkEmbedding(
            text_chunk_id=already_embedded_chunk.text_chunk_id,
            embedding_model="fake-embedding-model",
            embedding_vector=[0.1, 0.2, 0.3],
            embedding_dimension=3,
        )
    )

    embedding_client = FakeEmbeddingClient()

    service = build_service(
        repository=repository,
        embedding_client=embedding_client,
    )

    results = service.embed_missing_chunks()

    assert len(results) == 1
    assert results[0].text_chunk_id == missing_chunk.text_chunk_id
    assert results[0].status == "embedded"
    assert results[0].embedding_model == "fake-embedding-model"
    assert results[0].error_message is None

    assert embedding_client.embedded_texts == ["Missing embedding chunk."]


def test_embed_missing_chunks_respects_limit(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    first_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=0,
        text="First missing chunk.",
    )

    create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=1,
        text="Second missing chunk.",
    )

    embedding_client = FakeEmbeddingClient()

    service = build_service(
        repository=repository,
        embedding_client=embedding_client,
    )

    results = service.embed_missing_chunks(limit=1)

    assert len(results) == 1
    assert results[0].text_chunk_id == first_chunk.text_chunk_id
    assert results[0].status == "embedded"

    remaining_missing_chunks = repository.list_text_chunks_without_embedding(
        embedding_model="fake-embedding-model",
    )

    assert len(remaining_missing_chunks) == 1


def test_embed_missing_chunks_continues_after_failure(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    failing_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=0,
        text="This chunk should fail.",
    )

    successful_chunk = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=1,
        text="This chunk should succeed.",
    )

    embedding_client = FakeEmbeddingClient(
        fail_for_text="fail",
    )

    service = build_service(
        repository=repository,
        embedding_client=embedding_client,
    )

    results = service.embed_missing_chunks()

    results_by_id = {result.text_chunk_id: result for result in results}

    assert results_by_id[failing_chunk.text_chunk_id].status == "failed"
    assert results_by_id[failing_chunk.text_chunk_id].error_message is not None

    assert results_by_id[successful_chunk.text_chunk_id].status == "embedded"
    assert results_by_id[successful_chunk.text_chunk_id].error_message is None

    assert (
        repository.get_for_text_chunk(
            text_chunk_id=failing_chunk.text_chunk_id,
            embedding_model="fake-embedding-model",
        )
        is None
    )

    assert (
        repository.get_for_text_chunk(
            text_chunk_id=successful_chunk.text_chunk_id,
            embedding_model="fake-embedding-model",
        )
        is not None
    )


def test_embed_text_chunk_rejects_model_mismatch(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    text_chunk = create_text_chunk(session_factory=session_factory)

    embedding_client = FakeEmbeddingClient(
        model_name="expected-model",
        response_model="different-model",
    )

    service = build_service(
        repository=repository,
        embedding_client=embedding_client,
    )

    with pytest.raises(
        ChunkEmbeddingError,
        match="does not match",
    ):
        service.embed_text_chunk(text_chunk)


def test_embed_missing_chunks_returns_empty_list_when_no_missing_chunks(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    embedding_client = FakeEmbeddingClient()

    service = build_service(
        repository=repository,
        embedding_client=embedding_client,
    )

    assert service.embed_missing_chunks() == []
