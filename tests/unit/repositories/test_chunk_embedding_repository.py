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
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
    TextChunkForEmbeddingNotFoundError,
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
    """Insert one canonical-paper record directly for repository tests."""

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
    """Insert one downloaded PDF asset directly for repository tests."""

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
) -> UUID:
    """Insert one text chunk directly for repository tests."""

    resolved_pdf_asset_id = pdf_asset_id or create_pdf_asset(session_factory)
    text_chunk_id = uuid4()

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
                created_at=datetime.now(UTC),
            )
        )

    return text_chunk_id


def test_create_or_replace_saves_embedding(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    text_chunk_id = create_text_chunk(session_factory=session_factory)

    embedding = ChunkEmbedding(
        text_chunk_id=text_chunk_id,
        embedding_model="test-embedding-model",
        embedding_vector=[0.1, 0.2, 0.3],
        embedding_dimension=3,
    )

    saved_embedding = repository.create_or_replace(embedding)

    assert saved_embedding == embedding

    loaded_embedding = repository.get_for_text_chunk(
        text_chunk_id=text_chunk_id,
        embedding_model="test-embedding-model",
    )

    assert loaded_embedding == embedding


def test_create_or_replace_replaces_existing_embedding_for_same_model(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    text_chunk_id = create_text_chunk(session_factory=session_factory)

    first_embedding = ChunkEmbedding(
        text_chunk_id=text_chunk_id,
        embedding_model="test-embedding-model",
        embedding_vector=[0.1, 0.2, 0.3],
        embedding_dimension=3,
    )

    second_embedding = ChunkEmbedding(
        text_chunk_id=text_chunk_id,
        embedding_model="test-embedding-model",
        embedding_vector=[0.4, 0.5, 0.6],
        embedding_dimension=3,
    )

    repository.create_or_replace(first_embedding)
    repository.create_or_replace(second_embedding)

    loaded_embedding = repository.get_for_text_chunk(
        text_chunk_id=text_chunk_id,
        embedding_model="test-embedding-model",
    )

    assert loaded_embedding == second_embedding

    all_embeddings = repository.list_for_text_chunk(text_chunk_id)

    assert all_embeddings == [second_embedding]


def test_list_by_model_returns_all_embeddings_for_model(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    first_text_chunk_id = create_text_chunk(
        session_factory=session_factory,
        chunk_index=0,
        text="First chunk.",
    )

    second_text_chunk_id = create_text_chunk(
        session_factory=session_factory,
        chunk_index=1,
        text="Second chunk.",
    )

    third_text_chunk_id = create_text_chunk(
        session_factory=session_factory,
        chunk_index=2,
        text="Third chunk.",
    )

    first_embedding = ChunkEmbedding(
        text_chunk_id=first_text_chunk_id,
        embedding_model="target-model",
        embedding_vector=[0.1, 0.2, 0.3],
        embedding_dimension=3,
    )

    second_embedding = ChunkEmbedding(
        text_chunk_id=second_text_chunk_id,
        embedding_model="target-model",
        embedding_vector=[0.4, 0.5, 0.6],
        embedding_dimension=3,
    )

    other_model_embedding = ChunkEmbedding(
        text_chunk_id=third_text_chunk_id,
        embedding_model="other-model",
        embedding_vector=[0.7, 0.8, 0.9],
        embedding_dimension=3,
    )

    repository.create_or_replace(first_embedding)
    repository.create_or_replace(other_model_embedding)
    repository.create_or_replace(second_embedding)

    embeddings = repository.list_by_model("target-model")

    assert embeddings == [
        first_embedding,
        second_embedding,
    ]


def test_list_by_model_returns_empty_list_when_model_has_no_embeddings(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    assert repository.list_by_model("missing-model") == []


def test_list_by_model_rejects_empty_model_name(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    with pytest.raises(ValueError, match="Embedding model cannot be empty"):
        repository.list_by_model("   ")


def test_list_for_text_chunk_returns_all_models(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    text_chunk_id = create_text_chunk(session_factory=session_factory)

    first_embedding = ChunkEmbedding(
        text_chunk_id=text_chunk_id,
        embedding_model="model-a",
        embedding_vector=[0.1, 0.2],
        embedding_dimension=2,
    )

    second_embedding = ChunkEmbedding(
        text_chunk_id=text_chunk_id,
        embedding_model="model-b",
        embedding_vector=[0.3, 0.4],
        embedding_dimension=2,
    )

    repository.create_or_replace(second_embedding)
    repository.create_or_replace(first_embedding)

    assert repository.list_for_text_chunk(text_chunk_id) == [
        first_embedding,
        second_embedding,
    ]


def test_create_or_replace_rejects_missing_text_chunk(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    embedding = ChunkEmbedding(
        text_chunk_id=uuid4(),
        embedding_model="test-embedding-model",
        embedding_vector=[0.1, 0.2, 0.3],
        embedding_dimension=3,
    )

    with pytest.raises(TextChunkForEmbeddingNotFoundError):
        repository.create_or_replace(embedding)


def test_get_for_text_chunk_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    text_chunk_id = create_text_chunk(session_factory=session_factory)

    loaded_embedding = repository.get_for_text_chunk(
        text_chunk_id=text_chunk_id,
        embedding_model="missing-model",
    )

    assert loaded_embedding is None


def test_list_text_chunks_without_embedding_returns_missing_chunks(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    embedded_text_chunk_id = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=0,
        text="Embedded chunk.",
    )

    missing_text_chunk_id = create_text_chunk(
        session_factory=session_factory,
        pdf_asset_id=pdf_asset_id,
        chunk_index=1,
        text="Missing embedding chunk.",
    )

    repository.create_or_replace(
        ChunkEmbedding(
            text_chunk_id=embedded_text_chunk_id,
            embedding_model="test-embedding-model",
            embedding_vector=[0.1, 0.2, 0.3],
            embedding_dimension=3,
        )
    )

    missing_chunks = repository.list_text_chunks_without_embedding(
        embedding_model="test-embedding-model",
    )

    assert [chunk.text_chunk_id for chunk in missing_chunks] == [missing_text_chunk_id]


def test_list_text_chunks_without_embedding_is_model_specific(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    text_chunk_id = create_text_chunk(session_factory=session_factory)

    repository.create_or_replace(
        ChunkEmbedding(
            text_chunk_id=text_chunk_id,
            embedding_model="model-a",
            embedding_vector=[0.1, 0.2, 0.3],
            embedding_dimension=3,
        )
    )

    missing_chunks = repository.list_text_chunks_without_embedding(
        embedding_model="model-b",
    )

    assert [chunk.text_chunk_id for chunk in missing_chunks] == [text_chunk_id]


def test_list_text_chunks_without_embedding_respects_limit(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    first_text_chunk_id = create_text_chunk(
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

    missing_chunks = repository.list_text_chunks_without_embedding(
        embedding_model="test-embedding-model",
        limit=1,
    )

    assert len(missing_chunks) == 1
    assert missing_chunks[0].text_chunk_id == first_text_chunk_id


def test_list_text_chunks_without_embedding_rejects_invalid_limit(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    with pytest.raises(ValueError, match="at least 1"):
        repository.list_text_chunks_without_embedding(
            embedding_model="test-embedding-model",
            limit=0,
        )
