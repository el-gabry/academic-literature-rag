from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
    PdfAssetRecord,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.pdf_page_text import PdfPageText
from academic_literature_rag.repositories.pdf_page_text_repository import (
    PdfPageTextRepository,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.services.text_chunking_service import (
    TextChunkingError,
    TextChunkingService,
)
from academic_literature_rag.services.text_cleaning_service import (
    TextCleaningService,
)


def build_repositories(
    database_path: Path,
) -> tuple[
    PdfPageTextRepository,
    TextChunkRepository,
    sessionmaker[Session],
]:
    """Create isolated repositories for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return (
        PdfPageTextRepository(session_factory),
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


def build_service(
    *,
    pdf_page_text_repository: PdfPageTextRepository,
    text_chunk_repository: TextChunkRepository,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> TextChunkingService:
    """Create the text chunking service."""

    return TextChunkingService(
        pdf_page_text_repository=pdf_page_text_repository,
        text_chunk_repository=text_chunk_repository,
        text_cleaning_service=TextCleaningService(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def save_page_texts(
    *,
    pdf_page_text_repository: PdfPageTextRepository,
    pdf_asset_id: UUID,
    page_texts: list[PdfPageText],
) -> None:
    """Persist extracted page text for one PDF asset."""

    pdf_page_text_repository.replace_for_pdf_asset(
        pdf_asset_id=pdf_asset_id,
        page_texts=page_texts,
    )


def test_chunk_pdf_asset_creates_and_persists_chunks(
    tmp_path: Path,
) -> None:
    pdf_page_text_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    save_page_texts(
        pdf_page_text_repository=pdf_page_text_repository,
        pdf_asset_id=pdf_asset_id,
        page_texts=[
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=1,
                text=(
                    "First paragraph about recurrent neural networks.\n\n"
                    "Second paragraph about transformer models."
                ),
            )
        ],
    )

    service = build_service(
        pdf_page_text_repository=pdf_page_text_repository,
        text_chunk_repository=text_chunk_repository,
    )

    chunks = service.chunk_pdf_asset(pdf_asset_id)

    assert len(chunks) == 1

    assert chunks[0].pdf_asset_id == pdf_asset_id
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_page_number == 1
    assert chunks[0].end_page_number == 1
    assert chunks[0].text == (
        "First paragraph about recurrent neural networks.\n\n"
        "Second paragraph about transformer models."
    )
    assert chunks[0].char_count == len(chunks[0].text)

    assert text_chunk_repository.list_for_pdf_asset(pdf_asset_id) == chunks


def test_chunk_pdf_asset_cleans_text_before_chunking(
    tmp_path: Path,
) -> None:
    pdf_page_text_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    save_page_texts(
        pdf_page_text_repository=pdf_page_text_repository,
        pdf_asset_id=pdf_asset_id,
        page_texts=[
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=1,
                text=("  The trans-\nformer   model\x00 uses\t self-attention.  "),
            )
        ],
    )

    service = build_service(
        pdf_page_text_repository=pdf_page_text_repository,
        text_chunk_repository=text_chunk_repository,
    )

    chunks = service.chunk_pdf_asset(pdf_asset_id)

    assert len(chunks) == 1
    assert chunks[0].text == "The transformer model uses self-attention."


def test_chunk_pdf_asset_splits_large_text_into_multiple_chunks(
    tmp_path: Path,
) -> None:
    pdf_page_text_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    first_paragraph = " ".join(["alpha"] * 30)
    second_paragraph = " ".join(["beta"] * 30)
    third_paragraph = " ".join(["gamma"] * 30)

    save_page_texts(
        pdf_page_text_repository=pdf_page_text_repository,
        pdf_asset_id=pdf_asset_id,
        page_texts=[
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=1,
                text=f"{first_paragraph}\n\n{second_paragraph}",
            ),
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=2,
                text=third_paragraph,
            ),
        ],
    )

    service = build_service(
        pdf_page_text_repository=pdf_page_text_repository,
        text_chunk_repository=text_chunk_repository,
        chunk_size=120,
        chunk_overlap=0,
    )

    chunks = service.chunk_pdf_asset(pdf_asset_id)

    assert len(chunks) > 1

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    assert chunks[0].start_page_number == 1
    assert chunks[-1].end_page_number == 2

    assert text_chunk_repository.list_for_pdf_asset(pdf_asset_id) == chunks


def test_chunk_pdf_asset_replaces_existing_chunks(
    tmp_path: Path,
) -> None:
    pdf_page_text_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    save_page_texts(
        pdf_page_text_repository=pdf_page_text_repository,
        pdf_asset_id=pdf_asset_id,
        page_texts=[
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=1,
                text="Old extracted text.",
            )
        ],
    )

    service = build_service(
        pdf_page_text_repository=pdf_page_text_repository,
        text_chunk_repository=text_chunk_repository,
    )

    first_chunks = service.chunk_pdf_asset(pdf_asset_id)

    save_page_texts(
        pdf_page_text_repository=pdf_page_text_repository,
        pdf_asset_id=pdf_asset_id,
        page_texts=[
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=1,
                text="New extracted text.",
            )
        ],
    )

    second_chunks = service.chunk_pdf_asset(pdf_asset_id)

    assert len(first_chunks) == 1
    assert len(second_chunks) == 1
    assert second_chunks[0].text == "New extracted text."
    assert text_chunk_repository.list_for_pdf_asset(pdf_asset_id) == second_chunks


def test_chunk_pdf_asset_rejects_missing_page_text(
    tmp_path: Path,
) -> None:
    pdf_page_text_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    service = build_service(
        pdf_page_text_repository=pdf_page_text_repository,
        text_chunk_repository=text_chunk_repository,
    )

    with pytest.raises(
        TextChunkingError,
        match="No extracted page text exists",
    ):
        service.chunk_pdf_asset(pdf_asset_id)


def test_chunk_pdf_asset_rejects_empty_cleaned_text(
    tmp_path: Path,
) -> None:
    pdf_page_text_repository, text_chunk_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pdf_asset(session_factory)

    save_page_texts(
        pdf_page_text_repository=pdf_page_text_repository,
        pdf_asset_id=pdf_asset_id,
        page_texts=[
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=1,
                text="   \n\n\t   ",
            )
        ],
    )

    service = build_service(
        pdf_page_text_repository=pdf_page_text_repository,
        text_chunk_repository=text_chunk_repository,
    )

    with pytest.raises(
        TextChunkingError,
        match="no readable cleaned text",
    ):
        service.chunk_pdf_asset(pdf_asset_id)


def test_chunking_service_rejects_invalid_configuration(
    tmp_path: Path,
) -> None:
    pdf_page_text_repository, text_chunk_repository, _session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    with pytest.raises(ValueError, match="at least 100"):
        build_service(
            pdf_page_text_repository=pdf_page_text_repository,
            text_chunk_repository=text_chunk_repository,
            chunk_size=99,
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        build_service(
            pdf_page_text_repository=pdf_page_text_repository,
            text_chunk_repository=text_chunk_repository,
            chunk_overlap=-1,
        )

    with pytest.raises(ValueError, match="smaller than chunk size"):
        build_service(
            pdf_page_text_repository=pdf_page_text_repository,
            text_chunk_repository=text_chunk_repository,
            chunk_size=200,
            chunk_overlap=200,
        )
