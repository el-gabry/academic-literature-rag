from __future__ import annotations


from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
    SourcePaperRecord,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.repositories.canonical_paper_repository import (
    CanonicalPaperRepository,
)


def build_repository(
    database_path: Path,
) -> tuple[
    CanonicalPaperRepository,
    sessionmaker[Session],
]:
    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return CanonicalPaperRepository(session_factory), session_factory


def create_source_paper(
    *,
    session_factory: object,
    source: str,
    source_id: str,
    title: str,
    doi: str | None = None,
    arxiv_id: str | None = None,
) -> UUID:
    source_paper_id = uuid4()

    with session_factory.begin() as session:
        session.add(
            SourcePaperRecord(
                id=str(source_paper_id),
                source=source,
                source_id=source_id,
                title=title,
                landing_url=f"https://example.org/{source}/{source_id}",
                abstract="Example abstract.",
                authors_json=["A. Researcher", "B. Scientist"],
                publication_year=2026,
                venue=None,
                doi=doi,
                arxiv_id=arxiv_id,
                open_access_pdf_url=None,
                citation_count=None,
            )
        )

    return source_paper_id


def test_create_for_source_paper_creates_and_links_canonical_paper(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    source_paper_id = create_source_paper(
        session_factory=session_factory,
        source="semantic_scholar",
        source_id="semantic-1",
        title="Early Warning Signals in Clinical Time-Series",
        doi="https://doi.org/10.1000/ABC.123",
        arxiv_id="arXiv:2302.01204v2",
    )

    canonical_paper = repository.create_for_source_paper(source_paper_id)

    assert canonical_paper.title == ("Early Warning Signals in Clinical Time-Series")
    assert canonical_paper.normalized_title == ("early warning signals in clinical time series")
    assert canonical_paper.doi == "10.1000/abc.123"
    assert canonical_paper.arxiv_id == "2302.01204"

    assert repository.get(canonical_paper.canonical_paper_id) == canonical_paper

    assert repository.list_source_paper_ids(canonical_paper.canonical_paper_id) == [source_paper_id]

    with session_factory() as session:
        source_paper = session.get(
            SourcePaperRecord,
            str(source_paper_id),
        )

    assert source_paper is not None
    assert source_paper.canonical_paper_id == str(canonical_paper.canonical_paper_id)


def test_create_for_source_paper_is_idempotent(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    source_paper_id = create_source_paper(
        session_factory=session_factory,
        source="arxiv",
        source_id="2302.01204v2",
        title="Laplacian Change Point Detection",
        arxiv_id="2302.01204v2",
    )

    first_canonical_paper = repository.create_for_source_paper(source_paper_id)
    second_canonical_paper = repository.create_for_source_paper(source_paper_id)

    assert first_canonical_paper.canonical_paper_id == second_canonical_paper.canonical_paper_id

    with session_factory() as session:
        record_count = session.scalar(select(func.count()).select_from(CanonicalPaperRecord))

    assert record_count == 1


def test_this_phase_does_not_merge_distinct_source_papers_yet(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    first_source_paper_id = create_source_paper(
        session_factory=session_factory,
        source="semantic_scholar",
        source_id="semantic-1",
        title="A Shared Paper",
        doi="10.1000/shared-paper",
    )

    second_source_paper_id = create_source_paper(
        session_factory=session_factory,
        source="arxiv",
        source_id="2302.01204v1",
        title="A Shared Paper",
        doi="doi:10.1000/shared-paper",
    )

    first_canonical_paper = repository.create_for_source_paper(first_source_paper_id)
    second_canonical_paper = repository.create_for_source_paper(second_source_paper_id)

    assert first_canonical_paper.canonical_paper_id != second_canonical_paper.canonical_paper_id

    with session_factory() as session:
        record_count = session.scalar(select(func.count()).select_from(CanonicalPaperRecord))

    assert record_count == 2
