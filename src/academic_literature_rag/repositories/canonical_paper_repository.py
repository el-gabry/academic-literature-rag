from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
    SourcePaperRecord,
)
from academic_literature_rag.identity.normalizers import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
)
from academic_literature_rag.models.canonical_paper import CanonicalPaper


class SourcePaperNotFoundError(LookupError):
    """Raised when a requested source-paper record does not exist."""


class CanonicalPaperIntegrityError(RuntimeError):
    """Raised when canonical-paper persistence finds inconsistent records."""


class CanonicalPaperMatchConflictError(RuntimeError):
    """Raised when strong identifiers point to different canonical papers."""


class CanonicalPaperRepository:
    """Persists and safely links internal canonical-paper records."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def create_for_source_paper(
        self,
        source_paper_id: UUID,
    ) -> CanonicalPaper:
        """Create a separate canonical paper for one source paper.

        This method intentionally does not search for duplicate records.
        It is retained for explicit creation and migration-style workflows.
        """

        with self._session_factory.begin() as session:
            source_paper = self._get_source_paper(
                session=session,
                source_paper_id=source_paper_id,
            )

            linked_record = self._get_linked_canonical_paper(
                session=session,
                source_paper=source_paper,
            )

            if linked_record is not None:
                return self._to_model(linked_record)

            return self._create_and_link(
                session=session,
                source_paper=source_paper,
            )

    def link_or_create_for_source_paper(
        self,
        source_paper_id: UUID,
    ) -> CanonicalPaper:
        """Link a source paper to a strong match or create a new paper.

        Automatic linking is allowed only when a normalized DOI or normalized
        arXiv identifier matches exactly. Titles are never enough on their own.
        """

        with self._session_factory.begin() as session:
            source_paper = self._get_source_paper(
                session=session,
                source_paper_id=source_paper_id,
            )

            linked_record = self._get_linked_canonical_paper(
                session=session,
                source_paper=source_paper,
            )

            if linked_record is not None:
                return self._to_model(linked_record)

            matching_record = self._find_strong_match(
                session=session,
                source_paper=source_paper,
            )

            if matching_record is not None:
                source_paper.canonical_paper_id = matching_record.id

                return self._to_model(matching_record)

            return self._create_and_link(
                session=session,
                source_paper=source_paper,
            )

    def get(
        self,
        canonical_paper_id: UUID,
    ) -> CanonicalPaper | None:
        """Return one canonical paper by its internal identifier."""

        with self._session_factory() as session:
            record = session.get(
                CanonicalPaperRecord,
                str(canonical_paper_id),
            )

        if record is None:
            return None

        return self._to_model(record)

    def list_source_paper_ids(
        self,
        canonical_paper_id: UUID,
    ) -> list[UUID]:
        """Return source-paper IDs linked to one canonical paper."""

        statement = (
            select(SourcePaperRecord.id)
            .where(SourcePaperRecord.canonical_paper_id == str(canonical_paper_id))
            .order_by(SourcePaperRecord.id)
        )

        with self._session_factory() as session:
            source_paper_ids = session.scalars(statement).all()

        return [UUID(source_paper_id) for source_paper_id in source_paper_ids]

    @staticmethod
    def _get_source_paper(
        *,
        session: Session,
        source_paper_id: UUID,
    ) -> SourcePaperRecord:
        source_paper = session.get(
            SourcePaperRecord,
            str(source_paper_id),
        )

        if source_paper is None:
            raise SourcePaperNotFoundError(f"Source paper does not exist: {source_paper_id}")

        return source_paper

    @staticmethod
    def _get_linked_canonical_paper(
        *,
        session: Session,
        source_paper: SourcePaperRecord,
    ) -> CanonicalPaperRecord | None:
        if source_paper.canonical_paper_id is None:
            return None

        record = session.get(
            CanonicalPaperRecord,
            source_paper.canonical_paper_id,
        )

        if record is None:
            raise CanonicalPaperIntegrityError("Source paper references a missing canonical paper.")

        return record

    def _find_strong_match(
        self,
        *,
        session: Session,
        source_paper: SourcePaperRecord,
    ) -> CanonicalPaperRecord | None:
        normalized_doi = normalize_doi(source_paper.doi)
        normalized_arxiv_id = normalize_arxiv_id(source_paper.arxiv_id)

        conditions = []

        if normalized_doi is not None:
            conditions.append(CanonicalPaperRecord.doi == normalized_doi)

        if normalized_arxiv_id is not None:
            conditions.append(CanonicalPaperRecord.arxiv_id == normalized_arxiv_id)

        if not conditions:
            return None

        statement = select(CanonicalPaperRecord).where(or_(*conditions))

        records = list(session.scalars(statement).all())

        if not records:
            return None

        unique_record_ids = {record.id for record in records}

        if len(unique_record_ids) > 1:
            raise CanonicalPaperMatchConflictError(
                "Strong identifiers point to different canonical papers."
            )

        return records[0]

    def _create_and_link(
        self,
        *,
        session: Session,
        source_paper: SourcePaperRecord,
    ) -> CanonicalPaper:
        normalized_title = normalize_title(source_paper.title)

        if normalized_title is None:
            raise CanonicalPaperIntegrityError("Source paper title cannot be normalized.")

        canonical_paper = CanonicalPaper(
            title=source_paper.title,
            normalized_title=normalized_title,
            doi=normalize_doi(source_paper.doi),
            arxiv_id=normalize_arxiv_id(source_paper.arxiv_id),
            authors=list(source_paper.authors_json),
            publication_year=source_paper.publication_year,
        )

        record = CanonicalPaperRecord(
            id=str(canonical_paper.canonical_paper_id),
            title=canonical_paper.title,
            normalized_title=canonical_paper.normalized_title,
            doi=canonical_paper.doi,
            arxiv_id=canonical_paper.arxiv_id,
            authors_json=canonical_paper.authors,
            publication_year=canonical_paper.publication_year,
            created_at=canonical_paper.created_at,
            updated_at=canonical_paper.updated_at,
        )

        session.add(record)

        source_paper.canonical_paper_id = str(canonical_paper.canonical_paper_id)

        return canonical_paper

    @staticmethod
    def _to_model(
        record: CanonicalPaperRecord,
    ) -> CanonicalPaper:
        """Convert one database record into the domain model."""

        return CanonicalPaper(
            canonical_paper_id=UUID(record.id),
            title=record.title,
            normalized_title=record.normalized_title,
            doi=record.doi,
            arxiv_id=record.arxiv_id,
            authors=list(record.authors_json),
            publication_year=record.publication_year,
            created_at=CanonicalPaperRepository._as_utc(record.created_at),
            updated_at=CanonicalPaperRepository._as_utc(record.updated_at),
        )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        """Return a timezone-aware UTC datetime from SQLite data."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)
