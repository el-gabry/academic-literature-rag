from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    SearchRunPaperRecord,
    SourcePaperRecord,
)
from academic_literature_rag.models.paper_candidate import PaperCandidate


class SourcePaperRepository:
    """Persists source-specific paper records and run associations."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def save_for_run(
        self,
        *,
        run_id: UUID,
        papers: list[PaperCandidate],
    ) -> list[UUID]:
        """Save papers, link them to one run, and return source-paper IDs."""

        source_paper_ids: list[UUID] = []

        with self._session_factory.begin() as session:
            for result_position, paper in enumerate(papers, start=1):
                record = session.scalar(
                    select(SourcePaperRecord).where(
                        SourcePaperRecord.source == paper.source,
                        SourcePaperRecord.source_id == paper.source_id,
                    )
                )

                if record is None:
                    record = self._create_record(paper)
                    session.add(record)
                else:
                    self._update_record(record, paper)

                link = session.get(
                    SearchRunPaperRecord,
                    (str(run_id), record.id),
                )

                if link is None:
                    session.add(
                        SearchRunPaperRecord(
                            run_id=str(run_id),
                            source_paper_id=record.id,
                            result_position=result_position,
                            retrieved_at=paper.retrieved_at,
                        )
                    )
                else:
                    link.result_position = result_position
                    link.retrieved_at = paper.retrieved_at

                source_paper_ids.append(UUID(record.id))

        return source_paper_ids

    def list_for_run(
        self,
        run_id: UUID,
    ) -> list[PaperCandidate]:
        """Return paper candidates found in one retrieval run."""

        statement = (
            select(SourcePaperRecord, SearchRunPaperRecord)
            .join(
                SearchRunPaperRecord,
                SearchRunPaperRecord.source_paper_id == SourcePaperRecord.id,
            )
            .where(SearchRunPaperRecord.run_id == str(run_id))
            .order_by(SearchRunPaperRecord.result_position)
        )

        with self._session_factory() as session:
            rows = session.execute(statement).all()

        return [
            PaperCandidate(
                source=record.source,
                source_id=record.source_id,
                title=record.title,
                landing_url=record.landing_url,
                retrieved_at=link.retrieved_at,
                abstract=record.abstract,
                authors=list(record.authors_json),
                publication_year=record.publication_year,
                venue=record.venue,
                doi=record.doi,
                arxiv_id=record.arxiv_id,
                open_access_pdf_url=record.open_access_pdf_url,
                citation_count=record.citation_count,
            )
            for record, link in rows
        ]

    @staticmethod
    def _create_record(
        paper: PaperCandidate,
    ) -> SourcePaperRecord:
        return SourcePaperRecord(
            id=str(uuid4()),
            source=paper.source,
            source_id=paper.source_id,
            title=paper.title,
            landing_url=paper.landing_url,
            abstract=paper.abstract,
            authors_json=paper.authors,
            publication_year=paper.publication_year,
            venue=paper.venue,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            open_access_pdf_url=paper.open_access_pdf_url,
            citation_count=paper.citation_count,
        )

    @staticmethod
    def _update_record(
        record: SourcePaperRecord,
        paper: PaperCandidate,
    ) -> None:
        record.title = paper.title
        record.landing_url = paper.landing_url
        record.abstract = paper.abstract
        record.authors_json = paper.authors
        record.publication_year = paper.publication_year
        record.venue = paper.venue
        record.doi = paper.doi
        record.arxiv_id = paper.arxiv_id
        record.open_access_pdf_url = paper.open_access_pdf_url
        record.citation_count = paper.citation_count
