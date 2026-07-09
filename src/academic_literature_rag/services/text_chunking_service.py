from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from academic_literature_rag.models.pdf_page_text import PdfPageText
from academic_literature_rag.models.text_chunk import TextChunk
from academic_literature_rag.repositories.pdf_page_text_repository import (
    PdfPageTextRepository,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.services.text_cleaning_service import (
    TextCleaningService,
)


class TextChunkingError(RuntimeError):
    """Raised when extracted PDF text cannot be chunked."""


@dataclass(frozen=True)
class TextUnit:
    """Small cleaned text unit linked to one PDF page."""

    page_number: int
    text: str


class TextChunkingService:
    """Cleans extracted PDF page text and stores retrieval-ready chunks."""

    def __init__(
        self,
        *,
        pdf_page_text_repository: PdfPageTextRepository,
        text_chunk_repository: TextChunkRepository,
        text_cleaning_service: TextCleaningService,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 characters.")

        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        self._pdf_page_text_repository = pdf_page_text_repository
        self._text_chunk_repository = text_chunk_repository
        self._text_cleaning_service = text_cleaning_service
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_pdf_asset(
        self,
        pdf_asset_id: UUID,
    ) -> list[TextChunk]:
        """Create and persist text chunks for one PDF asset."""

        page_texts = self._pdf_page_text_repository.list_for_pdf_asset(pdf_asset_id)

        if not page_texts:
            raise TextChunkingError(f"No extracted page text exists for PDF asset: {pdf_asset_id}")

        text_units = self._build_text_units(page_texts)

        if not text_units:
            raise TextChunkingError("Extracted page text produced no readable cleaned text.")

        chunks = self._build_chunks(
            pdf_asset_id=pdf_asset_id,
            text_units=text_units,
        )

        return self._text_chunk_repository.replace_for_pdf_asset(
            pdf_asset_id=pdf_asset_id,
            chunks=chunks,
        )

    def _build_text_units(
        self,
        page_texts: list[PdfPageText],
    ) -> list[TextUnit]:
        """Clean page text and split it into chunkable units."""

        text_units: list[TextUnit] = []

        for page_text in page_texts:
            cleaned_text = self._text_cleaning_service.clean(page_text.text)

            if not cleaned_text:
                continue

            for paragraph in self._split_into_paragraphs(cleaned_text):
                text_units.extend(
                    TextUnit(
                        page_number=page_text.page_number,
                        text=unit_text,
                    )
                    for unit_text in self._split_large_text(paragraph)
                )

        return text_units

    @staticmethod
    def _split_into_paragraphs(
        text: str,
    ) -> list[str]:
        """Split text by blank lines while keeping non-empty paragraphs."""

        return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    def _split_large_text(
        self,
        text: str,
    ) -> list[str]:
        """Split text that is larger than the configured chunk size."""

        if len(text) <= self._chunk_size:
            return [text]

        words = text.split()
        units: list[str] = []
        current_words: list[str] = []
        current_length = 0

        for word in words:
            word_length = len(word)
            separator_length = 1 if current_words else 0

            if current_words and current_length + separator_length + word_length > self._chunk_size:
                units.append(" ".join(current_words))
                current_words = [word]
                current_length = word_length
                continue

            current_words.append(word)
            current_length += separator_length + word_length

        if current_words:
            units.append(" ".join(current_words))

        return units

    def _build_chunks(
        self,
        *,
        pdf_asset_id: UUID,
        text_units: list[TextUnit],
    ) -> list[TextChunk]:
        """Build text chunks from cleaned text units."""

        chunks: list[TextChunk] = []
        current_units: list[TextUnit] = []

        for text_unit in text_units:
            candidate_units = [*current_units, text_unit]
            candidate_text = self._join_units(candidate_units)

            if current_units and len(candidate_text) > self._chunk_size:
                chunks.append(
                    self._create_chunk(
                        pdf_asset_id=pdf_asset_id,
                        chunk_index=len(chunks),
                        text_units=current_units,
                    )
                )
                current_units = self._overlap_units(current_units)

            current_units.append(text_unit)

        if current_units:
            chunks.append(
                self._create_chunk(
                    pdf_asset_id=pdf_asset_id,
                    chunk_index=len(chunks),
                    text_units=current_units,
                )
            )

        return chunks

    @staticmethod
    def _join_units(
        text_units: list[TextUnit],
    ) -> str:
        """Join chunk text units into one chunk body."""

        return "\n\n".join(text_unit.text for text_unit in text_units).strip()

    def _overlap_units(
        self,
        text_units: list[TextUnit],
    ) -> list[TextUnit]:
        """Keep a small amount of previous text for chunk overlap."""

        if self._chunk_overlap == 0:
            return []

        overlap_units: list[TextUnit] = []
        overlap_length = 0

        for text_unit in reversed(text_units):
            text_length = len(text_unit.text)

            if overlap_units and overlap_length + text_length > self._chunk_overlap:
                break

            if text_length > self._chunk_overlap:
                break

            overlap_units.append(text_unit)
            overlap_length += text_length

        return list(reversed(overlap_units))

    def _create_chunk(
        self,
        *,
        pdf_asset_id: UUID,
        chunk_index: int,
        text_units: list[TextUnit],
    ) -> TextChunk:
        """Create a domain text chunk from text units."""

        text = self._join_units(text_units)

        return TextChunk(
            pdf_asset_id=pdf_asset_id,
            chunk_index=chunk_index,
            start_page_number=text_units[0].page_number,
            end_page_number=text_units[-1].page_number,
            text=text,
            char_count=len(text),
        )
