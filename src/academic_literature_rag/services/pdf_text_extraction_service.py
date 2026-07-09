from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pypdf import PdfReader

from academic_literature_rag.models.pdf_page_text import PdfPageText
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)
from academic_literature_rag.repositories.pdf_page_text_repository import (
    PdfPageTextRepository,
)


class PdfTextExtractionError(RuntimeError):
    """Raised when text cannot be extracted from a downloaded PDF."""


class PdfTextExtractionService:
    """Extracts page-level text from downloaded PDF assets."""

    def __init__(
        self,
        *,
        pdf_asset_repository: PdfAssetRepository,
        pdf_page_text_repository: PdfPageTextRepository,
    ) -> None:
        self._pdf_asset_repository = pdf_asset_repository
        self._pdf_page_text_repository = pdf_page_text_repository

    def extract(
        self,
        pdf_asset_id: UUID,
    ) -> list[PdfPageText]:
        """Extract text from one downloaded PDF asset and persist page text."""

        pdf_asset = self._pdf_asset_repository.get(pdf_asset_id)

        if pdf_asset is None:
            raise PdfTextExtractionError(f"PDF asset does not exist: {pdf_asset_id}")

        if pdf_asset.download_status != "downloaded":
            raise PdfTextExtractionError("PDF asset must be downloaded before text extraction.")

        if pdf_asset.local_file_path is None:
            raise PdfTextExtractionError("Downloaded PDF asset is missing a local file path.")

        pdf_path = Path(pdf_asset.local_file_path)

        if not pdf_path.exists():
            raise PdfTextExtractionError(f"Downloaded PDF file does not exist: {pdf_path}")

        page_texts = self._extract_page_texts(
            pdf_asset_id=pdf_asset.pdf_asset_id,
            pdf_path=pdf_path,
        )

        return self._pdf_page_text_repository.replace_for_pdf_asset(
            pdf_asset_id=pdf_asset.pdf_asset_id,
            page_texts=page_texts,
        )

    def _extract_page_texts(
        self,
        *,
        pdf_asset_id: UUID,
        pdf_path: Path,
    ) -> list[PdfPageText]:
        """Extract normalized text from each page of a PDF file."""

        try:
            reader = PdfReader(pdf_path)
        except Exception as error:
            raise PdfTextExtractionError(f"Failed to read PDF file: {pdf_path}") from error

        if not reader.pages:
            raise PdfTextExtractionError("PDF file does not contain any pages.")

        page_texts: list[PdfPageText] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as error:
                raise PdfTextExtractionError(
                    f"Failed to extract text from page {page_number}."
                ) from error

            page_texts.append(
                PdfPageText(
                    pdf_asset_id=pdf_asset_id,
                    page_number=page_number,
                    text=self._normalize_text(raw_text),
                )
            )

        if not any(page_text.text for page_text in page_texts):
            raise PdfTextExtractionError("PDF text extraction produced no readable text.")

        return page_texts

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        """Normalize extracted PDF text while preserving paragraph content."""

        lines = [" ".join(line.split()) for line in value.splitlines()]

        non_empty_lines = [line for line in lines if line]

        return "\n".join(non_empty_lines)
