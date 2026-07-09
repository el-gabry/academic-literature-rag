from __future__ import annotations

import re


class TextCleaningService:
    """Cleans extracted PDF text before chunking."""

    def clean(
        self,
        text: str,
    ) -> str:
        """Return normalized text suitable for chunking."""

        cleaned_text = self._normalize_line_breaks(text)
        cleaned_text = self._remove_null_bytes(cleaned_text)
        cleaned_text = self._join_hyphenated_line_breaks(cleaned_text)
        cleaned_text = self._normalize_inline_whitespace(cleaned_text)
        cleaned_text = self._normalize_lines(cleaned_text)

        return cleaned_text.strip()

    @staticmethod
    def _normalize_line_breaks(
        text: str,
    ) -> str:
        """Normalize Windows and old Mac line breaks."""

        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _remove_null_bytes(
        text: str,
    ) -> str:
        """Remove null bytes that can appear in extracted PDF text."""

        return text.replace("\x00", "")

    @staticmethod
    def _join_hyphenated_line_breaks(
        text: str,
    ) -> str:
        """Join words split by PDF line wrapping, e.g. trans-\nformer."""

        return re.sub(
            r"(?<=\w)-\n(?=\w)",
            "",
            text,
        )

    @staticmethod
    def _normalize_inline_whitespace(
        text: str,
    ) -> str:
        """Collapse spaces and tabs while preserving line boundaries."""

        return re.sub(
            r"[ \t]+",
            " ",
            text,
        )

    @staticmethod
    def _normalize_lines(
        text: str,
    ) -> str:
        """Strip lines and remove repeated empty lines."""

        lines = [line.strip() for line in text.splitlines()]

        normalized_lines: list[str] = []
        previous_line_was_empty = False

        for line in lines:
            if not line:
                if not previous_line_was_empty:
                    normalized_lines.append("")

                previous_line_was_empty = True
                continue

            normalized_lines.append(line)
            previous_line_was_empty = False

        return "\n".join(normalized_lines)
