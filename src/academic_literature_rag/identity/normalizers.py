from __future__ import annotations

import re
import unicodedata
from typing import Final


_DOI_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:https?://)?(?:dx\.)?doi\.org/",
    re.IGNORECASE,
)

_DOI_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^doi\s*:\s*",
    re.IGNORECASE,
)

_DOI_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^10\.\d{4,9}/\S+$",
    re.IGNORECASE,
)

_ARXIV_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/",
    re.IGNORECASE,
)

_ARXIV_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^arxiv\s*:\s*",
    re.IGNORECASE,
)

_ARXIV_MODERN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{4}\.\d{4,5}(?:v\d+)?\b",
    re.IGNORECASE,
)

_ARXIV_LEGACY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[a-z-]+(?:\.[a-z-]+)?/\d{7}(?:v\d+)?\b",
    re.IGNORECASE,
)

_ARXIV_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"v\d+$",
    re.IGNORECASE,
)

_TRAILING_REFERENCE_PUNCTUATION: Final[str] = ".,;:"


def normalize_doi(value: str | None) -> str | None:
    """Return a comparable DOI, or None when no valid DOI is available."""

    if value is None:
        return None

    cleaned_value = _normalize_whitespace(value)

    if not cleaned_value:
        return None

    cleaned_value = _DOI_PREFIX_PATTERN.sub("", cleaned_value)
    cleaned_value = _DOI_LABEL_PATTERN.sub("", cleaned_value)
    cleaned_value = cleaned_value.strip().rstrip(_TRAILING_REFERENCE_PUNCTUATION)

    if not _DOI_PATTERN.fullmatch(cleaned_value):
        return None

    return cleaned_value.casefold()


def normalize_arxiv_id(value: str | None) -> str | None:
    """Return a versionless arXiv identifier, or None when unavailable."""

    if value is None:
        return None

    cleaned_value = _normalize_whitespace(value)

    if not cleaned_value:
        return None

    cleaned_value = _ARXIV_URL_PATTERN.sub("", cleaned_value)
    cleaned_value = _ARXIV_LABEL_PATTERN.sub("", cleaned_value)
    cleaned_value = cleaned_value.removesuffix(".pdf")

    match = _ARXIV_MODERN_PATTERN.search(cleaned_value)

    if match is None:
        match = _ARXIV_LEGACY_PATTERN.search(cleaned_value)

    if match is None:
        return None

    return _ARXIV_VERSION_PATTERN.sub("", match.group(0)).casefold()


def normalize_title(value: str | None) -> str | None:
    """Return a lowercase, punctuation-free title for conservative matching."""

    if value is None:
        return None

    cleaned_value = _normalize_whitespace(value)

    if not cleaned_value:
        return None

    decomposed_value = unicodedata.normalize("NFKD", cleaned_value)

    without_diacritics = "".join(
        character for character in decomposed_value if not unicodedata.combining(character)
    )

    normalized_punctuation = re.sub(
        r"[\W_]+",
        " ",
        without_diacritics.casefold(),
        flags=re.UNICODE,
    )

    normalized_title = _normalize_whitespace(normalized_punctuation)

    return normalized_title or None


def _normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and remove surrounding whitespace."""

    return " ".join(value.split())
