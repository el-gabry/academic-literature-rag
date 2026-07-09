from __future__ import annotations

from academic_literature_rag.services.text_cleaning_service import (
    TextCleaningService,
)


def test_clean_removes_outer_whitespace() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("   Example text.   ")

    assert cleaned_text == "Example text."


def test_clean_normalizes_line_breaks() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("First line\r\nSecond line\rThird line")

    assert cleaned_text == "First line\nSecond line\nThird line"


def test_clean_removes_null_bytes() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("First\x00 text\x00.")

    assert cleaned_text == "First text."


def test_clean_joins_hyphenated_line_breaks() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("The trans-\nformer architecture.")

    assert cleaned_text == "The transformer architecture."


def test_clean_keeps_normal_hyphenated_words() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("Self-attention is useful.")

    assert cleaned_text == "Self-attention is useful."


def test_clean_collapses_spaces_and_tabs() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("This    text\t\tcontains   spaces.")

    assert cleaned_text == "This text contains spaces."


def test_clean_strips_each_line() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("  First line  \n  Second line  ")

    assert cleaned_text == "First line\nSecond line"


def test_clean_collapses_repeated_empty_lines() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("First paragraph.\n\n\n\nSecond paragraph.")

    assert cleaned_text == "First paragraph.\n\nSecond paragraph."


def test_clean_handles_mixed_pdf_artifacts() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean(
        "  Recurrent   neural\t networks\r\n"
        "\r\n"
        "long short-term memo-\n"
        "ry models\x00   are useful.\n\n\n"
        "  Transformers use self-attention.  "
    )

    assert cleaned_text == (
        "Recurrent neural networks\n\n"
        "long short-term memory models are useful.\n\n"
        "Transformers use self-attention."
    )


def test_clean_returns_empty_string_for_empty_input() -> None:
    service = TextCleaningService()

    cleaned_text = service.clean("   \n\n\t  ")

    assert cleaned_text == ""
