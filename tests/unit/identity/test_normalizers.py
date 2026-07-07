from __future__ import annotations

import pytest

from academic_literature_rag.identity.normalizers import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
)


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("10.1000/ABC.123", "10.1000/abc.123"),
        ("doi:10.1000/ABC.123", "10.1000/abc.123"),
        ("DOI: 10.1000/ABC.123", "10.1000/abc.123"),
        ("https://doi.org/10.1000/ABC.123", "10.1000/abc.123"),
        ("http://dx.doi.org/10.1000/ABC.123", "10.1000/abc.123"),
        ("  10.1000/ABC.123.  ", "10.1000/abc.123"),
        (None, None),
        ("", None),
        ("not a doi", None),
        ("10.1000", None),
    ],
)
def test_normalize_doi(
    raw_value: str | None,
    expected_value: str | None,
) -> None:
    assert normalize_doi(raw_value) == expected_value


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("2302.01204v1", "2302.01204"),
        ("2302.01204", "2302.01204"),
        ("arXiv:2302.01204v2", "2302.01204"),
        ("ARXIV: 2302.01204v12", "2302.01204"),
        ("https://arxiv.org/abs/2302.01204v2", "2302.01204"),
        ("https://arxiv.org/pdf/2302.01204v2.pdf", "2302.01204"),
        ("hep-th/9901001v3", "hep-th/9901001"),
        ("math.GT/0309136v1", "math.gt/0309136"),
        (None, None),
        ("", None),
        ("not-an-arxiv-id", None),
    ],
)
def test_normalize_arxiv_id(
    raw_value: str | None,
    expected_value: str | None,
) -> None:
    assert normalize_arxiv_id(raw_value) == expected_value


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        (
            "  Early-Warning Signals: A Review!  ",
            "early warning signals a review",
        ),
        (
            "Change–Point Detection for Continuous Glucose Monitoring",
            "change point detection for continuous glucose monitoring",
        ),
        (
            "Análisis de Señales en Datos Clínicos",
            "analisis de senales en datos clinicos",
        ),
        (
            "Machine_Learning: Methods & Applications",
            "machine learning methods applications",
        ),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_title(
    raw_value: str | None,
    expected_value: str | None,
) -> None:
    assert normalize_title(raw_value) == expected_value
