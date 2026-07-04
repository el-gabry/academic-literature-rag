from __future__ import annotations

from pydantic import BaseModel

from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.models.search_run import SearchRun


class RetrievalResult(BaseModel):
    """The normalized papers and metadata produced by one retrieval run."""

    run: SearchRun
    papers: list[PaperCandidate]
