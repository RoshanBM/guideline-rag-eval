"""The interface every retriever implements, so BM25, dense and hybrid retrievers can be run
through the same evaluation harness (``scripts/run_retrieval_eval.py``)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """A retriever over the indexed chunk corpus (``src/retrieve/corpus.py``)."""

    name: str

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """The top ``k`` chunks for ``query`` as ``(chunk_id, score)``, best first.

        May return fewer than ``k`` results (e.g. when fewer chunks match at all). Ties must be
        broken deterministically so that runs are reproducible.
        """
        ...
