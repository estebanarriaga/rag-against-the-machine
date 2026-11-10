"""Abstract base classes shared by every retrieval component."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from . import PROJECT_ROOT
from .models import (
    MinimalSearchResults, MinimalSource, SearchResult, StudentSearchResults, UnansweredQuestion
)


class Retriever(ABC):
    """Anything that ranks chunks for a query: BM25, dense, hybrid.

    Subclasses implement build_index() and search(), and call
    self._validate_query() first in search(), so every engine rejects bad
    input the same way from the CLI.

    The Indexer builds every retriever from the same chunks, each carrying
    metadata["chunk_id"], so their rankings can be fused later.
    """
    name: str

    @staticmethod
    def _validate_query(query: str, k: int) -> None:
        """Raise ValueError on an empty query or a k below 1."""
        if not query:
            raise ValueError("Query cannot be empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")

    @abstractmethod
    def build_index(self, documents: Sequence[Document]) -> None:
        """Index the chunks and save them, replacing any previous index."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to k chunks, best first."""

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[SearchResult]]:
        """Run several queries, one ranking per query in the same order.

        Default: one search() per query. Override it when the engine can
        batch queries (it is what search_dataset() calls).
        """
        return [self.search(query, k) for query in queries]

    def search_dataset(
        self, questions: Sequence[UnansweredQuestion], k: int = 10
    ) -> StudentSearchResults:
        """Retrieve k sources for every dataset question."""
        rankings = self.search_many([question.question for question in questions], k)
        return StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[self._to_source(result) for result in ranking],
                )
                for question, ranking in zip(questions, rankings)
            ],
            k=k,
        )

    @staticmethod
    def _to_source(result: SearchResult) -> MinimalSource:
        """Turn a ranked chunk into a MinimalSource with a project-relative path."""
        metadata = result.document.metadata
        source_path = Path(metadata["file_path"])
        if source_path.is_absolute():
            source_path = source_path.relative_to(PROJECT_ROOT)
        return MinimalSource(
            file_path=source_path.as_posix(),
            first_character_index=metadata["first_character_index"],
            last_character_index=metadata["last_character_index"],
        )
