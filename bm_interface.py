"""Tokenization, indexing, persistence, and search through bm25s."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import bm25s
from bm25s.tokenization import Tokenized
from langchain_core.documents import Document


DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "datasets_public" / "raw"


@dataclass(frozen=True)
class SearchResult:
    """A ranked corpus chunk and its BM25 score."""

    document: Document
    score: float


class BestMatchingInterface:
    """Own a BM25 retriever and the documents associated with its index.

    Building an index automatically saves it, including corpus.jsonl with each
    chunk's text and metadata. Rebuilding replaces the previous index.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_path: str | Path = DEFAULT_INDEX_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self._corpus: list[Document] = []

    def tokenize(self, texts: str | Sequence[str] | Sequence[Document]) -> Tokenized:
        """Tokenize a query, a batch of queries, or corpus documents consistently."""
        if isinstance(texts, str):
            texts = [texts]
        content = [item.page_content if isinstance(item, Document) else item for item in texts]
        return bm25s.tokenize(content, stopwords="en", show_progress=False)

    def build_index(self, documents: Sequence[Document]) -> bm25s.BM25:
        """Index corpus chunks, save the index and corpus, and return the retriever."""
        corpus = list(documents)
        if not corpus:
            raise ValueError("Cannot build an index from an empty corpus.")
        tokens = self.tokenize(corpus)
        if not tokens.vocab:
            raise ValueError("The corpus contains no searchable tokens.")
        self.retriever.index(tokens, show_progress=False)
        self._corpus = corpus
        self.store_index()
        return self.retriever

    def store_index(self, path: str | Path | None = None) -> Path:
        """Save the current index and corpus text/metadata to the given directory."""
        self._require_index()
        destination = Path(path) if path is not None else self.index_path
        corpus = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self._corpus
        ]
        self.retriever.save(str(destination), corpus=corpus, show_progress=False)
        return destination

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to k matching chunks in descending score order."""
        self._require_index()
        if k < 1:
            raise ValueError("k must be at least 1.")
        tokens = self.tokenize(query)
        if not tokens.vocab:
            return []
        indices, scores = self.retriever.retrieve(
            tokens, k=min(k, len(self._corpus)), show_progress=False
        )
        return [
            SearchResult(document=self._corpus[int(index)], score=float(score))
            for index, score in zip(indices[0], scores[0])
        ]

    def _require_index(self) -> None:
        if not self._corpus:
            raise RuntimeError("Build an index with build_index() first.")
