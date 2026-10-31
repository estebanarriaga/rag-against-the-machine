"""Tokenization, indexing, persistence, and search through bm25s."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Sequence

import bm25s
from langchain_core.documents import Document


DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "data" / "processed"


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
        expand_identifiers: bool = True,
    ) -> None:
        self.index_path = Path(index_path)
        self.expand_identifiers = expand_identifiers
        self.retriever = bm25s.BM25(k1=k1, b=b)

    def tokenize(self, texts: str | Sequence[str] | Sequence[Document]) -> bm25s.tokenization.Tokenized:
        """Tokenize a query, a batch of queries, or corpus documents consistently."""
        if isinstance(texts, str):
            texts = [texts]
        content = []
        for item in texts:
            if isinstance(item, Document):
                # Search filenames without altering the original source text or offsets.
                filename = Path(item.metadata.get("file_path", "")).stem
                text = filename + "\n" + item.page_content
            else:
                text = item
            if self.expand_identifiers:
                # Preserve exact identifiers and add words for natural-language queries.
                words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
                text += "\n" + words
            content.append(text)
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
        self.retriever.corpus = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in corpus
        ]
        self.store_index()
        return self.retriever

    def store_index(self, path: str | Path | None = None) -> Path:
        """Save the current index and corpus text/metadata to the given directory."""
        if self.retriever.corpus is None:
            raise RuntimeError("No in-memory index to save. Build or load an index first.")
        destination = Path(path) if path is not None else self.index_path
        self.retriever.save(str(destination), show_progress=False)
        (destination / "tokenization.json").write_text(
            json.dumps({"expand_identifiers": self.expand_identifiers}) + "\n"
        )
        return destination

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to k matching chunks in descending score order."""
        self._ensure_index_loaded()
        if k < 1:
            raise ValueError("k must be at least 1.")
        tokens = self.tokenize(query)
        if not tokens.vocab:
            return []
        documents, scores = self.retriever.retrieve(
            tokens, k=min(k, len(self.retriever.corpus)), show_progress=False
        )
        return [
            SearchResult(document=Document(**document), score=float(score))
            for document, score in zip(documents[0], scores[0])
        ]

    def _ensure_index_loaded(self) -> None:
        """Load the saved index and corpus on the first search, or report missing files."""
        if self.retriever.corpus is not None:
            return
        corpus_file = self.index_path / "corpus.jsonl"
        if not corpus_file.is_file():
            raise RuntimeError("Missing index file. Run the index command first.")
        try:
            self.retriever = self.retriever.load(
                str(self.index_path),
                load_corpus=True,
                show_progress=False
            )
            
        except Exception as error:
            raise RuntimeError(
                f"Cannot load index from {self.index_path}: {error}. Run the index command to rebuild it."
            ) from error
