"""Tokenization, indexing, persistence, and search through bm25s."""

import json
from pathlib import Path
import re
from typing import Sequence

import bm25s # type: ignore[import-untyped]
from langchain_core.documents import Document
import logging
from . import INDEX_PATH
from .base import Retriever
from .models import SearchResult


logger = logging.getLogger("BM25Interface")


class BestMatchingInterface(Retriever):
    """Own a BM25 retriever and the documents associated with its index.

    Building an index automatically saves it, including corpus.jsonl with each
    chunk's text and metadata. Rebuilding replaces the previous index.
    """

    name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_path: str | Path = INDEX_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.retriever = bm25s.BM25(k1=k1, b=b)

    def tokenize(self, texts: str | list[str] | list[Document]) -> bm25s.tokenization.Tokenized:
        """

        Tokenize a query, a batch of queries, or corpus documents consistently.

        """
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


            # Preserve exact identifiers and add words for natural-language queries.
            # I would like this in its own class, to expand identifiers, and textual surgery
            # And there we will also use a stemmer for markdown files or python files, but
            # depending on the place: comments, context, etc. Same with markdown
            # This area is called text normalization.
            """
            Identifier expansion is a small example of a much larger family of techniques.
            They involve text normalization, linguistic analysis, semantic enrichment,
            query expansion, document representation, and even structural analysis of source code.
            """

            words = re.sub(
                r"([a-z0-9])([A-Z])",
                r"\1 \2",
                text
                ).replace("_", " ")
            text += "\n" + words
            content.append(text)
        logger.debug(f"Tokenizing {len(content)} items with expanded identifiers.")
        return (
            bm25s.tokenize(
                content,
                stopwords="en",
                show_progress=False
                )
        )

    def build_index(self, documents: Sequence[Document]) -> None:
        """

        Index corpus chunks, then save the index and corpus.

        """
        corpus = list(documents)
        if not corpus:
            raise ValueError("Cannot build an index from an empty corpus.")

        logger.info("Building index...")
        tokens = self.tokenize(corpus)
        if not tokens.vocab:
            raise ValueError("The corpus contains no searchable tokens.")

        self.retriever.index(
            tokens,
            show_progress=False
            )
        self.retriever.corpus = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in corpus
        ]
        self.store_index()

    def store_index(self, path: str | Path | None = None) -> Path:
        """

        Save the current index and corpus text/metadata to the given directory.

        """
        if self.retriever.corpus is None:
            raise RuntimeError("No in-memory index to save. Build or load an index first.")

        destination = Path(path) if path is not None else self.index_path
        logger.info(f"Saving index and corpus to {destination}")
        self.retriever.save(
            str(destination),
            show_progress=False
            )
        return destination

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to k matching chunks in descending score order."""
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[SearchResult]]:
        """Tokenize every query once and score them all in a single bm25s call."""
        for query in queries:
            self._validate_query(query, k)
        if not queries:
            return []
        self._ensure_index_loaded()

        documents, scores = self.retriever.retrieve(
            self.tokenize(list(queries)),
            k=min(k, len(self.retriever.corpus)),
            show_progress=False
        )
        return [
            [
                SearchResult(document=Document(**document), score=float(score))
                for document, score in zip(query_documents, query_scores)
            ]
            for query_documents, query_scores in zip(documents, scores)
        ]

    def _ensure_index_loaded(self) -> None:
        """Load the saved index and corpus on the first search, or report missing files."""
        if self.retriever.corpus is not None:
            return

        corpus_file = self.index_path / "corpus.jsonl"
        if not corpus_file.is_file():
            raise RuntimeError("Missing index file. Run the index command first.")

        try:
            retriever = bm25s.BM25.load(
                str(self.index_path),
                load_corpus=True,
                show_progress=False
            )
            if not retriever.corpus or len(retriever.corpus) != retriever.scores["num_docs"]:
                raise ValueError("Saved corpus size does not match the index.")

        except Exception as error:
            raise RuntimeError(
                f"Cannot load index from {self.index_path}: {error}. Run the index command to rebuild it."
            ) from error
        self.retriever = retriever
