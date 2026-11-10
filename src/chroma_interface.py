"""Embedding, indexing, persistence, and search through ChromaDB."""

from pathlib import Path
import logging
from typing import Sequence

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.documents import Document

from . import INDEX_PATH
from .base import Retriever
from .models import SearchResult


logger = logging.getLogger("ChromaInterface")

COLLECTION_NAME = "corpus"


class ChromaInterface(Retriever):
    """Own a Chroma collection holding one embedding per corpus chunk.

    A Retriever, like BestMatchingInterface: build_index() then search().

    - Embeddings: Chroma's default embedding function (all-MiniLM-L6-v2 run
      through ONNX, no torch needed). It is applied both when adding chunks and
      when querying, so documents and queries live in the same vector space.
    - Storage: a PersistentClient writes the collection under data/processed/chroma.
    - IDs: each chunk is stored under metadata["chunk_id"], the same ID the BM25
      corpus uses, so the hybrid retriever can fuse both rankings.
    """

    name = "dense"

    def __init__(self, index_path: str | Path = INDEX_PATH / "chroma") -> None:
        self.index_path = Path(index_path)
        self.client = chromadb.PersistentClient(path=str(self.index_path))
        self.embedding_function = DefaultEmbeddingFunction()

    def build_index(self, documents: Sequence[Document]) -> None:
        """Embed and store every chunk. Rebuilding replaces the previous collection."""
        corpus = list(documents)
        if not corpus:
            raise ValueError("Cannot build an index from an empty corpus.")

        # Start from a clean collection so stale chunks never survive a re-index.
        if COLLECTION_NAME in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(COLLECTION_NAME)
        collection = self.client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,  # type: ignore[arg-type]
            configuration={"hnsw": {"space": "cosine"}},
        )

        logger.info(f"Embedding {len(corpus)} chunks into Chroma...")
        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(corpus), batch_size):
            batch = corpus[start:start + batch_size]
            collection.add(
                ids=[doc.metadata["chunk_id"] for doc in batch],
                documents=[doc.page_content for doc in batch],
                metadatas=[doc.metadata for doc in batch],
            )
            logger.debug(f"Embedded {start + len(batch)}/{len(corpus)} chunks.")
        logger.info(f"Saved Chroma collection to {self.index_path}")

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to k chunks, most similar first. Score is cosine similarity."""
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[SearchResult]]:
        """Embed every query in one batch and query the collection once."""
        for query in queries:
            self._validate_query(query, k)
        if not queries:
            return []
        collection = self._get_collection()
        results = collection.query(
            query_texts=list(queries),
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns one list per query, in the order they were sent.
        return [
            [
                SearchResult(
                    document=Document(page_content=text, metadata=dict(metadata)),
                    score=1.0 - distance,  # cosine distance -> cosine similarity
                )
                for text, metadata, distance in zip(documents, metadatas, distances)
            ]
            for documents, metadatas, distances in zip(
                results["documents"],  # type: ignore[arg-type]
                results["metadatas"],  # type: ignore[arg-type]
                results["distances"],  # type: ignore[arg-type]
            )
        ]

    def _get_collection(self) -> chromadb.Collection:
        """Open the saved collection, or report that the index is missing."""
        try:
            return self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,  # type: ignore[arg-type]
            )
        except Exception as error:
            raise RuntimeError(
                f"Cannot load Chroma index from {self.index_path}: {error}. "
                "Run the index command to rebuild it."
            ) from error
