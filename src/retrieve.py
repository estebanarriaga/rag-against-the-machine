"""Reciprocal Rank Fusion and the hybrid retriever."""

from collections import defaultdict
import logging
from typing import Sequence

from langchain_core.documents import Document

from .base import Retriever
from .models import SearchResult


logger = logging.getLogger("Retriever")


def rrf(
    rankings: list[list[SearchResult]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[SearchResult]:
    """Fuse several rankings with weighted Reciprocal Rank Fusion.

        RRF(d) = sum over rankings r of  w_r / (k + rank_r(d))

    Only the rank of each chunk is used, never its raw score: a BM25 score
    and a cosine similarity are on different scales and cannot be compared.
    A chunk missing from a ranking adds 0 for that ranking. Chunks are matched
    across rankings by metadata["chunk_id"].
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("One weight per ranking required.")

    scores: dict[str, float] = defaultdict(float)
    results: dict[str, SearchResult] = {}

    for ranking, weight in zip(rankings, weights):
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.document.metadata["chunk_id"]
            scores[chunk_id] += weight / (k + rank)
            results.setdefault(chunk_id, result)

    fused = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        SearchResult(document=results[chunk_id].document, score=scores[chunk_id])
        for chunk_id in fused
    ]


class HybridRetriever(Retriever):
    """Run a sparse (BM25) and a dense (embeddings) search, then fuse them with RRF.

    It owns no index of its own: build_index() builds both branches from the
    same chunks, so their chunk_ids match when the rankings are fused.

    Each branch returns more candidates than k (`candidates`), so a chunk ranked
    low by one retriever but high by the other can still reach the final top-k.
    """

    name = "hybrid"

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        candidates: int = 50,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self.candidates = candidates
        self.weights = weights

    def build_index(self, documents: Sequence[Document]) -> None:
        """Build the sparse and the dense index from the exact same chunks."""
        corpus = list(documents)
        logger.info(f"Building {self.sparse.name} index...")
        self.sparse.build_index(corpus)
        logger.info(f"Building {self.dense.name} index...")
        self.dense.build_index(corpus)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to k chunks fused from the sparse and dense rankings."""
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[SearchResult]]:
        """Batch each branch over all queries, then fuse the rankings per query."""
        for query in queries:
            self._validate_query(query, k)
        depth = max(k, self.candidates)
        sparse_rankings = self.sparse.search_many(queries, k=depth)
        dense_rankings = self.dense.search_many(queries, k=depth)
        logger.debug(
            f"Fusing {self.sparse.name} and {self.dense.name} rankings "
            f"for {len(queries)} queries."
        )
        return [
            rrf([sparse_results, dense_results], weights=list(self.weights))[:k]
            for sparse_results, dense_results in zip(sparse_rankings, dense_rankings)
        ]
