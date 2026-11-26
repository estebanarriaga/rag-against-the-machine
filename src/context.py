"""Run-wide settings shared by every RAG component.

Build one Context at startup and pass it to the classes that need it.
In tests, derive variants with `ctx.model_copy(update={...})`.
"""
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

class Context(BaseModel):
    """General values the program needs, in one validated place."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Input data (must exist before the program runs)
    raw_data_path: Path = Path('data/raw/vllm-0.10.1')
    unanswered_questions_path: Path = Path('data/datasets/public/UnansweredQuestions')
    answered_questions_path: Path = Path('data/datasets/public/AnsweredQuestions')

    # Logging
    debug: bool = False
    verbose: bool = False

    # LLM
    model_name: str = 'openai/qwen3:0.6b'
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    api_base: str = 'http://localhost:11434/v1'
    api_key: str = 'EMPTY'
    max_tokens: int = Field(default=2048, gt=0)
    dspy_cache: bool = True

    # Query rewriting
    use_query_expansion: bool = False
    use_hyde: bool = False

    # Dense index (Chroma)
    use_chroma: bool = False
    chromadb_collection_name: str = 'chunks'
    chromadb_path: str = 'data/processed/chunks/chromadb'

    # Sparse index (BM25)
    bm25_k1: float = Field(default=2.0, ge=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    processed_bm25_index_path: str = 'data/processed/bm25_index'
    processed_chunks_path: str = 'data/processed/chunks/contents.json'

    # Reciprocal rank fusion weights
    rrf_weights_bm25: float = Field(default=1.2, ge=0.0)
    rrf_weights_chroma: float = Field(default=1.0, ge=0.0)
    rrf_weights_bm25_expanded: float = Field(default=0.3, ge=0.0)
    rrf_weights_chroma_expanded: float = Field(default=0.5, ge=0.0)
    rrf_weights_HyDE: float = Field(default=0.7, ge=0.0)

    @staticmethod
    def _check_path(path: Path, is_dir: bool) -> None:
        """Raise if `path` is missing or is not the expected kind (dir or file)."""
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        if is_dir and not path.is_dir():
            raise ValueError(f"Expected a directory: {path}")
        if not is_dir and not path.is_file():
            raise ValueError(f"Expected a file: {path}")

    @model_validator(mode='after')
    def _check_input_paths(self) -> Self:
        """Check every input path in order; the first bad one stops validation."""
        checks: list[tuple[Path, bool]] = [
            (self.raw_data_path, True),
            (self.unanswered_questions_path, True),
            (self.answered_questions_path, True),
        ]
        for path, is_dir in checks:
            self._check_path(path, is_dir)
        return self
