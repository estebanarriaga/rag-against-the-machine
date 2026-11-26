from enum import Enum

from langchain_core.documents import Document
from pydantic import BaseModel, Field
import uuid
from typing import List

class MinimalSource(BaseModel):
    """The MinimalSource model represents a single source of information:"""
    file_path: str
    first_character_index: int
    last_character_index: int

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
    str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

class RagDataset(BaseModel):
    """Represent a dataset of RAG questions"""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent the search results and an answer:"""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str

"""
The StudentSearchResults and StudentSearchResultsAndAnswer 
models represent search results and search results with answers:
"""
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int
class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int

class FileType(Enum):
    CODE = 'code'
    DOCS = 'docs'
    ALL = 'all'


class RetrievalMethod(Enum):
    """Which ranking the CLI uses: lexical, semantic, or both fused with RRF."""
    BM25 = 'bm25'
    DENSE = 'dense'
    HYBRID = 'hybrid'


class SearchResult(BaseModel):
    """A ranked corpus chunk and its score.

    The score scale depends on the retriever (BM25 score, cosine similarity,
    or RRF score), so only compare scores coming from the same retriever.
    """
    document: Document
    score: float


def chunk_id(file_path: str, first_character_index: int, last_character_index: int) -> str:
    """Stable chunk ID shared by every index: the same chunk always gets the same ID."""
    return f"{file_path}:{first_character_index}:{last_character_index}"
