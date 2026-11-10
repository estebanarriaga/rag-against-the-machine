"""Small helpers shared across the package."""

from .base import Retriever
from .bm_interface import BestMatchingInterface
from .chroma_interface import ChromaInterface
from .models import RetrievalMethod
from .retrieve import HybridRetriever


def get_retriever(method: str) -> Retriever:
    """Return the Retriever for a RetrievalMethod value: 'bm25', 'dense' or 'hybrid'.

    Only the indexes the method needs are loaded: 'bm25' never opens Chroma.
    """
    try:
        retrieval_method = RetrievalMethod(method)
    except ValueError:
        valid = ", ".join(m.value for m in RetrievalMethod)
        raise ValueError(f"Unknown retrieval method {method!r}, expected one of: {valid}") from None

    match retrieval_method:
        case RetrievalMethod.BM25:
            return BestMatchingInterface()
        case RetrievalMethod.DENSE:
            return ChromaInterface()
        case RetrievalMethod.HYBRID:
            return HybridRetriever(sparse=BestMatchingInterface(), dense=ChromaInterface())
