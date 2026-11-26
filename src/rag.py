
import logging
import json
from pathlib import Path

from src.logger import setup_logging
from .indexing import Indexer
from .generation import Generator
from .evaluation import Evaluator
from .utils import get_retriever
from .context import Context
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResultsAndAnswer
)
import uuid


class RAG:
    """RAG command-line interface."""

    def __init__(self,
        debug: bool = True,
        verbose: bool = False,
        model_name: str = 'openai/qwen3:0.6b',
        # model_name: str = 'openai/Qwen/Qwen3-0.6B',
        temperature: float = 0.3,
        api_base: str = 'http://localhost:11434/v1',
        api_key: str = 'EMPTY',
        max_tokens: int = 2048,
        dspy_cache: bool = True,

        use_query_expansion: bool = False,
        use_hyde: bool = False,

        use_chroma: bool = False,
        chromadb_collection_name: str = 'chunks',
        chromadb_path: str = 'data/processed/chunks/chromadb',

        bm25_k1: float = 2.0,
        bm25_b: float = 0.75,

        processed_bm25_index_path: str = 'data/processed/bm25_index',
        processed_chunks_path: str = 'data/processed/chunks/contents.json',

        rrf_weights_bm25: float = 1.2,
        rrf_weights_chroma: float = 1.0,
        rrf_weights_bm25_expanded: float = .3,
        rrf_weights_chroma_expanded: float = .5,
        rrf_weights_HyDE: float = .7

    ) -> None:
        self.config = Context(
            debug=debug,
            verbose=verbose,
            model_name=model_name,
            temperature=temperature,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            dspy_cache=dspy_cache,
            use_query_expansion=use_query_expansion,
            use_hyde=use_hyde,
            use_chroma=use_chroma,
            chromadb_collection_name=chromadb_collection_name,
            chromadb_path=chromadb_path,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            processed_bm25_index_path=processed_bm25_index_path,
            processed_chunks_path=processed_chunks_path,
            rrf_weights_bm25=rrf_weights_bm25,
            rrf_weights_chroma=rrf_weights_chroma,
            rrf_weights_bm25_expanded=rrf_weights_bm25_expanded,
            rrf_weights_chroma_expanded=rrf_weights_chroma_expanded,
            rrf_weights_HyDE=rrf_weights_HyDE
        )

        self.logger = logging.getLogger("RAG")
        setup_logging(debug=debug)


    def index(
        self,
        max_chunk_size: int,
        method: str = "bm25"
    ):
        """Build the index for --method; hybrid builds BM25 and Chroma from the same chunks."""
        try:
            Indexer(
                corpus_path=self.config.raw_data_path,
                max_chunk_size=max_chunk_size,
                config=self.config,
                interface=get_retriever(method)
            ).create_index()
        except Exception as e:
            self.logger.error(f"Error occurred while creating index: {e}")
        print(f"Ingestion complete! Indices saved under data/processed/")


    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "bm25"
    ):
        """Search indexed documents."""
        try:
            results = get_retriever(
                    method
                ).search(
                    query=query,
                    k=k
                )
            for result in results:
                print(result.score, result.document.metadata)
        except Exception as e:
            self.logger.error(f"Error occurred while searching: {e}")


    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 5,
        method: str = "bm25",
    ):
        """Run retrieval against every dataset question."""
        retriever = get_retriever(method)
        dataset_file = Path(dataset_path)
        output_directory = Path(save_directory)

        questions = RagDataset.model_validate_json(
            dataset_file.read_text()
        ).rag_questions

        results = retriever.search_dataset(questions, k)

        output_directory.mkdir(parents=True, exist_ok=True)
        output_file = output_directory / dataset_file.name
        output_file.write_text(results.model_dump_json(indent=2) + "\n")

    def answer(self, query: str, k: int = 5, method: str = "bm25"):
        """Generate an answer for a query."""
        retrieved = get_retriever(method).search(query=query, k=k)
        sources = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=[
                MinimalSource(**result.document.metadata)
                for result in retrieved
            ],
        )
        response = Generator().generate(query=query, sources=sources)
        print(response.answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ):
        """Generate answers for a dataset."""
        generator = Generator()
        for result in json.loads(
            Path(student_search_results_path).read_text()
        )["search_results"]:
            sources = MinimalSearchResults(
                question_id=result["question_id"],
                question=result["question"],
                retrieved_sources=[
                    MinimalSource(**source) for source in result["retrieved_sources"]
                ],
            )
            print(sources)
            response = generator.generate(query=result["question"], sources=sources)
            result["answer"] = response.answer
        return StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(**result) for result in result["search_results"]
            ],
            k=result["k"],
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ):
        """Evaluate retrieval results."""
        evaluator = Evaluator()
        evaluator.evaluate(student_search_results_path, dataset_path)


# if __name__ == "__main__":
#     arguments = sys.argv[1:]
#     debug = any(argument in ("-debug", "--debug") for argument in arguments)
#     arguments = [argument for argument in arguments if argument not in ("-debug", "--debug")]

#     rag = RAG(debug=debug)
#     fire.Fire(rag, command=arguments)
