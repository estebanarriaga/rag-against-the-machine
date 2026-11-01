
import fire # type: ignore[import-untyped]
import json

from pathlib import Path
from .indexing import Indexer
from .retrieve import Retriever
from .generation import Generator
from .bm_index import BestMatchingInterface
from . import CORPUS_PATH, PROJECT_ROOT
from .models import MinimalSearchResults, MinimalSource


class RAG:
    """RAG command-line interface."""

    def __init__(self):
        self._interface = None

    @property
    def interface(self):
        if self._interface is None:
            self._interface = BestMatchingInterface()
        return self._interface


    def index(self, max_chunk_size: int):
        """Create the document index."""
        indexer = Indexer(
            CORPUS_PATH,
            max_chunk_size=max_chunk_size,
            interface=self.interface,
        )
        indexer.create_index()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5):
        """Search indexed documents."""
        retriever = Retriever(interface=self.interface)
        results = retriever.search(query=query, k=k)

        for result in results:
            print(result.score, result.document.metadata)
            print(result.document.page_content)

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 5,
    ):
        """Run retrieval against every dataset question."""
        dataset_file = Path(dataset_path)
        output_directory = Path(save_directory)

        questions = json.loads(
            dataset_file.read_text()
        )["rag_questions"]

        results = []

        for question in questions:
            print(question)
            sources = []

            for result in self.interface.search(
                question["question"], k
            ):
                metadata = result.document.metadata.copy()
                source_path = Path(metadata["file_path"])
                if source_path.is_absolute():
                    source_path = source_path.relative_to(PROJECT_ROOT)
                metadata["file_path"] = source_path.as_posix()
                sources.append(metadata)

            results.append({
                "question_id": question["question_id"],
                "question": question["question"],
                "retrieved_sources": sources,
            })

        output_directory.mkdir(
            parents=True, exist_ok=True
        )

        output_file = (
            output_directory / dataset_file.name
        )

        output_file.write_text(
            json.dumps(
                {"search_results": results, "k": k},
                indent=2,
            ) + "\n"
        )

    def answer(self, query: str, k: int = 5):
        """Generate an answer for a query."""
        retrieved = Retriever(interface=self.interface).search(query=query, k=k)
        sources = MinimalSearchResults(
            question=query,
            retrieved_sources=[
                MinimalSource(**result.document.metadata)
                for result in retrieved
            ],
        )
        response = Generator().generate(query=query, sources=sources)
        print(response.answer)
        return response

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ):
        """Generate answers for a dataset."""
        print("Results:", student_search_results_path)
        print("Save to:", save_directory)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ):
        """Evaluate retrieval results."""
        print("Results:", student_search_results_path)
        print("Dataset:", dataset_path)


if __name__ == "__main__":
    fire.Fire(RAG)
