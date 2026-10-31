
import fire # type: ignore[import-untyped]
import dspy # type: ignore[import-untyped]
import json

from pathlib import Path
from index_module import Indexer
from retriever_module import Retriever
from bm_interface import BestMatchingInterface


class RAG:
    """RAG command-line interface."""

    def __init__(self):
        self._interface = None

    @property
    def interface(self):
        if self._interface is None:
            self._interface = BestMatchingInterface()
        return self._interface

    def configure_lm(self):
        lm = dspy.LM(
            model="openai/qwen3:0.6b",
            api_base="http://localhost:11434/v1",
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
        dspy.configure(lm=lm)

    def index(self, max_chunk_size: int):
        """Create the document index."""
        indexer = Indexer(
            "./vllm-0.10.1",
            max_chunk_size=max_chunk_size,
            interface=self.interface,
        )
        indexer.create_index()

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
                metadata["file_path"] = str(
                    Path("data/raw") / metadata["file_path"]
                )
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
            output_directory / "datasets_docs_public.json"
        )

        output_file.write_text(
            json.dumps(
                {"search_results": results, "k": k},
                indent=2,
            ) + "\n"
        )

    def answer(self, query: str, k: int = 5):
        """Generate an answer for a query."""
        self.configure_lm()
        print("Query:", query)
        print("Top-k:", k)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ):
        """Generate answers for a dataset."""
        self.configure_lm()
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
