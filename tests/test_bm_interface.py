import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from bm_interface import BestMatchingInterface
from index_module import Indexer


class BestMatchingInterfaceTests(unittest.TestCase):
    def test_build_save_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            interface = BestMatchingInterface(index_path=directory)
            documents = [
                Document(page_content="penguin glacier ice", metadata={"file_path": "ice.md"}),
                Document(page_content="tomato basil garden", metadata={"file_path": "garden.md"}),
            ]
            self.assertIs(interface.build_index(documents), interface.retriever)
            results = interface.search("penguin", k=10)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].document, documents[0])
            self.assertGreater(results[0].score, results[1].score)
            saved = [json.loads(line) for line in (Path(directory) / "corpus.jsonl").read_text().splitlines()]
            self.assertEqual(saved[0]["metadata"], documents[0].metadata)
            self.assertEqual(saved[0]["page_content"], documents[0].page_content)
            self.assertTrue((Path(directory) / "params.index.json").exists())
            self.assertEqual(interface.search("the and"), [])
            with self.assertRaises(ValueError):
                interface.search("penguin", k=0)

    def test_invalid_corpus_and_missing_index(self):
        interface = BestMatchingInterface()
        with self.assertRaises(RuntimeError):
            interface.search("penguin")
        for corpus in ([], [Document(page_content="the and")]):
            with self.assertRaises(ValueError):
                interface.build_index(corpus)
        self.assertIsNot(interface.retriever, BestMatchingInterface().retriever)

    def test_indexer_builds_searchable_saved_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "guide.md").write_text("penguin glacier ice")
            indexer = Indexer(str(source), 2000)
            indexer.bm_interface = BestMatchingInterface(index_path=Path(directory) / "index")
            indexer.create_index()
            result = indexer.bm_interface.search("penguin")[0]
            self.assertEqual(result.document.metadata["file_path"], str(source / "guide.md"))


if __name__ == "__main__":
    unittest.main()
