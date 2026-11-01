"""Offline generation tests; no local LLM server required."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.generation import Generator, NO_ANSWER
from src.models import MinimalSearchResults, MinimalSource


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.file = self.root / "sample.py"
        self.file.write_text("hello from vllm\nother text", encoding="utf-8")

        self.generator = object.__new__(Generator)
        self.generator.corpus_root = self.root
        self.generator.predictor = Mock(
            return_value=SimpleNamespace(answer="A grounded answer")
        )

    def result(self, sources):
        return MinimalSearchResults(
            question_id="question-1",
            question="What does this code say?",
            retrieved_sources=sources,
        )

    def test_generate_reads_offsets_and_preserves_metadata(self):
        source = MinimalSource(
            file_path=str(self.file),
            first_character_index=6,
            last_character_index=15,
        )
        response = self.generator.generate(
            "What does this code say?", self.result([source])
        )
        self.assertEqual(response.question_id, "question-1")
        self.assertEqual(response.sources, [source])
        self.assertEqual(response.answer, "A grounded answer")
        context = self.generator.predictor.call_args.kwargs["context"]
        self.assertIn("from vllm", context)
        self.assertIn(str(self.file), context)

    def test_no_sources_does_not_call_model(self):
        response = self.generator.generate("What does this code say?", self.result([]))
        self.assertEqual(response.answer, NO_ANSWER)
        self.generator.predictor.assert_not_called()

    def test_wrong_question_rejected(self):
        with self.assertRaises(ValueError):
            self.generator.generate("Different question", self.result([]))

    def test_source_outside_corpus_rejected(self):
        source = MinimalSource(
            file_path=str(self.root.parent / "other.py"),
            first_character_index=0,
            last_character_index=1,
        )
        with self.assertRaises(ValueError):
            self.generator.generate("What does this code say?", self.result([source]))

    def test_invalid_offsets_rejected(self):
        source = MinimalSource(
            file_path=str(self.file),
            first_character_index=0,
            last_character_index=10000,
        )
        with self.assertRaises(ValueError):
            self.generator.generate("What does this code say?", self.result([source]))


if __name__ == "__main__":
    unittest.main()
