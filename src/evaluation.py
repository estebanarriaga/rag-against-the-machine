from pathlib import Path
import logging
import json
from .models import MinimalSource

KS = (1, 3, 5, 10)

logger = logging.getLogger("Evaluator")


class Evaluator():
    """
    Run recall@k in search results against a dataset path.
    Params:
        - `student_search_results_path`: The program generated
        results to evaluate
        - `dataset_path`: The supervised dataset of relevant 
        items (AnwseredQuestions)
    
    We run recall@k for k = {1, 3, 5, 10}
    For this we have to evaluate the top-k sources in 
    each MinimalSearchResult question_id in each one.
    Once we have the MinimalSource (file_path,  first_character_index,
    last_character_index)
        - We now check if the source file_path matches
        - If they match, do they math first char and last_char? Leaving some
        range.
        - If they dont, then it doesnt count.
        - If they do. We add it to the score
    """

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> dict[int, float]:
        results = json.loads(Path(student_search_results_path).read_text())
        dataset = json.loads(Path(dataset_path).read_text())

        results = self._parse_data(results, "search_results", "retrieved_sources")
        dataset = self._parse_data(dataset, "rag_questions", "sources", labeled_only=True)
        if not dataset:
            raise ValueError("No labeled questions")
        
        # Populate scores for each with 0.0 at first
        totals =  {k: 0.0 for k in KS}

        for question_id, sources in dataset.items():
            retrieved = results.get(question_id, [])
            for k in KS:
                score = self._recall_at_k(retrieved, sources, k)
                totals[k] += score
        
        count = len(dataset)
        missing = len(set(dataset) - set(results))
        if missing:
            logger.warning(f"Missing predictions: {missing}")
        else:
            logger.info("No missing predictions, all questions answered.")
        
        mean_recall = {k: totals[k] / count for k in KS}
        for k in KS:
            print(f"Recall@{k}: {mean_recall[k]:.2%}")
        
        return mean_recall
    

    def _recall_at_k(
        self,
        retrieved: list[MinimalSource],
        references: list[MinimalSource],
        k: int
    ):
        if not references:
            raise ValueError("No reference passages")

        hits = 0

        for reference in references:
            for candidate in retrieved[:k]:
                if self._source_matches(candidate, reference):
                    hits += 1
                    break

        return hits / len(references)
    
    def _source_matches(
        self,
        candidate: MinimalSource,
        reference: MinimalSource
    ):
        if candidate.file_path != reference.file_path:
            return False

        start = max(
            candidate.first_character_index,
            reference.first_character_index
        )
        end = min(
            candidate.last_character_index,
            reference.last_character_index
        )

        overlap = max(0, end - start)

        reference_length = (
            reference.last_character_index
            - reference.first_character_index
        )

        if reference_length <= 0:
            return False

        return overlap / reference_length >= 0.05

    
    def _parse_data(
        self,
        data: dict,
        items_key: str,
        sources_key: str,
        *,
        labeled_only: bool = False,
    ) -> dict[str, list[MinimalSource]]:
        """Index nested source lists from decoded JSON by question ID."""
        parsed = {}
        for item in data[items_key]:
            sources = item.get(sources_key, [])
            if labeled_only and not sources:
                continue
            question_id = item["question_id"]
            if question_id in parsed:
                raise ValueError(f"Duplicate question_id: {question_id}")
            parsed[question_id] = [MinimalSource(**source) for source in sources]
        return parsed
