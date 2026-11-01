
import json
from pathlib import Path

KS = (1, 3, 5, 10)


def source_matches(candidate, reference):
    if candidate["file_path"] != reference["file_path"]:
        return False

    start = max(
        candidate["first_character_index"],
        reference["first_character_index"]
    )
    end = min(
        candidate["last_character_index"],
        reference["last_character_index"]
    )

    overlap = max(0, end - start)

    reference_length = (
        reference["last_character_index"]
        - reference["first_character_index"]
    )

    if reference_length <= 0:
        return False

    return overlap / reference_length >= 0.05


def recall_at_k(retrieved, references, k):
    if not references:
        raise ValueError("No reference passages")

    hits = 0

    for reference in references:
        for candidate in retrieved[:k]:
            if source_matches(candidate, reference):
                hits += 1
                break

    return hits / len(references)


def evaluate(results_path, dataset_path):
    results = json.loads(
        Path(results_path).read_text()
    )
    dataset = json.loads(
        Path(dataset_path).read_text()
    )

    # Ground-truth sources indexed by question_id
    gold = {}

    for question in dataset["rag_questions"]:
        if question.get("sources"):
            gold[question["question_id"]] = question["sources"]

    # Student retrieval results indexed by question_id
    predictions = {}

    for result in results["search_results"]:
        question_id = result["question_id"]

        if question_id in predictions:
            raise ValueError("Duplicate question_id")

        predictions[question_id] = result["retrieved_sources"]

    if not gold:
        raise ValueError("No labeled questions")

    totals = {k: 0.0 for k in KS}
    failures = []

    for question_id, references in gold.items():
        retrieved = predictions.get(question_id, [])

        for k in KS:
            score = recall_at_k(retrieved, references, k)
            totals[k] += score

            if k == 5 and score < 1.0:
                failures.append((question_id, score))

    count = len(gold)

    print(f"Questions evaluated: {count}")
    print(f"Missing predictions: {len(set(gold) - set(predictions))}")

    for k in KS:
        mean_recall = totals[k] / count
        print(f"Recall@{k}: {mean_recall:.2%}")

    print("\nFirst five incomplete Recall@5 cases:")
    for question_id, score in failures[:5]:
        print(question_id, f"{score:.2%}")

if __name__ == "__main__":
    import sys

    evaluate(sys.argv[1], sys.argv[2])