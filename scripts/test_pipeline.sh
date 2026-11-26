#!/usr/bin/env bash
# Run the retrieval pipeline end to end for every method, in order:
#   bm25 -> dense -> hybrid
# Each method: index -> search_dataset (code + docs) -> evaluate (recall@k).
#
# Usage: scripts/test_pipeline.sh [method ...]     (default: bm25 dense hybrid)
# Env:   RAG_CMD, MAX_CHUNK_SIZE, K, OUTPUT_ROOT
set -euo pipefail

: "${RAG_CMD:=uv run --python 3.12 python -m src}"
: "${MAX_CHUNK_SIZE:=2000}"
: "${K:=10}"   # evaluate reports recall@1/3/5/10, so retrieve at least 10
: "${OUTPUT_ROOT:=data/output/search_results}"

UNANSWERED_DIR=data/datasets/public/UnansweredQuestions
ANSWERED_DIR=data/datasets/public/AnsweredQuestions
DATASETS=(dataset_code_public.json dataset_docs_public.json)

if [ "$#" -gt 0 ]; then
    METHODS=("$@")
else
    METHODS=(bm25 dense hybrid)
fi

run() {
    echo "+ $RAG_CMD $*"
    # shellcheck disable=SC2086  # RAG_CMD is meant to be split into words
    $RAG_CMD "$@"
}

for method in "${METHODS[@]}"; do
    echo
    echo "=================== $method ==================="
    output_dir="$OUTPUT_ROOT/$method"

    # 1. Index. Hybrid rebuilds BM25 and Chroma from the same chunks.
    run index "$MAX_CHUNK_SIZE" --method "$method"

    for dataset in "${DATASETS[@]}"; do
        # 2. Search every question of the unanswered dataset.
        run search_dataset "$UNANSWERED_DIR/$dataset" "$output_dir" --k "$K" --method "$method"

        # RAG commands log errors instead of failing, so check the output exists.
        if [ ! -s "$output_dir/$dataset" ]; then
            echo "FAIL [$method] search_dataset wrote no $output_dir/$dataset" >&2
            exit 1
        fi

        # 3. Evaluate the results against the answered dataset.
        echo "--- [$method] $dataset"
        run evaluate "$output_dir/$dataset" "$ANSWERED_DIR/$dataset"
    done
done

echo
echo "Pipeline OK for: ${METHODS[*]}"
