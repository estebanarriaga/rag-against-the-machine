PYTHON_VERSION := 3.12
UV := uv
TMP_PROJECT_DIR ?= /tmp/$(USER)-rag
UV_PROJECT_ENVIRONMENT ?= $(TMP_PROJECT_DIR)/venv
UV_CACHE_DIR ?= $(TMP_PROJECT_DIR)/uv-cache
TMPDIR ?= /tmp
OLLAMA_ROOT ?= /goinfre/$(USER)/ollama
OLLAMA_MODEL ?= qwen3:0.6b

export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR TMPDIR

.PHONY: install run debug clean lint lint-strict ollama test-pipeline

install:
	$(UV) sync --dev --python $(PYTHON_VERSION)

ollama:
	OLLAMA_ROOT="$(OLLAMA_ROOT)" OLLAMA_MODEL="$(OLLAMA_MODEL)" bash scripts/ollama.sh

run:
	$(UV) run --python $(PYTHON_VERSION) python -m src

test-pipeline:
	RAG_CMD="$(UV) run --python $(PYTHON_VERSION) python -m src" bash scripts/test_pipeline.sh $(METHODS)

debug:
	$(UV) run --python $(PYTHON_VERSION) python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

lint:
	$(UV) run --python $(PYTHON_VERSION) flake8 .
	$(UV) run --python $(PYTHON_VERSION) mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run --python $(PYTHON_VERSION) flake8 .
	$(UV) run --python $(PYTHON_VERSION) mypy . --strict
