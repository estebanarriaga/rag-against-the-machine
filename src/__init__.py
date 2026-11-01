"""Shared project paths, independent of the current working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_PATH = DATA_PATH / "raw"
CORPUS_PATH = RAW_PATH / "vllm-0.10.1"
INDEX_PATH = DATA_PATH / "processed"
DATASETS_PATH = DATA_PATH / "datasets" / "public"
