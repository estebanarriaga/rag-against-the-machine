
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
import logging
from typing import Any, Sequence
from pydantic import BaseModel, Field

from src.context import Context

from .bm_interface import BestMatchingInterface
from .chroma_interface import ChromaInterface
from .base import Retriever
from .models import FileType, chunk_id
from . import PROJECT_ROOT


logger = logging.getLogger("Indexer")


class IndexerParams(BaseModel):
    """
    Validates that the parameters passed to the Indexer class are correct.
    This is a Pydantic model that will be used to validate the parameters passed to the Indexer class.
    """
    
    corpus_path: str | Path
    max_chunk_size: int = Field(..., ge=1, le=2000, description="Maximum chunk size in characters (1-2000).")


class Indexer():
    """ This class is in charge of:
        1. Receives the documents path and:
            1.1 Validates valid path and permissions (handles open() errors or simplibs-path valid too)
            1.2 Retrieves only .py files and .md files
            By the end it has a list of all the valid documents that we need to create the index next
        2. Takes the document array and splits it into chuncks of max_chunk_size (<= 2000 this control
            can be done in the parser via Pydantic model validation, since max_chunk_size is passed there)
        3. Take this chuncks array and pass them to bm25s. The package will be in charge of:
            3.1 Tokenize the chunks using bm25s.tokenizer
                3.1.2 We can add a flag here to enable the stemmer, we can check its performance in the future
                    once the basic things are handled, maybe better to use in .md files only
            3.2 Then we create the numpy vector space with bm25s.index()

    > Note: bm25s should be its own class/interface, since it sits in between the indexing and the retrieval:
    > we use it to create the index, and then to search and anwser. In this class only index corpus generation
    > will be used.

    Docs in RecursiveCharacterTextSplitter(): https://docs.langchain.com/oss/python/langchain/knowledge-base
    lang_chain Document: contains page_content and metadata. Metadata is not used in index
    """
    FILES_TYPES: dict[FileType, set[str]] = {
        FileType.CODE: {'.py'},
        FileType.DOCS: {'.md', '.txt'},
    }

    files_splitter = dict[str, RecursiveCharacterTextSplitter]


    def __init__(self,
        corpus_path: str | Path,
        max_chunk_size: int,
        config: Context,
        interface: Retriever
    ) -> None:
        self.corpus_path = corpus_path
        self.chunk_size = max_chunk_size
        self.config = config
        self.interface = interface


    def create_index(self):
        """Point 1, 2 and 3"""

        file_content = self._get_file_content()
        corpus = self._split_documents(file_content)

        logger.info(f"Creating index with chunk size: {self.chunk_size} characters.")
        logger.debug(f"Corpus created with {len(corpus)} chunks.")

        for document in corpus:
            start = document.metadata.pop("start_index") # This should be the MinimalSource model
            document.metadata["first_character_index"] = start
            document.metadata["last_character_index"] = start + len(document.page_content)
            # Same ID in every index, so their rankings can be fused later (RRF).
            document.metadata["chunk_id"] = chunk_id(
                document.metadata["file_path"],
                document.metadata["first_character_index"],
                document.metadata["last_character_index"],
            )
        # Every index is built from the exact same corpus, so they stay in sync.
        logger.info(f"Building {self.interface.name} index...")
        self.interface.build_index(corpus)


    def _get_file_content(self) -> list[Document]:
        """Load every code and docs file under corpus_path as one Document each.

        file_path is stored relative to PROJECT_ROOT when possible, so sources
        stay valid if the project moves. Unreadable files are skipped with a warning.
        """
        code_suffixes = self.FILES_TYPES[FileType.CODE]
        allowed_suffixes = code_suffixes | self.FILES_TYPES[FileType.DOCS]

        documents: list[Document] = []
        for file_path in sorted(Path(self.corpus_path).resolve().rglob("*")):
            if not file_path.is_file() or file_path.suffix not in allowed_suffixes:
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                logger.warning(f"Skipping {file_path}: {error}")
                continue

            source_path = (
                file_path.relative_to(PROJECT_ROOT)
                if file_path.is_relative_to(PROJECT_ROOT)
                else file_path
            )
            language = Language.PYTHON if file_path.suffix in code_suffixes else Language.MARKDOWN
            documents.append(Document(
                page_content=content,
                metadata={"file_path": source_path.as_posix(), "file_type": language.value},
            ))

        logger.debug(f"Loaded {len(documents)} files from {self.corpus_path}.")
        return documents


    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Split each document with the splitter for its language (metadata["file_type"]).

        from_language() swaps the default separators for language-aware ones:
        Python splits on class/def boundaries first, Markdown on headings.
        One splitter is built per language and reused for every file.
        """
        splitters: dict[str, RecursiveCharacterTextSplitter] = {}
        chunks: list[Document] = []
        for document in documents:
            language = document.metadata["file_type"]
            if language not in splitters:
                splitters[language] = RecursiveCharacterTextSplitter.from_language(
                    Language(language),
                    chunk_size=self.chunk_size,
                    chunk_overlap=min(200, self.chunk_size // 5),
                    add_start_index=True,
                )
            chunks.extend(splitters[language].split_documents([document]))
        return chunks

    def cache(self) -> Any:
        """
        Cache the index results when the query is the same.
        """
        
        return NotImplementedError
