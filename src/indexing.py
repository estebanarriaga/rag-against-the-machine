
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from pathlib import Path

from .bm_index import BestMatchingInterface
from . import PROJECT_ROOT

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
    """
    def __init__(self, directory_path: str | Path, max_chunk_size: int, interface: BestMatchingInterface) -> None:
        self.directory_path = directory_path
        self.chunk_size = max_chunk_size
        self.bm_interface = interface

    def create_index(self):
        """Point 1, 2 and 3"""
        if not self:
            print("initialize the Indexer")
        
        file_content = self._get_file_content()
        if not 1 <= self.chunk_size <= 2000:
            raise ValueError("max_chunk_size must be between 1 and 2000 characters.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=min(200, self.chunk_size // 5),
            add_start_index=True,
        )
        # in this step we can use the stemmer
        corpus = splitter.split_documents(file_content)
        for document in corpus:
            start = document.metadata.pop("start_index")
            document.metadata["first_character_index"] = start
            document.metadata["last_character_index"] = start + len(document.page_content)
        bm = self.bm_interface
        bm.build_index(corpus)

    def _get_file_content(self) -> list[Document]:
        # validate it is a valid directory
        
        file_content: list[Document] = []
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if file.endswith((".py", ".md")):
                    file_path = (Path(root) / file).resolve()
                    try:
                        source_path = file_path.relative_to(PROJECT_ROOT)
                    except ValueError:
                        source_path = file_path
                    file_content.append(
                        Document(
                            page_content=Path(file_path).read_text(),
                            metadata={
                                "file_path": source_path.as_posix(),
                            }))

        return file_content

    # def _split_documents(self):
    #     for document in self.documents
    #     text_splitter = RecursiveCharacterTextSplitter()
