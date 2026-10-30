
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from path import Path
from bm_interface import BestMatchingInterface

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
    def __init__(self, directory_path: str, max_chunk_size) -> None:
        self.directory_path = directory_path
        self.chunk_size = max_chunk_size
        self.bm_interface = BestMatchingInterface()

    def create_index(self):
        """Point 1, 2 and 3"""
        if not self:
            print("initialize the Indexer")
        
        file_content = self._get_file_content()
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        # in this step we can use the stemmer
        corpus = splitter.split_documents(file_content)
        bm = BestMatchingInterface()
        bm.build_index(corpus)

        for result in bm.search("How does caching work?", k=5):
            print(result.score)
            print(result.document.page_content)
            print(result.document.metadata)


    def _get_file_content(self) -> list[Document]:
        # validate it is a valid directory
        
        file_content: list[Document] = []
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if file.endswith((".py", ".md")):
                    file_path = root + os.sep + file
                    file_content.append(
                        Document(
                            page_content=Path(file_path).read_text(),
                            metadata={
                                "file_path": file_path,
                            }))

        return file_content

    # def _split_documents(self):
    #     for document in self.documents
    #     text_splitter = RecursiveCharacterTextSplitter()
