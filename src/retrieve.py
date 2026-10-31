from bm_interface import BestMatchingInterface, SearchResult

class Retriever():
    """
    Ther retriever receives a question, then searches the corpus for the top-k
    results. It returns a list of documents. Thats it. It doest interact with
    the model at all. It is a simple abstraction over it.

    Depending on simplicity this could be built directly as a method of the indexer
    and make the Indexer class the RAG main class.
    """
    def __init__(self, interface: BestMatchingInterface) -> None:
        self.bm_interface = interface
    
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """
        Search the BM25 index based on an query.
        The steps are:
            - Search the index (BestMatchingInterface.search()). This function
            already tokenizes our query and does all the hard work.
        
        > Note. We may even not need the Retriever() because of the simplicity of the task
        > we can just add it to the RAG class
        """
        return self.bm_interface.search(query=query, k=k)
