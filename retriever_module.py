

class Retriever():
    """
    Ther retriever receives a question, then searches the corpus for the top-k
    results. It returns a list of documents. Thats it. It doesn not interact with
    the model at all. It is a simple abstraction over it.

    Depending on simplicity this could be built directly as a method of the indexer
    and make the Indexer class the RAG main class.
    """
    
    pass
