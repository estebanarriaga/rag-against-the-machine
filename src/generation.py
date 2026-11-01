import dspy # type: ignore[import-untyped]
import json

from .models import AnsweredQuestion, MinimalSearchResults

class Generator():
    """
    This class is called in the __main__ program.
    As arguments to work in this step of the RAG we receive:
        - Query: str = It is the user query about the documents
        - K: int = The number of documents to retrieve

    For us to perform the generation need to 
    """

    def __init__(self) -> None:
        """
        Initialize the model configuration
        It is done once, and we can then use it freely across the project
        """
        
        lm = dspy.LM(
            model="openai/qwen3:0.6b",
            api_base="http://localhost:11434/v1",
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
        dspy.configure(lm=lm)


    def generate(self, query: str, sources: MinimalSearchResults) -> AnsweredQuestion:
        """
        For us to generate a concise anwser we may first have to build context.
        This means:
            - We have to create a system prompt. 
            - Constraining the model on what to anwsner and what not. In case a
            question is out of the context
            - We need to receive top-k chunks (the content of this top k)
                - To achieve this, we can call the retriever.search(query, k) and then
                receive this document info: chunk and filename and perform a reading.
                This is what we call augmenting: getting the actual context in those
                chunks
        """
        question = self._augment(query=query, sources=sources) # create the context
        predictor = dspy.Predict("question -> answer")
        return predictor(question=question)


    @staticmethod 
    def _augment(query: str, sources: MinimalSearchResults) -> str: # str or json or smthng
        """
        Build model context before generation
        """
        lm = dspy.settings.lm # we can configure stuff with dspy. Lookup later
        return "not implemented"
    

    @staticmethod
    def _build_anwser():
        """
        In case we need to return a model, instead of just a str
        """
        
        pass