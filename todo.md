
--- Nov. 03

Today we looked at Reciprocal Rank Fusion, Semantic Transformation and Lexical Transformation.
All at a very deep level. We also studied the exam and stuff.

Tomorrow we can deep dive into Chromadb and see how can we implement it in the project

The anwser_dataset() function crashes the computer, we have to look at optimization
ways so we can avoid this. As well as considering batching.

We also saw that RecursiveCharacterTextSplitter has a method that transforms the text in 
different ways depending on the laguage (md, py). Implementing this perhaps can 
improve our implementation even more.

We can then clean the code, and implement a testing layer. Where we can
test different semantic transformations, some examples:
    - **Raw code + BM25**: Establish the baseline with no identifier expansion.
    - **Identifier-expanded BM25**: Measure whether camelCase and snake_case splitting improve lexical recall.
    - **BM25 + contextual descriptions**: Test semantic vocabulary added to the lexical index.
    - **Original code embeddings**: Compare semantic retrieval against the lexical baseline.
    - **Contextual embeddings**: Test whether class, module, and purpose descriptions improve retrieval.
    - **Hybrid BM25 + embeddings**: Fuse lexical and semantic rankings with RRF.
    - **Hybrid + reranking**: Test whether deeper query-document evaluation improves top-k ranking.

>DONE:I dont love our implementation of both indexing/retrieving interfaces. It should be more "abstracted",
>the user should just put the method to use in the cli, bm25 by default. Then we should have a 
>lexical transformation step before indexing that can apply for both interfaces. Another Class or ABC
>if needed. Then the chroma db interface indexing step should be modularized as much as possible so 
>we can test it.

>Done: But perhaps it is going to be cleaner if we start cleaning from the RAG() class and simplify this so we can
>later focuss on the rest.

TODO:
>Done: Separate RecursiveCharacterTextSplitter into languages, in order to get language aware
>text splitting rather than purely character aware. This configures separators that prioritize
>python structures, such as:
>       - Class definitions
>       - Function definitions
>       - Identification and line boundaries
>       - Blank lines
    
    - Optimize anwser_dataset() to avoid crashing the computer.
    - Test different semantic transformations.