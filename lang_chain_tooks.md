LangChain provides several useful text transformation, document processing, and query transformation tools for RAG. Some manipulate the textual representation itself, while others transform the structure, enrich metadata, or change how retrieval operates.

For your code-retrieval experiments comparing BM25, embeddings, contextual enrichment, and HyDE, I would distinguish four layers.

## 1. Document segmentation and structural transformation

These operate primarily at indexing time, before BM25 indexing or embedding generation.

| LangChain tool                   | Technique                              | Use in your RAG                                          |
| -------------------------------- | -------------------------------------- | -------------------------------------------------------- |
| `RecursiveCharacterTextSplitter` | Hierarchical, separator-based chunking | Baseline code chunking                                   |
| `PythonCodeTextSplitter`         | Syntax-aware heuristic splitting       | Prefer function/class boundaries in Python               |
| `JSFrameworkTextSplitter`        | Framework-aware segmentation           | Preserve JSX, Vue and Svelte component boundaries        |
| `MarkdownHeaderTextSplitter`     | Header-based structural segmentation   | Preserve documentation hierarchy                         |
| `HTMLHeaderTextSplitter`         | DOM/header-aware segmentation          | Extract meaningful website sections                      |
| `RecursiveJsonSplitter`          | Hierarchy-preserving JSON partitioning | Index configuration files and structured data            |
| `TokenTextSplitter`              | Token-budget segmentation              | Control chunk lengths according to token counts          |
| `SpacyTextSplitter`              | Sentence-boundary segmentation         | Avoid cutting natural-language explanations mid-sentence |

These are documented in [LangChain Text Splitters](https://reference.langchain.com/python/langchain-text-splitters?utm_source=chatgpt.com).

A crucial distinction: syntax-aware chunking is not AST-based semantic chunking. `PythonCodeTextSplitter`, for example, still uses separator heuristics rather than a complete Python syntax tree.

## 2. Document enrichment and transformation

These modify content or its associated representation.

| Tool                         | Technique                                | Potential application                                  |
| ---------------------------- | ---------------------------------------- | ------------------------------------------------------ |
| `OpenAIMetadataTagger`       | LLM-based structured metadata extraction | Extract module purpose, responsibilities or categories |
| `DoctranQATransformer`       | Synthetic question-answer generation     | Generate questions that source chunks could answer     |
| `DoctranPropertyExtractor`   | Structured property extraction           | Extract document attributes                            |
| `EmbeddingsRedundantFilter`  | Embedding-based deduplication            | Remove semantically redundant chunks                   |
| `EmbeddingsClusteringFilter` | Vector clustering                        | Select representative documents                        |
| `MarkdownifyTransformer`     | HTML-to-Markdown normalization           | Clean HTML before indexing                             |

These come primarily from `langchain_community.document_transformers`.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+1



For your experiments, synthetic question generation is particularly interesting. Rather than indexing only:

```
def calculate_total(items):    return sum(item.price for item in items)
```

You could additionally index a generated question:

> How is the total price of a collection of items calculated?

This is document-side query augmentation: generating likely queries from documents to improve retrieval alignment. You would need to associate the generated question with its original code chunk.

## 3. Query transformation

These operate at retrieval time, transforming the user's query rather than the indexed corpus.

| Tool                           | Technique                 | Use                                                           |
| ------------------------------ | ------------------------- | ------------------------------------------------------------- |
| `MultiQueryRetriever`          | LLM-based query expansion | Generate several query formulations and retrieve their union  |
| `RePhraseQueryRetriever`       | Query rewriting           | Convert ambiguous questions into more explicit search queries |
| `HypotheticalDocumentEmbedder` | HyDE                      | Generate hypothetical answers to improve dense retrieval      |

For example, the query "Where are payments validated?" could be expanded into "payment verification function", "transaction validation logic", and "checkout authorization checks".

These implementations are available in `langchain-classic`, although its retriever APIs are legacy; for new projects, equivalent pipelines can be composed using LangChain's current runnable interfaces.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+3



## 4. Post-retrieval transformations

These process documents after initial retrieval, before generation.

| Tool                      | Technique                                | Use                                                           |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------- |
| `LLMChainExtractor`       | Query-conditioned information extraction | Retain only relevant passages from retrieved chunks           |
| `EmbeddingsFilter`        | Query-document semantic filtering        | Discard insufficiently relevant documents                     |
| `LLMChainFilter`          | LLM-based relevance classification       | Remove irrelevant retrieved results                           |
| `LongContextReorder`      | Context-position optimization            | Reorder documents to mitigate the lost-in-the-middle effect   |
| `ParentDocumentRetriever` | Hierarchical retrieval                   | Retrieve using small chunks but return larger parent contexts |

These are useful for improving the context supplied to the generator without necessarily changing the search index.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+2



## 5. What I would test in your RAG project

Based on your existing experimental strategy, I would prioritize the following additions:

| Experiment                                       | Layer          | Hypothesis                                                         |
| ------------------------------------------------ | -------------- | ------------------------------------------------------------------ |
| Python-aware chunking                            | Indexing       | Better code boundaries improve Recall\@k                           |
| Identifier splitting (`camelCase`, `snake_case`) | Indexing       | Better lexical term matching improves BM25                         |
| Synthetic question generation                    | Indexing       | Question-like representations bridge the query-code vocabulary gap |
| Metadata-enriched chunks                         | Indexing       | Module, class and function context improves retrieval              |
| Multi-query expansion                            | Query          | Alternative formulations recover additional relevant chunks        |
| HyDE                                             | Query          | Hypothetical code or descriptions improve dense recall             |
| Parent-document retrieval                        | Post-retrieval | Larger context improves answer completeness                        |
| LLM contextual extraction                        | Post-retrieval | Removing irrelevant text improves answer faithfulness              |

One important qualification: LangChain does not provide a dedicated, general-purpose code identifier expansion transformer in these standard components. For that experiment, I would implement a deterministic preprocessing function, rather than use an LLM.

Also, keep retrieval and generation metrics separate: Recall\@k and MRR evaluate retrieval quality; answer correctness and faithfulness evaluate downstream generation.

My suggested progression is to test deterministic transformations first, then query expansion and HyDE, and finally post-retrieval compression. This will make it easier to attribute any improvement to the correct layer.

The two API references worth keeping are [Text Splitters](https://reference.langchain.com/python/langchain-text-splitters?utm_source=chatgpt.com) and [Document Transformers](https://reference.langchain.com/python/langchain-community/document_transformers?utm_source=chatgpt.com).