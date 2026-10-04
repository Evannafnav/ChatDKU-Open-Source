# Candidate change: consistent source file metadata

## Problem and path through the code

`chatdku/chatdku/core/tools/llama_index.py` combines vector and keyword results in
`DocRetrieverOuter.DocumentRetriever`, converts them with `nodes_to_dicts`, and
passes the result to the agent's later reasoning stages. The two retrievers use
different keys for the source filename:

| Producer | File key | Other source fields |
| --- | --- | --- |
| `retriever/vector_retriever.py` | `file_name` | `url`, `page_number` |
| `retriever/keyword_retriever.py` | `filename` | `url`, `page_number` |

Consequently, a consumer looking for `file_name` can identify a vector hit's
file but miss the same information on a keyword hit. The change normalizes the
metadata at the common conversion boundary. It adds `file_name` when only
`filename` exists; the original `filename`, URL, page number, and other fields
are retained. It returns a copy so the retrieved node is not mutated.

## How to verify

From the repository root, with Python 3.11 or newer:

```bash
cd chatdku
PYTHONPATH=. python -m unittest discover -s tests -p test_source_metadata.py -v
```

The four service-free tests cover keyword and vector metadata, conflicting
aliases, and missing names. They do not test retrieval ranking, generation,
citations in final answers, or a deployed ChatDKU instance. Those require the
configured Redis, ChromaDB, embedding service, and LLM. A useful next evaluation
is to inspect mixed vector/keyword hits in a running agent trace and measure
whether the answer actually cites the correct document and page. A normalized
field alone cannot guarantee a correct citation.
