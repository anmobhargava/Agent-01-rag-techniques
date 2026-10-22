# RAG technique agent

"RAG" is not one thing. Plain vector search misses names and numbers, struggles with broad questions, and returns chunks that never say what they are about. Each problem has its own fix, and few people explain when to use which.

So I built 16 retrieval techniques in one repo, all over the same small set of documents, and put an agent on top that picks between them. It decides how much searching a question needs, picks a technique, checks what came back, switches if the results are weak, answers with citations, and checks its own answer. The documents are invented (a snack brand, its campaigns and agencies), so there is no real company data.


## The 16 techniques

Naive, Hybrid, Hybrid + rerank, HyDE, Multi-query, Decomposition, Parent document, Contextual retrieval, RAPTOR, GraphRAG, Corrective RAG, Sentence window, Late chunking, ColBERT, Metadata filter, Iterative.

Each one is a short function in `src/rag_agent/techniques.py`. The Compare tab in the app runs your question through all of them side by side.

## How the agent decides

1. Rewrite a follow-up like "and its fee?" into a question that stands alone
2. Reuse an answer if the same question was already answered
3. Decide how much searching it needs: none, one search, or several
4. Pick a technique and say why
5. Grade the chunks it got back, and switch technique if they are weak
6. Answer only from the chunks, with citations
7. Check every claim against the chunks, and rewrite once if one is unsupported
8. Check that every citation points at a chunk it was actually given

## What is under it

- **Haystack** for the pipelines
- **Qdrant** for vectors, filters and ColBERT-style search
- **Neo4j with Graph Data Science** (optional) for the graph, with an in-memory version by default
- **A real embedding model**, with a simple fallback so it runs anywhere
- **A semantic cache** that will not reuse an answer when the names or numbers differ
- **Fine-tuning scripts** that train the embedding model and report before and after
- **Gradio** for the chat app, plus Docker, a Hugging Face Space and CI


- `src/rag_agent/` the agent, techniques, embeddings, cache and graph
- `eval/` generate and run an evaluation set
- `finetune/` fine-tune the embedding model
- `data/docs/` the 14 invented documents
