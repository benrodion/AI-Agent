# Changes as of 2025-10-01, 18:00:

- src/agent/helpers.py: `execute_agentic_rag` now calls `expand_with_precomputed_neighbours`, so that for each retrieved document, its top-m (currently m = 2) neighbours are retrieved) 
- src/agent/rag.indexing.py: equipped with `precompute_and_store_neighbours` --> having indexed, we once call this function which using the Cosine Similarity calculates the top-m neighbours for each chunk and stores their document IDs in this chunk's metadata
- src/agent/rag.helpers.py: created `expand_with_precomputed_neighbours`--> is called by `execute_agentic_rag`, so that upon each RAG-call, not just the retrieved documents are returned, but also their top-m neighbours. Currently, the function is set in such a way that only neighbours from the same document are considered. Can be changed.
- tool call metric results can now be parsed and added to the `df_agentic.csv`


# RAG-Run: 
- ran RAG with all 40 questions and pulled top-m neighbours for each retrieved document
- in `expand_with_precomputed_neighbours`, `same_parent_only`currently set to `True`

## TO DO  
- telemetry needs to be added to normal RAG
