# Changes as of 2025-14-13, 12:00:

## High Level Overview
- like in 2025-10-10, agent chooses if and how many neighbors to retrieve
- like in 2025-10-10, agent is asked to make trade-off btw. added value of more context through higher m and decline in retrieval precision
- NEW: because agent sometimes abandons a question when it cannot find an answer after the first try, the system prompt asks it to fully use its available thinking steps:
  `IMPORTANT: in the first RAG-call, do not retrieve any neighbors, i.e. set m = 0. Instead, retrieve a higher number of chunks (through `top_k`) and identify which are relevant. Then call RAG a second time and get the m-neighbors of the most important chunks.`

## Run Details
- ran with `agent_eval_questions`

### Improvements 
- **Answer Recall**: 
	- Performance nähert sich an Baseline an
	- Gleich viele Fragen mit Wert 0
	- Etwas mehr Fragen mit Wert 1 
- **Answer Precision**:
	- Nur noch halb so viele Fragen mit Wert 0 
	- Steigerung der Fragen mit Wert 1
- **Retrieval Recall**: 
	- Leichte Steigerugng der Fragen mit Wert 1 
- **Retrieval Precision**: 
	- Höhere Spitzenwerte (0.5 statt 0.4)
 - Retrieval Precision baut zwar immer noch ab, aber nicht so drastisch
