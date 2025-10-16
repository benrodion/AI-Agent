# Changes as of 2025-15-13, 19:00:

## High Level Overview
**Removed from system-prompt**:
`2) RETRIEVE (if needed)
   - IMPORTANT: in the first RAG-call, do not retrieve any neighbors, i.e. set m = 0. Instead, retrieve a higher number of chunks (through `top_k`) and identify which are relevant. Then call RAG a second time and get the m-neighbors of the most important chunks.`

**Added to system-prompt**:
`2) RETRIEVE (if needed)
   - IMPORTANT: ALWAYS retrieve AT LEAST 2 neighbors. I.e. in EACH call of `execute_agentic_rag`, set the parameter `m` to AT LEAST 2:`

## Run Details
- ran with `agent_eval_questions`
- Vergleich: RAG mit mindestens 2 Neighbors pro Aufruf vs. RAG, das im ersten Aufruf ein höheres k und m = 0 hat und erst im 2. Aufruf Neighbors zieht

### Improvements 
- **Answer Recall**: 
	- Die mittleren 50% bewegen sich im selben Intervall, aber mit deutlich verbessertem Median
 
- **Answer Precision**:
	- Mit m = 4 verbessert gegenüber m = 2

### Setbacks
- **Retrieval Recall**: 
	- Deutlich größere IQR
	- Niedrigerer Median

- **Retrieval Precision**: 
	- Erwartungsgemäß niedriger, wenn mehr Neighbors gezogen werden

### Unchanged
In beiden Szenarien scheinen 2 RAG-Aufrufe pro Frage optimal zu sein, bei 3 RAG.Aufrufen stellen sich über alle Metriken hinweg Verschlechterungen ein
