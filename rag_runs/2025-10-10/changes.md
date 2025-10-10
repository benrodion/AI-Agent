# Changes as of 2025-10-10, 15:30:

## High Level Overview
Inserted the following passage in the system prompt: 
`CRITICAL: be very conservative regarding the value of the `execute_agentic_rag`-parameter "m". A higher can improve Retrieval and Answer Recall but will most likely decrease Retrieval Precision. Carefully consider whether adding the neighbours of the retrieved chunks will add value and if so, how many neighbours are really needed.`



## Results 
### Progress (compared to 2025-10-08)
- deutliche Reduktion der Fragen mit Answer Precision = 0*
- deutliche Reduktion der Fragen mit Answer Recall = 0*
- leichte Verbesserung von Retrieval Recall, insbes. mehr Fragen mit Wert 1
- leichte Verbesserung von Retrieval Precision 

### Challenges
- deutlich verringerte Retrieval Precision 
