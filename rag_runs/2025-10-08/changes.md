# Changes as of 2025-10-08, 19:00:

## High Level Overview
- unlike in run 2025-10-06, the neighbours are now retrieved right after the initial and then passed to the LLM before answer generation. In the previous version, this was erroneously not done. Instead, neighbours were retrieved but never fed to the LLM for answer generation
- the amount of neighbours retrieved and if they need to come from the same parent document is now at the agent's discretion


## Run Details
- ran with `problem_questions_retrieval_precision`
- results for retrieval precision declined, as is to be expected when in addition to chunks selected by the retriever we also retrieve their neighbours
