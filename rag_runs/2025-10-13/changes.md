# Changes as of 2025-10-13, 19:00:

## High Level Overview
- like in 2025-10-10, agent chooses if and how many neighbors to retrieve
- like in 2025-10-10, agent is asked to make trade-off btw. added value of more context through higher m and decline in retrieval precision
- NEW: because agent sometimes abandons a question when it cannot find an answer after the first try, the system prompt asks it to fully use its available thinking steps:
  `IMPORTANT: when you struggle with a question/do not immediately find the chunks to answer the question, please make use of the amount of thinking steps and RAG-loops available. Do NOT give up before you either found the relevant chunks or reached the limit of your thinking steps.`


## Run Details
- ran with `problem_questions_retrieval_precision`
- no improvements in any metric
- questions with value = 0 increased for all metrics 
