# Changes as of 2025-09-08, 17:20:

- src/agent/prompts.py: changed the system prompt to force the agent to always use RAG. Fixes the issue that previously, the agent only answered 29/40 questions using RAG  
- src/rag_eval/eval_loop.py: added a safeguard that lets us know when the agent did not answer a question using RAG and gives us the index of this question
- created plot to compare performance of agentic vs. normal RAG