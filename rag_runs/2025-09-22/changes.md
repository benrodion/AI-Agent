# Changes as of 2025-09-22, 17:00:

- src/agent/prompts.py: changed the system prompt to equip the agent with an complete operating protocol --> RAG-retries no longer hardcoded, but built into the system prompt
- src/agent/telemetry.py: a function for metering token usage per answer per question
- src/agent/llm.py: a wrapper for the OpenAI-SDK that equips the agent with the Tokenmeter
- src/rag_eval/eval_loop.py: small tweak to ensure that `retrieved_context`is passed to the `EvalContainer`as a list
- created plot to compare performance of agentic vs. normal RAG
- 

## TO DO  
- telemetry does not seem to work --> appears to add up token usage across the answers for all questions
