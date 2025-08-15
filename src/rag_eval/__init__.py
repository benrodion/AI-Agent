from __future__ import annotations
from rag_eval.protocol import LLMClient
from rag_eval.client import AzureClient
from rag_eval.components import ClaimExtractor, EntailmentJudge
from rag_eval.data_models import EvalContainer, ClaimVerdict
from rag_eval.evaluators import AnswerPrecisionEvaluator, RetrievalPrecisionEvaluator
from rag_eval.coordinator import RAGEvaluator
from rag_eval.helpers import _safe_json

"""
RAG Evaluation Library

Components: 
- LLMClient (Protocol) -->  we are not tied to a specific model: anything with a .complete() method fits.
                            The LLMClient forwards forwards LLM-calls to Azure
- AzureClient (adapter) --> a wrapper around the Azure SDK; gets calls from LLMClient
- ClaimExtractor (component) --> splits any text into atomic claims; calls llm.complete() via LLMClient to
                                 get JSON of claims
- EntailmentJudge (component) --> checks natural language inference (NLI) of (claim, evidence)-pairs; calls llm.complete() via LLM-client
- AnswerPrecisionEvaluator (concrete evaluator) --> calculates how many of the claims in the answer are ground truth claims; 
                                                    uses ClaimExtractor on predicted answers and ground truths, then calls EntailmentJudge
- RetrievalPrecisionEvaluator (concrete evaluator) --> compares retrieved context against ground truth by calling ClaimExtractor and EntailmentJudge
- RAGEValuator (coordinator) --> lets us connect ClaimExtractor and EntailmentJudge into evaluators and call evaluate_all()

Notes: 
- Library uses composition principles --> objects use other objects instead of being them,
                                          i.e. ClaimExtractor uses LLMClient, so we can swap LLM without changing every object
- RAGEvaluator is thin coordinator --> it just connects components, but the magic is contained within the components 
"""

__all__ = [
    "LLMClient",
    "AzureClient",
    "ClaimExtractor",
    "EntailmentJudge",
    "EvalExample",
    "ClaimVerdict",
    "AnswerPrecisionEvaluator",
    "RetrievalPrecisionEvaluator",
    "RAGEvaluator",
    "helpers"
]