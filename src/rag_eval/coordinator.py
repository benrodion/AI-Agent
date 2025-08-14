# -----------------------------
# Coordinator 
# -----------------------------
from protocol import LLMClient
from client import AzureClient
from data_models import EvalContainer, ClaimVerdict
from components import EntailmentJudge, ClaimExtractor
from evaluators import AnswerPrecisionEvaluator, RetrievalPrecisionEvaluator
from typing import Dict, Any


class RAGEvaluator:
    def __init__(
            self, 
            llm_extractor: LLMClient, # type hint: in production, pass an AzureClient-instance conforming to LLMClient-protocol
            llm_judge: LLMClient    # type hint: in production, pass an AzureClient-instance conforming to LLMClient-protocol
            ) -> None:
        self.extractor = ClaimExtractor(llm_extractor, model="gpt-4.1")
        self.judge = EntailmentJudge(llm_judge, model="gpt-4.1")
        self.answer_eval = AnswerPrecisionEvaluator(self.extractor, self.judge)
        self.retrieval_eval = RetrievalPrecisionEvaluator(self.extractor, self.judge)

    def evaluate_all(self, ex: EvalContainer) -> Dict[str, Any]:
        return {
            "answer": self.answer_eval.evaluate(ex),
            "retrieval": self.retrieval_eval.evaluate(ex),
        }
