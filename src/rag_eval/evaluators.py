# -----------------------------
# Evaluators
# -----------------------------
from typing import Dict, List, Any
from rag_eval.data_models import EvalContainer, ClaimVerdict
from rag_eval.components import ClaimExtractor, EntailmentJudge

class AnswerPrecisionEvaluator:
    """
    Rates how well the generated answer matches GT claims (recall) and avoids hallucinations (precision).
    """

    def __init__(
            self, 
            extractor: ClaimExtractor, # type hint
            judge: EntailmentJudge  # type hint
            ) -> None:
        self.extractor = extractor
        self.judge = judge

    def evaluate(self, ex: EvalContainer) -> Dict[str, Any]:
        # break down GT and generated answer into individual statements 
        gt_claims = self.extractor.extract(ex.ground_truth_answer)
        gen_claims = self.extractor.extract(ex.generated_answer)

        # Ground truth claim recall: does the generated answer entail *each* GT claim? 
        gt_verdicts: List[ClaimVerdict] = []
        supported_gt = 0
        for c in gt_claims: 
            verdict, rationale = self.judge.judge(c, ex.generated_answer)
            gt_verdicts.append(ClaimVerdict(claim=c, verdict=verdict, rationale=rationale))
            if verdict == "supported":
                supported_gt += 1

        gt_claim_recall = supported_gt / max(1, # safeguard against zero-division
                                             len(gt_claims))

        # Hallucination precision: of the answer's claims, how many are supported by Ground Truth answer? 
        gen_verdicts: List[ClaimVerdict] = []
        supported_gen = 0
        for c in gen_claims:
            verdict, rationale = self.judge.judge(c, ex.ground_truth_answer)
            gen_verdicts.append(ClaimVerdict(claim=c, verdict=verdict, rationale=rationale))
            if verdict == "supported":
                supported_gen += 1
        answer_precision  = supported_gen / max(1,  # safeguard against zero-division
                                                       len(gen_claims))

        return{
            "gt_claims": gt_claims,
            "gen_claims": gen_claims,
            "answer_recall": gt_claim_recall,
            "answer_precision": answer_precision,
            "gt_claim_verdicts": [cv.__dict__ for cv in gt_verdicts],
            "gen_claim_verdicts": [cv.__dict__ for cv in gen_verdicts],
        }

class RetrievalPrecisionEvaluator:
    """
    Rates the precision with which relevant documents are retrieved by holding
    the retrieved docs against the ground truth claims.
    """

    def __init__(
            self, 
            extractor: ClaimExtractor,  # type hint
            judge: EntailmentJudge  # type hint
            ) -> None:
        self.extractor = extractor
        self.judge = judge

    # precision of pooled context: does context "back up" GT claims?
    def evaluate(self, ex: EvalContainer, *, pooled: bool = True) -> Dict[str, Any]: 
        gt_claims = self.extractor.extract(ex.ground_truth_answer)

        pooled_evidence = "\n\n".join(ex.retrieved_texts)
        coverage_supported = 0
        pooled_verdicts: List[ClaimVerdict] = []
        for c in gt_claims:
            verdict, rationale = self.judge.judge(c, pooled_evidence)
            pooled_verdicts.append(ClaimVerdict(claim=c, verdict=verdict, rationale=rationale))
            if verdict == "supported":
                coverage_supported +=1
        retrieval_recall = coverage_supported / max(1, # safeguard against zero-division
                                                      len(gt_claims))


    # precision on doc-level: docs supporting at least one GT claim are marked as relevant
        doc_relevant_flags: List[bool] = [] # stores True/False for each retrieved doc depending on GT support
        claim_doc_support: List[List[int]] = [[] for _ in gt_claims] # one inner list for each GT claim, each holding the indices of
                                                                     # the retrieved docs supporting that claim
        # for each doc_text: compare against each GT claim
        # to check if it supports any GT claim
        # if doc_text X is relevant for GT claim Y, mark as relevant and store index in claim_doc_support
        for i, doc_text in enumerate(ex.retrieved_texts):
            doc_supported_any = False
            for idx , c in enumerate(gt_claims):
                verdict, _ = self.judge.judge(c, doc_text)
                if verdict == "supported":
                    doc_supported_any = True
                    claim_doc_support[idx].append(i)
            doc_relevant_flags.append(doc_supported_any)

        relevant_docs = sum(1 for f in doc_relevant_flags if f)
        retrieval_precision = (relevant_docs / max(1, # safeguard against zero-division
                                              len(ex.retrieved_texts))) if ex.retrieved_texts else 0.0

        return{
            "gt_claims": gt_claims,
            "retrieval_recall": retrieval_recall,
            "retrieval_precision": retrieval_precision,
            "relevant_docs": relevant_docs,
            "n_docs": len(ex.retrieved_texts),
            "pooled_claim_verdicts": [cv.__dict__ for cv in pooled_verdicts],
            "claim_supported_by_docs": claim_doc_support,
        }
