# -----------------------------
# Data containers
# -----------------------------
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EvalContainer:
    "A  single unit of evaluation keeping all the inputs we need to evaluate retrieval and answer precision."
    query: str
    ground_truth_answer: str
    generated_answer: str
    retrieved_texts: List[str] # plain text from retrieved context docs

@dataclass
class ClaimVerdict:
    """
    A single container for the results of judging one claim against available evidence.
    """
    claim: str
    verdict: str  # supported | contradicted | not_enough_info
    rationale: str = ""
    doc_indices: Optional[List[int]] = None  # for retrieval doc-level support. More lean than storing full-context