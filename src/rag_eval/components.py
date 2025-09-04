# -----------------------------
# Components
# -----------------------------
# Both components are just system prompts which forward their specific inputs to the AzureClient
# Based on the prompt, the AzureClient outputs the desired JSON for parsing
# ClaimExtractor parses JSON to list, EntailmentJudge to dictionary 
from rag_eval.helpers import _safe_json
from rag_eval.protocol import LLMClient
from typing import List, Tuple, Optional

class ClaimExtractor:
    """
    Prompt that uses LLMClient to turn text (predicted answers, ground truth)
    into atomic, checkable claims.
    """

    SYSTEM_PROMPT = """
    You receive text and split it into atomic, verifiable claims. Split conjunctions, 
    separate quantities, dates and named entities into distinct claims when needed. Please make use of the question,
    which is also passed to you, so you extract claims adequately. E.g. if a question is "When did WW2 start?" and the ground truth
    is "1939", the claim in the ground truth is not "1939 is a year" but "WW2 started in 1939".
    Return a JSON: {\"claims\": [\"...\", \"...\"]}. """

    def __init__(
            self, llm: LLMClient,
            *, 
            model: str = "gpt-4.1",
            temperature: float = 0.0
            ):
        self.llm = llm
        self.model = model
        self.temperature = temperature

    def extract(self, text: str, query: Optional[str] = None) -> List[str]:
        "Split input into atomic claims."

        if query: 
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": (text  + query
                                            ),
                },
            ]

        else: 
             messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": text
            },
        ]
        raw = self.llm.complete(messages, model=self.model, temperature=self.temperature) # where the magic happens: claims get split
        data = _safe_json(raw) or {}

        if data and isinstance(data.get("claims"), list):
            claims = [str(c).strip() for c in data["claims"] 
                      if str(c).strip()]    # check if string exists after stripping, filter out "" and " "-strings
            return claims


class EntailmentJudge:
    """
    LLM-based Natural Language Inference-checker.
    Checks if evidence provides SUPPORT; CONTRADICT or NOT_ENOUGH_INFO for a claim.
    """
    SYSTEM_PROMPT = """
    You are a strict fact-checking judge. Given a claim and evidence text, output
    a JSON with the fields: {"verdict": ("supported"/"contradicted"/"not_enough_info"), "rationale": "..."}.
    Be conservative: if evidence is missing or ambiguous, revert to not_enough_info. 
    Exact numeric mismatches are contradicted.
    """

    def __init__(
            self, 
            llm: LLMClient, 
            *, 
            model: str = "gpt-4.1", 
            temperature: float = 0.0
            ):
        self.llm = llm
        self.model = model
        self.temperature = temperature

    def judge(self, claim: str, evidence: str) -> Tuple[str, str]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": ("Claim: " +
                                         claim
                                         + "\n\n Evidence: \n"
                                         + evidence
                                         + "\n\nReturn JSON."
                                         ),
                                    },
            ]

        raw = self.llm.complete(messages, model=self.model, temperature=self.temperature)
        data = _safe_json(raw) or {}
        verdict = str(data.get("verdict", "")).strip().lower()
        rationale = str(data.get("rationale", "")).strip().lower()
        if verdict not in {"supported", "contradicted", "not_enough_info"}:
            verdict="not_enough_info"
        return verdict, rationale
        