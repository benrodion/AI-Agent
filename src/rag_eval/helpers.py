import re
import json
from typing import Optional

def _safe_json(s: str) -> Optional[dict]:
    "For more robust JSON-parsing, in case LLM returns corrupted JSON."
    s = s.strip()
    # Look for JSON object anywhere in string and parse
    match = re.search(r"\{[\s\S]*\}$", s)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        
     # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        return None
    
from transformers import AutoTokenizer


def query_gt_extractor(
        *, 
        path: str, 
        encoder: Optional[str] = None, 
        tokenize: bool = False
        ) -> Optional[tuple[list[str], list[str]]]:
    """Accesses the evaluation-JSON with queries and ground truth.
    
            Parameters:
                path (str): path JSON-file from which to extract queries and ground truth
                encoder (str): encoder for tokenization 

            Returns:
                queries (list[str]): a list containing the queries
                gt (list[str]): a list containing the ground truth
    """
    queries, gt = [], []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        q = data["questions"]
        gts = data["ground_truths"]

        # prepare tokenizer only if needed
        tokenizer = AutoTokenizer.from_pretrained(encoder) if tokenize else None

        for query, g in zip(q, gts):
            if tokenize and tokenizer:
                queries.append(tokenizer.tokenize(query))
                gt.append(tokenizer.tokenize(g))
            else:
                queries.append(query)
                gt.append(g)

        return queries, gt

    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading evaluation data: {e}")
        return None
