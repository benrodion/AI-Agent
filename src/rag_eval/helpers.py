import re
import json
from typing import Optional

def _safe_json(s: str) -> Optional[dict]:
    "For more robust JSON-parsing, in case LLM"
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