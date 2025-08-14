from typing import List, Dict, Optional,  Protocol

class LLMClient(Protocol): # LLMClient is a type, not an object
    "Interface for swapping out LLM. Defines inputs for .complete()-method. Returns a string"
    def complete(
            self,
            messages: List[Dict[str, str]],
            *,  # enhanced readability of function
            model: str,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
    ) -> str: ...
