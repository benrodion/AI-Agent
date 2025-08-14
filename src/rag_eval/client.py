import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
from openai import AzureOpenAI

# load env-variables
load_dotenv("../../.env")
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
model = os.getenv("OPENAI_DEPLOYMENT")
base_url = os.getenv("OPENAI_CHATCOMPLETIONS_URL")

class AzureClient:
    "Complete implementation of LLMClient that runs with Azure OpenAI SDK"

    def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            api_version: str,
            model: str

    ) -> None: 
        
        # initiate client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            base_url=base_url
        )
        self.model = model

    # make call to LLM (wrapper for chat.completions.create)
    def complete(
            self,
            messages: List[Dict[str, str]],
            *, 
            model: str = "gpt-4.1",
            temperature: float = 0.0,
            max_tokens: Optional[int]=None,
    ) -> str:
        model= model

        kwargs = dict(model=model, 
                      messages=messages, 
                      temperature=temperature, 
                      max_tokens=max_tokens, 
                      response_format={"type": "json_object"}) # enforcing structured outputs

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""