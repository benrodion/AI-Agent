# for RAG-pipeline
from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder 
from haystack.components.generators import AzureOpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils.auth import Secret
from haystack import Pipeline
from rag.indexing import embedding_model, document_store
from haystack.components.preprocessors import DocumentCleaner


# for client
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # from src/agent/*.py up to project root
env_file = ROOT / ".env"
load_dotenv(env_file)

# load variables for client
base_url = os.getenv("OPENAI_CHATCOMPLETIONS_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_DEPLOYMENT")
api_version = os.getenv("OPENAI_API_VERSION")

# Haystack needs base_url in different form 
base_endpoint = base_url.split('/openai')[0] + '/' if '/openai' in base_url else base_url



# prompt template
template = """
You have to answer the following question based on your available general knowledge and the given context.

Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:

"""
# initiate retrieval pipeline
basic_rag = Pipeline()

# enrich with components
basic_rag.add_component("query_embedder",
                        SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=True))
basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store,
                                                                top_k=10)) 
basic_rag.add_component("prompt_builder", PromptBuilder(template=template))
basic_rag.add_component("llm", AzureOpenAIGenerator(
    azure_endpoint=base_endpoint,
    api_key=Secret.from_token(api_key),
    api_version=api_version,
    azure_deployment=model
))

basic_rag.connect("query_embedder", "retriever.query_embedding")
basic_rag.connect("retriever", "prompt_builder.documents")
basic_rag.connect("prompt_builder", "llm")

