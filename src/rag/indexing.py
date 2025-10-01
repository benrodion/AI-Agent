# for indexing pipeline
import os
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from pathlib import Path

#---
# Set up retrieval pipeline
#---

# Set up embedding model & document store
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
document_store = InMemoryDocumentStore()

# file path
files_path = Path(__file__).resolve().parents[2] / "data" / "RAG_docs"

# modular pipeleine for embedding docs and writing to memory
index_pipeline = Pipeline()

index_pipeline.add_component("converter", PyPDFToDocument())
index_pipeline.add_component("cleaner", DocumentCleaner())
index_pipeline.add_component("splitter", DocumentSplitter(split_length=50, split_by="word"))
index_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
index_pipeline.add_component("writer", DocumentWriter(document_store=document_store,
                                                policy=DuplicatePolicy.SKIP))

# assemble pipeline
index_pipeline.connect("converter", "cleaner")
index_pipeline.connect("cleaner", "splitter")
index_pipeline.connect("splitter", "embedder")
index_pipeline.connect("embedder", "writer")

# file names
pdf_files = [files_path / f_name for f_name in os.listdir(files_path)]


def precompute_and_store_neighbors(m: int = 2):
    """
    Calculates the m-neighbours of a chunk as meta-data via cosine-similarity.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    docs = document_store.filter_documents()    # takes all docs
    if not docs:
        print("No documents in store; did you run index_pipeline first?")
        return

    M = np.vstack([d.embedding for d in docs]) # M = 2D-matrix with dims (N,D), N = num. chunks, D = embedding dim
    S = cosine_similarity(M, M)     # S = similarity matrix 
    np.fill_diagonal(S, -1.0)       # set self-similarity to -1 --> don't want a document to be picked as its own neighbour 

    topm = np.argpartition(-S, m, axis=1)[:, :m]    # find indexes of top-m similar docs
    
    # for each chunk, map neighour indices to doc-IDs
    for i, d in enumerate(docs):
        nn_ids = [docs[j].id for j in topm[i].tolist()]
        d.meta = dict(d.meta or {})
        d.meta["nn_ids"] = nn_ids

    document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
    print(f"Stored nn_ids for {len(docs)} chunks.")