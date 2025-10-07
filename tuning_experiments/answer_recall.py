from rag_eval.protocol import LLMClient
from rag_eval.client import AzureClient
from rag_eval.components import ClaimExtractor, EntailmentJudge
from rag_eval.data_models import EvalContainer, ClaimVerdict
from rag_eval.evaluators import AnswerPrecisionEvaluator, RetrievalPrecisionEvaluator
from rag_eval.coordinator import RAGEvaluator
from rag_eval.helpers import _safe_json
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import os


# ----- INDEXING -----
from rag.indexing import pdf_files, index_pipeline, document_store, precompute_and_store_neighbors
index_pipeline.run({"converter": {"sources": pdf_files}})

# compute neighbours and store info in metadata 
precompute_and_store_neighbors(m=2)

# ----- CLIENT INITIALIZATION -----
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
base_url = os.getenv("OPENAI_CHATCOMPLETIONS_URL")
model = os.getenv("OPENAI_DEPLOYMENT")
password = os.getenv("CORRECT_PASSWORD")

llm = AzureClient(
    api_key=api_key,
    api_version=api_version,
    base_url=base_url,
    model=model
)

# ----- RUN AGENT WITH SUBSET OF QUERY-GT-LIST -----
import agent.main as main
from rag_eval.eval_loop import rag_loop_agent
from pathlib import Path

# this file lives in: <project_root>/tuning_experiments/retrieval_precision.py
ROOT = Path(__file__).resolve().parents[1]      # go up to <project_root>
DATA = ROOT / "data"
EVAL_PATH = DATA / "problem_queries_answer_recall.json"

if not EVAL_PATH.exists():
    raise FileNotFoundError(f"Expected file at: {EVAL_PATH}")

result, token_usage, failures, tool_calls, tool_args, question_id = rag_loop_agent(path=str(EVAL_PATH))


# --- safe results ---
#intermediary result-save
import pickle

#save result
with open("result.pkl", "wb") as f:
       pickle.dump(result, f)

# access pickled result
with open("result.pkl", "rb") as f:
    result = pickle.load(f)

data = result


with open("token_usage.pkl", "wb") as f:
        pickle.dump(token_usage, f)

# access pickled result
with open("token_usage.pkl", "rb") as f:
    token_usage = pickle.load(f)

token_usage = token_usage


#save tool call metrics
with open("tool_calls.pkl", "wb") as f:
       pickle.dump(tool_calls, f)

# access pickled result
with open("tool_calls.pkl", "rb") as f:
    tool_calls = pickle.load(f)

tool_calls = tool_calls


# ----- EVALUATE RESULTS -----
# run evaluator
import pandas as pd

# run evaluator to create metrics: answer_precision, answer_recall, retrieval_precision, retrieval_recall
coord = RAGEvaluator(llm_extractor=llm, llm_judge=llm)

# run rag_evaluation and extract metrics from it 
answer_recall = []
answer_precision = []
retrieval_recall = []
retrieval_precision = []

for i, res in enumerate(data):
    #iteration tracker
    print(f"=== ITERATION-No. {i} ===")

    result = coord.evaluate_all(res)

    # extracting all variables needed
    answer_recall.append(result["answer_quality"]["answer_recall"])
    answer_precision.append(result["answer_quality"]["answer_precision"])
    retrieval_recall.append(result["retrieval_quality"]["retrieval_recall"])
    retrieval_precision.append(result["retrieval_quality"]["retrieval_precision"])


# ----- PARSE AND SAFE RESULTS -----
# tool calls
from agent.helpers import tool_call_parser
from agent.helpers import triplets_to_df

# turn tool call info into df 
tool_call_results = tool_call_parser(tool_calls=tool_calls)
df_tools = triplets_to_df(tool_call_results)


# create and save df with all info 
total_tokens = [dictionary.get("total") for dictionary in token_usage]
df_data_agentic = {
    "answer_recall": answer_recall,
    "answer_precision": answer_precision,
    "retrieval_recall": retrieval_recall,
    "retrieval_precision": retrieval_precision, 
    "total_tokens": total_tokens,
    #"question_id": question_id     # is currently longer than the other arrays --> leads to error
}


### ERROR OCCURS HERE
df_agentic = pd.DataFrame(df_data_agentic).reset_index(drop=True)

# Falls df_agentic ebenfalls 40 Zeilen hat: direkt concat
df_agentic = pd.concat([df_agentic, df_tools.reset_index(drop=True)], axis=1)

# ohne Index speichern (kein 'Unnamed: 0')
df_agentic.to_csv("df_agentic.csv", index=False)


# ----- EVALUATE PLAIN RAG -----
from rag_eval.data_models import EvalContainer
from rag.retrieval import basic_rag
from rag_eval.helpers import query_gt_extractor

queries, gts = query_gt_extractor(path=str(EVAL_PATH), tokenize=False)

# initiate lists to store EvalContainer-Ojbects
eval_conts = []
predicted_answers = []
retrieved_contexts = []
token_usage = []

for idx, query in enumerate(queries):

    # run RAG
    result = basic_rag.run({"query_embedder":{"text": query}, 
                            "prompt_builder":{"question": query}}, 
                            include_outputs_from="retriever")
    
    # store generated answer
    predicted_answers.append(result["llm"]["replies"][0])
    
    # for each question, store content from Document-object in list
    docs = result["retriever"]["documents"]
    retrieved_contexts.append([d.content for d in docs])

    # extract what is needed for EvalContainer-object
    query = queries[idx]
    ground_truth_answer = gts[idx]
    retrieved_texts = retrieved_contexts[idx]
    generated_answer = predicted_answers[idx]

    eval_cont = EvalContainer(query=query,
                                ground_truth_answer=ground_truth_answer,
                                generated_answer=generated_answer,
                                retrieved_texts=retrieved_texts # BUGFIX: retrieved_texts MUST be list, otherwise RetrievalPrecisionEvaluator will treat each letter of retrieved_texts as a doc_text
                                )
    # get number of tokens used 
    usage = None
    if "llm" in result and "usage" in result["llm"]:
        usage = result["llm"]["usage"]  # often contains prompt_tokens, completion_tokens, total_tokens

    
    eval_conts.append(eval_cont)
    token_usage.append(usage)

    
# ------ EVALUATE PLAIN RAG -----
import pandas as pd
import importlib
import rag_eval.evaluators as evaluators
import rag_eval.coordinator as coordinator

importlib.reload(evaluators)  # force reimport after changes
importlib.reload(coordinator)


# initiate llm 
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
base_url = os.getenv("OPENAI_CHATCOMPLETIONS_URL")
model = os.getenv("OPENAI_DEPLOYMENT")
password = os.getenv("CORRECT_PASSWORD")

llm = AzureClient(
    api_key=api_key,
    api_version=api_version,
    base_url=base_url,
    model=model
)


# run evaluator to create metrics: answer_precision, answer_recall, retrieval_precision, retrieval_recall
coord = RAGEvaluator(llm_extractor=llm, llm_judge=llm)

answer_recall = []
answer_precision = []
retrieval_recall = []
retrieval_precision = []

for i, res in enumerate(eval_conts):
    #iteration tracker
    print(f"=== ITERATION-No. {i+1} ===")

    result = coord.evaluate_all(res)

    # extracting all variables needed
    answer_recall.append(result["answer_quality"]["answer_recall"])
    answer_precision.append(result["answer_quality"]["answer_precision"])
    retrieval_recall.append(result["retrieval_quality"]["retrieval_recall"])
    retrieval_precision.append(result["retrieval_quality"]["retrieval_precision"])


# ----- SAFE PLAIN RAG TO DF -----
    # turn data into df for visualisation
df_data_plain_rag = {
     "answer_recall": answer_recall,
     "answer_precision": answer_precision,
     "retrieval_recall": retrieval_recall,
     "retrieval_precision": retrieval_precision
}

df_plain_rag = pd.DataFrame(df_data_plain_rag)

#print(df_plain_rag)

df_plain_rag.to_csv("df_plain_rag.csv", index=True)


#----- VISUALIZE RESULTS -----
import matplotlib.pyplot as plt
import pandas as pd

# import data
df_agentic_rag = pd.read_csv("df_agentic.csv", index_col=False)
df_plain_rag   = pd.read_csv("df_plain_rag.csv", index_col=False).drop(columns=["Unnamed: 0"])

columns = df_agentic_rag.columns[:4]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, col in enumerate(columns):
    ax = axes[i//2, i%2]

    # sort both series by agentic metric value (descending)
    order = df_agentic_rag[col].sort_values(ascending=False).index
    agentic_sorted = df_agentic_rag.loc[order, col].reset_index(drop=True)
    plain_sorted   = df_plain_rag.loc[order, col].reset_index(drop=True)

    ax.plot(agentic_sorted.index, agentic_sorted, label="Agentic RAG", alpha=0.7)
    ax.plot(plain_sorted.index, plain_sorted, label="Normales RAG", alpha=0.7)

    ax.set_title(f"{col} (sorted)")
    ax.set_xlabel("Fragen (nach Präzision sortiert)")
    ax.set_ylabel("Wert")
    ax.legend()

plt.tight_layout()
plt.show()

