def order_food(food_name: list, food_price, wallet_balance):
    "Function to simulate placement of food order"
    print("Invoked order_food!")
    if food_price > wallet_balance: 
        print("Your wallet balance is insufficient. Please top it up.")

    else:
        balance = wallet_balance - food_price
        result = {
            "food_name": food_name,
            "food_price": food_price,
            "balance": balance
        }
        return result
    
#---
# Define "Wallet"-class 
#---
import os
correct_password = os.getenv("PASSWORD")

class Wallet: 
    "A class that initiates a wallet for food ordering. "
    "Inputs: "
    "balance (int): amount of money on wallet. Can be topped up if needed."
    def __init__(self, balance=30): # default balance
        self.balance = balance

    # function to access account balance 
    def get_balance(self, password: str):
        print("Invoked get_balance!")
        if password == correct_password:
            print(f"Your balance is EUR {self.balance}.")
            return self.balance
        else:
            print("Incorrect password. Please try again.")
            pass
    


    # function to top-up balance
    def top_up(self, amount: float, password: str):
        print("Invoked top_up!")
        if password == correct_password: 
            self.balance += amount 
            return self.balance
        else:
            print("Incorrect password. Please try again.")
            pass

#---
# Wrapper functions for class methods
#---
# initiate wallet 
my_wallet = Wallet()
def get_wallet_balance(password: str):
    return my_wallet.get_balance(password)

def top_up_wallet(amount: float, password: str):
    return my_wallet.top_up(amount, password)

#---
# 2. Wrapper Functions for Class-methods   
#---
# wrapper function for RAG
from rag.retrieval import basic_rag

def execute_plain_rag(question):
    print("Invoked plain RAG!")
    result = basic_rag.run({"query_embedder": {"text":question},
                            "prompt_builder":{"question":question}},
                            include_outputs_from="retriever")
    # store docs
    docs = result["retriever"]["documents"]

    # serialize so it can be converted to JSON
    serialized = [
        {
            "id": d.id,
            "content": d.content,
            "metadata": d.meta,
            "score": d.score
        }
        for d in docs
    ]

    return serialized

from rag.helpers import expand_with_precomputed_neighbors
from rag.indexing import document_store
def execute_agentic_rag(question: str, top_k: int):
    """
    A modified version of execute_plain_rag. 

        Parameters:
            question: a string

        Returns:
            generated_answer: a string 
            context: a list of strings with the retrieved context  + top-m neighbours for eqch question.

    """
    print("Invoked agentic RAG!")
    generated_answers = []
    retrieved_context = []

    result = basic_rag.run(
        data={
            # 1) Send text to the embedder 
            "query_embedder": {"text": question},

            # 2) Override retriever-default for `top_k` here
            "retriever": {"top_k": top_k},

            # 3) Provide template vars to the prompt builder
            "prompt_builder": {"question": question},
        },
        include_outputs_from="retriever",
    )

    print(f"Top K is: {top_k}")     # for tracking 
    generated_answers.append(result["llm"]["replies"][0])

     # --- NEW: expand with neighbors

    docs = result["retriever"]["documents"]

    expanded_docs = expand_with_precomputed_neighbors(
        anchors=docs,
        document_store=document_store,
        m=2,
        same_parent_only=True
    )

    retrieved_context.append([d.content for d in docs])

    # --- NEW: try to extract token usage
    usage = None
    if "llm" in result and "usage" in result["llm"]:
        usage = result["llm"]["usage"]  # often contains prompt_tokens, completion_tokens, total_tokens

    result = (generated_answers, retrieved_context)

    return result, usage


# STRICT extractor: akzeptiert NUR das von dir gezeigte Objektformat
def extract_contexts_strict(rag_tuple) -> list[str]:
    """
    Expects: ( [answer_strs], [ [context_strs ... ] ] )
    Returns: list[str]  (the contexts from the inner list)
    """
    # Unpack the exact structure
    answer_list, contexts_outer = rag_tuple          # contexts_outer == [ list[str] ]
    ctx_list = contexts_outer[0]                     # list[str]
    # Keep only strings (defensive but still strict)
    return [c for c in ctx_list if isinstance(c, str)]


import pandas as pd
import re


def tool_call_parser(tool_calls: list[list[dict]]) -> list:
    """
    Input per question: a list of tool-call dicts (0..n).
    Output per question: THREE dicts (name_*, query_*, top_k_*) with 5 slots each.
    Final output: flat list like [name_dict, query_dict, topk_dict,  name_dict, query_dict, topk_dict, ...]
    """
    all_tools = []

    for tool_call in tool_calls:
        tool_name, query, top_k = {}, {}, {}

        for counter in range(5):
            if counter < len(tool_call):
                item = tool_call[counter]
                args = item.get("args", {}) or {}
                tool_name[f"name_{counter}"]  = item.get("name")
                query[f"query_{counter}"]     = args.get("question")
                top_k[f"top_k_{counter}"]     = args.get("top_k")
            else:
                tool_name[f"name_{counter}"]  = None
                query[f"query_{counter}"]     = None
                top_k[f"top_k_{counter}"]     = None

        # IMPORTANT: extend ONCE per question (not inside the loop!)
        all_tools.extend([tool_name, query, top_k])

    return all_tools


def triplets_to_df(obj):
    """obj is the flat [name, query, topk, name, query, topk, ...] list."""
    if len(obj) % 3 != 0:
        raise ValueError(f"Length {len(obj)} not divisible by 3.")

    rows = []
    for i in range(0, len(obj), 3):
        name_d, query_d, topk_d = obj[i], obj[i+1], obj[i+2]
        row = {}
        for d in (name_d, query_d, topk_d):
            row.update({k: v for k, v in d.items()
                        if k.startswith(("name_", "query_", "top_k_"))})
        rows.append(row)

    df = pd.DataFrame(rows)

      # order columns by trailing index
    def order(prefix):
        cols = [c for c in df.columns if re.match(rf"^{re.escape(prefix)}_\d+$", c)]
        return sorted(cols, key=lambda c: int(re.search(r'(\d+)$', c).group(1)))

    expected = order("name") + order("query") + order("top_k")
    for col in expected:
        if col not in df:
            df[col] = None
    return df.reindex(columns=expected)


