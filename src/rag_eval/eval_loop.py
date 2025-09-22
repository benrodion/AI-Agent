from rag_eval.helpers import query_gt_extractor
from rag_eval.data_models import EvalContainer
# force reload of agent because I update it often
import importlib
import agent.main as main
importlib.reload(main)  # forces reload right now


def rag_loop_agent() -> list[EvalContainer]:
    eval_conts = []
    token_usage = []
    failures = []
    queries, gts = query_gt_extractor(path="data/agent_eval_questions.json", tokenize=False) # get questions and ground truth
    # print("lens:", len(queries), len(gts))  # sanity check
    for idx, (q, gt) in enumerate(zip(queries, gts)): # maybe we can do without zip and do not need to iterate over gts because I only need idx to access the right gt
        full_result = main.food_agent(user_input=q)


        if full_result: 

            # extract what is needed for EvalContainer-object
            query = queries[idx]
            ground_truth_answer = gts[idx]
            retrieved_texts = full_result.get("retrieved_contexts", [])
            generated_answer = full_result.get("answer", "")

            # extract usage
            usage = full_result.get("token_usage")

            eval_cont = EvalContainer(query=query,
                                      ground_truth_answer=ground_truth_answer,
                                      generated_answer=generated_answer,
                                      retrieved_texts=retrieved_texts 
                                      )
            
            eval_conts.append(eval_cont)
            token_usage.append(usage)

        # safeguard: if RAG does not get executed I know exactly at which question
        else: 
            print(f"Failed question: {idx}")
            failures.append(idx)

    return eval_conts, token_usage, failures