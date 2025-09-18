from agent.tools import tools
from agent.prompts import system_prompt_extended  
import agent.helpers as helpers
from importlib import reload
helpers = reload(helpers)
from textwrap import dedent
from openai import AzureOpenAI
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from agent.llm import LLMWithMeter
from agent.telemetry import TokenUsage

ROOT = Path(__file__).resolve().parents[2]   # from src/agent/*.py up to project root
env_file = ROOT / ".env"
load_dotenv(env_file)

# load env-variables
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
model = os.getenv("OPENAI_DEPLOYMENT")
base_url = os.getenv("OPENAI_CHATCOMPLETIONS_URL")

# initiate client
client = AzureOpenAI(
     base_url=base_url,
     api_key=api_key,
     api_version=api_version
)

llm = LLMWithMeter(client=client)

def food_agent(
    max_steps: int = 6,
    max_rag_loops: int = 5,
    *,
    user_input: str = ""
):
    """
    Agenten-Loop:
    - LLM steuert eigenständig Retrieval → Draft → Selbstreflexion → ggf. Retry (neue Query / neues top_k) → Final.
    - Final wird durch 'FINAL_ANSWER:' signalisiert.
    """

    messages = [
        {"role": "system", "content": dedent(system_prompt_extended)}
    ]

    # base case: user quits
    if not user_input or user_input.lower().strip() in ("quit","exit","bye"):
        print("Goodbye 🍕")
        return

    # first user turn
    messages.append({"role":"user","content": user_input})

    rag_calls = 0
    final_answer = None
    combined_contexts: list[str] = []   # collect context over all RAG-calls

    for step in range(1, max_steps + 1):
        print(f"\n▶️ Step {step}: thinking…")

        response = llm.chat(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # agent decides which tool to use 
        )

        # Save assistant message
        msg = response.choices[0].message
        messages.append(msg)

        # Tool-calling case
        if msg.tool_calls:
            for call in msg.tool_calls:
                print("TOOL CALL ▶", call.function.name, call.function.arguments)

                name = call.function.name
                args = json.loads(call.function.arguments or "{}")

                # ---- RAG tool ---------------------------------------------------
                if name == "execute_agentic_rag":
                    # Fallback: if model omits top-k or tries to set a non-accepted value
                    if "top_k" not in args or not isinstance(args["top_k"], (int, float)) or args["top_k"] <= 0:
                        args["top_k"] = 5

                    # robust unpacking of result: either result + usage, or just result
                    _ret = helpers.execute_agentic_rag(**args)
                    if isinstance(_ret, tuple) and len(_ret) == 2:
                        result, usage = _ret
                    else:
                        result, usage = _ret, None

                    rag_calls += 1

                    combined_contexts.extend(helpers.extract_contexts_strict(result))

                    # Token-Metering of Helper-Calls
                    if usage:
                        print("RAG token usage:", usage)
                        llm.total_usage.prompt += usage.get("prompt_tokens", 0)
                        llm.total_usage.completion += usage.get("completion_tokens", 0)
                        llm.total_usage.total += usage.get("total_tokens", 0)

                    # Tool-Result fed back into message-context
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "tool_call_id": call.id,
                        "content": json.dumps(result)
                    })

                    # safety net against endless RAG-loops from tool calling
                    if rag_calls >= max_rag_loops:
                        print("ℹ️  max_rag_loops erreicht – Agent muss jetzt finalisieren.")
                    # move on to next thinking step 
                    continue

                # ---- other tools - irrelevant right now  ----
                elif name == "get_wallet_balance":
                    result = helpers.get_wallet_balance(**args)

                elif name == "top_up_wallet":
                    result = helpers.top_up_wallet(**args)

                elif name == "order_food":
                    result = helpers.order_food(**args)

                else:
                    result = {"error": f"unknown function: {name}"}

                # Append tool output to messages
                messages.append({
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

                # food ordering case
                if name == "order_food":
                    meals = result.get("food_name")
                    price = result.get("food_price")
                    balance = result.get("balance")
                    print(f"🍕 Fooder: The meal(s) {meals} have been ordered for {price} €. The remaining balance on your pizza wallet is {balance} €. Goodbye!")
                    return

            # keep iterating after tool use - agent might reflect again
            continue

        # no tool call this round --> check if finalisation has taken place
        content = msg.content or ""

        # convention from system_prompt: FINAL_ANSWER: <Text>
        if "FINAL_ANSWER:" in content:
            final_answer = content.split("FINAL_ANSWER:", 1)[1].strip()
            break

        # safety net
        if rag_calls >= max_rag_loops:
            # force final answer with last assistant message
            final_answer = content.strip()
            print("ℹ️  Abschluss mit letzter Nachricht, da max_rag_loops erreicht.")
            break

    # Print and return final response
    print("\n✅ Fertig.")
    if final_answer:
        print("\n🧾 FINAL_ANSWER:\n", final_answer)
    else:
        print("\n(Kein expliziter FINAL_ANSWER ermittelt – gebe letzte Assistant-Nachricht zurück.)")
        last = messages[-1]
        final_answer = ((last["content"] if isinstance(last, dict) else last.content) or "").strip()
        
    #print(combined_contexts)


    return {
        "answer": final_answer,
        "token_usage": llm.total_usage.to_dict(),
        "calls": llm.calls, # num of llm-calls (int)
        "rag_calls": rag_calls, # num of RAG-calls (int)
        "retrieved_contexts": combined_contexts
    }


