from agent.tools import tools
from agent.prompts import system_prompt
import agent.helpers as helpers
from textwrap import dedent
from openai import AzureOpenAI
import json
import os
from dotenv import load_dotenv
from pathlib import Path

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


def food_agent(max_steps=20, *, user_input: str=""): # pass user_input as keyword arg
    messages = [
      {"role":"system","content":dedent(system_prompt)}
    ]

    # first user turn
    #user_input = user_input

    # base case: user quits
    if not user_input or user_input.lower().strip() in ("quit","exit", "bye"):
        print("Goodbye 🍕")
        return
    # safe user_input to messages
    messages.append({"role":"user","content":user_input})

    # initiate reasoning loop 
    for step in range(1, max_steps+1):
        print(f"\n▶️ Step {step}: thinking…")    # for tracking 
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # save Agent's message to messages
        msg = response.choices[0].message
        messages.append(msg)

        # Tool-calling case
        if msg.tool_calls:
            # handle each call in order
            for call in msg.tool_calls:
                print("TOOL CALL ▶", call.function.name, call.function.arguments)
                name = call.function.name   # get function name
                args = json.loads(call.function.arguments or "{}")  # get function args

                # 2) we have all args -->  actually run the function
                if name=="get_wallet_balance":
                    result = helpers.get_wallet_balance(**args)
                elif name=="top_up_wallet":
                    result = helpers.top_up_wallet(**args)
                elif name=="order_food":
                    result = helpers.order_food(**args)
                elif name=="execute_agentic_rag":
                    result = helpers.execute_agentic_rag(**args)
                    return result
                else:
                    result = {"error":"unknown function"}

                # 3) append tool response 
                messages.append({
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

                # If order_pizza succeeded, print summary and exit
                if name == "order_food": 
                    #print(result)  # just to check 
                    meals = result.get("food_name")
                    price = result.get("food_price")
                    balance = result.get("balance")
                    print(f"🍕 Fooder: The meal(s) {meals} have been ordered for {price} €. The remaining balance on your pizza wallet is {balance} €. Goodbye!")
                    return  # exit entire pizza_agent function 

            # after handling call, move on to next thinking-step until finished
            continue

        # no no more tool calls invoked by thinking-step --> text-response 
        print("🍕 Fooder:", msg.content)
        break   # just for the RAG-loop: agent either returns RAG results and exits loop, or generates answer autonomously and then breaks
        # otherwise get another user turn
        nxt = input("YOU: ")
        if nxt.lower() in ("quit","exit"):
            print("🍕 Fooder: Goodbye 🍕"); return
        messages.append({"role":"user","content":nxt})
    
    #print("⚠️  Reached max steps without a final answer.")
