from typing import List, Dict, Any

tools: List[Dict[str, Any]] = [
     {
        "type": "function",
        "function": {
            "name": "get_wallet_balance",
            "description": "A function that returns the balance of the user's pizza wallet if the correct password is entered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "description": "The password to access the wallet."
                    }
                },
                "required": ["password"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top_up_wallet",
            "description": "A function to increase the balance on the wallet, if the password is correct..",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The amount by which to increase the balance on the wallet.t"
                    },
                    "password": {
                        "type": "string",
                    "description": "The password to access the wallet."
                    }
                },
                "required": ["password", "amount"],
                "additionalProperties": False
            },
            "strict": True
        }
    }, 
    {
        "type": "function",
        "function": {
            "name": "order_food",
            "description": "Places an order for the meal desired by the user, provided  his wallet balance is sufficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "food_name": {
                        "type": "string",
                        "description": "A list containing the meal or meals to be ordered."
                    },
                    "food_price": {
                        "type": "number",
                        "description": "How much the meal or meals costs."
                    },
                    "wallet_balance": {
                        "type": "number",
                        "description": "The user's wallet balance. It can be inferred from pizza_wallet."
                    }
                },
                "required": ["food_name", "food_price", "wallet_balance"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_agentic_rag",
            "description": "For the agent from main. Makes it possible to retrieve documents like restaurant menus as well as additional information on a country's cuisine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question of the user based on which RAG retrieves the right documents."
                    },
                    "top_k":{
                        "type": "number",
                        "description": "The amount of documents to retrieve in order to create an answer with high answer and retrieval precision."
                    }
                },
                "required": ["question", "top_k"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]