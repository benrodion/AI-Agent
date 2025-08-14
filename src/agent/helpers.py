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

def execute_rag(question):
    print("Invoked RAG!")
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