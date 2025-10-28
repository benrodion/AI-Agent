from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
from pydantic import Field
from fastmcp import FastMCP
import os


# ------
# MCP Server
# ------
server = FastMCP(
    name="Food Ordering Tools Server",
    instructions=(
        "This server exposes tools for asking questions about restaurant menus, ordering food, managing a wallet with password protection."
    ),
)

# --------
# Wallet class
# --------
_CORRECT_PASSWORD = "anton" # for test purposes only  


class Wallet:
    """A simple wallet for food ordering."""

    def __init__(self, balance: float = 30.0):
        self.balance = float(balance)

    def get_balance(self, password: str) -> float:
        if not _CORRECT_PASSWORD:
            # if no PASSWORD is configured
            raise ValueError("Server PASSWORD is not configured in environment.")
        if password != _CORRECT_PASSWORD:
            raise ValueError("Incorrect password.")
        return self.balance

    def top_up(self, amount: float, password: str) -> float:
        if not _CORRECT_PASSWORD:
            raise ValueError("Server PASSWORD is not configured in environment.")
        if password != _CORRECT_PASSWORD:
            raise ValueError("Incorrect password.")
        if amount < 0:
            raise ValueError("Top-up amount must be non-negative.")
        self.balance += amount
        return self.balance


# Single shared wallet instance
_my_wallet = Wallet()


# -------
# Tools
# -------

@server.tool(
    name="order_food",
    description="Simulate placing a food order using a provided wallet balance (not the server wallet)."
)
def order_food_mcp(
    food_name: List[str] = Field(..., description="List of food item names to order."),
    food_price: float = Field(..., description="Total price for the order in EUR.", ge=0),
    wallet_balance: float = Field(..., description="Available balance in the wallet used for this order.", ge=30)
) -> Dict[str, Any]:
    """
    MCP-friendly version of my "order_food" function.
    Uses the provided wallet_balance (does not touch the server's wallet).
    """
    if food_price > wallet_balance:
        return {
            "status": "insufficient_funds",
            "message": "Your wallet balance is insufficient. Please top it up.",
            "food_name": food_name,
            "food_price": food_price,
            "provided_wallet_balance": wallet_balance,
        }

    new_balance = wallet_balance - food_price
    return {
        "status": "success",
        "food_name": food_name,
        "food_price": food_price,
        "balance": new_balance,
    }


@server.tool(
    name="get_wallet_balance",
    description="Get the current balance of the server-side wallet (password protected)."
)
def get_wallet_balance_mcp(
    password: str = Field(..., description="The wallet password (must match `correct_password`).")
) -> Dict[str, Any]:
    """
    Wraps Wallet.get_balance with password protection.
    """
    try:
        balance = _my_wallet.get_balance(password)
        return {"status": "success", "balance": balance}
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@server.tool(
    name="top_up_wallet",
    description="Top up the server-side wallet (password protected)."
)
def top_up_wallet_mcp(
    amount: float = Field(..., description="Amount to add to the wallet in EUR.", ge=0),
    password: str = Field(..., description="The wallet password (must match the server's PASSWORD env var).")
) -> Dict[str, Any]:
    """
    Wraps Wallet.top_up with password protection.
    """
    try:
        new_balance = _my_wallet.top_up(amount, password)
        return {"status": "success", "balance": new_balance}
    except ValueError as e:
        return {"status": "error", "error": str(e)}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Start an HTTP MCP transport; adjust host/port as needed.
    server.run(transport="http", host="127.0.0.1", port=8901)
