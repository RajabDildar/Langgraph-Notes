import os
import sqlite3
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchResults

import requests
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# -------------------------------------
# Tools
# -------------------------------------
search_tool = DuckDuckGoSearchResults(max_results=3)


@tool
def calculator(a: float, b: float, operation: str) -> dict:
    """This tool will do basic arithmetic operations on two integers. supported arithmetic operations: add, mul, div, sub"""
    try:
        if operation == "add":
            result = a + b
        elif operation == "sub":
            result = a - b
        elif operation == "mul":
            result = a * b
        elif operation == "div":
            if b == 0:
                return {"error": "division by zero is not allowed"}
            result = a / b
        else:
            return {"error": "Unsupported operation"}

        return {"a": a, "b": b, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


ALPHA_VANTAGE_API_KEY = os.environ["ALPHA_VANTAGE_API_KEY"]


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g, 'AAPL','TSLA') using Alpha vantage with API key in the url."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)
    return r.json()


tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------


class ChatState(TypedDict):
    """LLM node that may answer or request a tool call"""

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    """LLM node that may answer or request a tool call"""
    messages = state["messages"]
    res = llm_with_tools.invoke(messages)
    return {"messages": [res]}


tool_node = ToolNode(tools)


conn = sqlite3.connect("graph_history.db", check_same_thread=False)

checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def get_all_threads():
    thread_ids = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        thread_ids.add(thread_id)

    return thread_ids
