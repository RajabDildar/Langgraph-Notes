from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import asyncio

from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# -------------------------------------
# Tools
# -------------------------------------


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


tools = [calculator]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------


class ChatState(TypedDict):
    """LLM node that may answer or request a tool call"""

    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():

    async def chat_node(state: ChatState) -> ChatState:
        """LLM node that may answer or request a tool call"""
        messages = state["messages"]
        res = await llm_with_tools.ainvoke(messages)
        return {"messages": [res]}

    tool_node = ToolNode(tools)  # this node is by default async

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    return chatbot


async def main():
    chatbot = build_graph()
    result = await chatbot.ainvoke({"messages": HumanMessage("what is 15 + 12?")})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
