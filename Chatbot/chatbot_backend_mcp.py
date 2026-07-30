from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import asyncio

from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Mcp client fastmcp local
client = MultiServerMCPClient(
    {
        "Calculator": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/home/rajab/Downloads/code/MCPs/local-mcp-fastmcp/main.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        },
    }
)

# ---------------------------------


class ChatState(TypedDict):
    """LLM node that may answer or request a tool call"""

    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    tools = await client.get_tools()
    print(tools)
    llm_with_tools = llm.bind_tools(tools)

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
    chatbot = await build_graph()
    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Add an expense of Rs. 1100 - udemy course on 25 july."
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
