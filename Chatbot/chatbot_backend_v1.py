from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage], add_messages
    ]  # here we can also use operator.add as reducer, but add_messages is especially optimized for chatbots and is recommended by langgraph.


def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    res = model.invoke(messages)
    return {"messages": [res]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
