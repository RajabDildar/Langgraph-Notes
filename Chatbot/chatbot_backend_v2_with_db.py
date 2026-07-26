import sqlite3
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    res = model.invoke(messages)
    return {"messages": [res]}


conn = sqlite3.connect("graph_history.db", check_same_thread=False)

checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def get_all_threads():
    thread_ids = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        thread_ids.add(thread_id)

    return thread_ids
