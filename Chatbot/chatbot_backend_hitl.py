from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


@tool
def calculator(a: float, b: float, operation: str) -> dict:
    """Perform basic arithmetic operations on two numbers."""

    decision = interrupt(
        f"""
        Calculator tool wants to execute:

        Operation : {operation}
        A         : {a}
        B         : {b}

        Approve? (yes/no)
        """
    )

    if str(decision).strip().lower() not in ("yes", "y"):
        return {"approved": "no", "message": "User denied calculator execution."}

    if operation == "add":
        result = a + b
    elif operation == "sub":
        result = a - b
    elif operation == "mul":
        result = a * b
    elif operation == "div":
        if b == 0:
            return {"error": "Division by zero is not allowed."}
        result = a / b
    else:
        return {"error": "Unsupported operation."}

    return {
        "approved": "yes",
        "a": a,
        "b": b,
        "operation": operation,
        "result": result,
    }


tools = [calculator]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

graph = builder.compile(checkpointer=InMemorySaver())

config = {
    "configurable": {
        "thread_id": "thread_1",
    }
}


while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    while "__interrupt__" in result:
        question = result["__interrupt__"][0].value
        print(question)

        answer = input(">>> ")

        result = graph.invoke(
            Command(resume=answer),
            config=config,
        )
    # -----------------------------------------

    print("\nAssistant:", result["messages"][-1].content)
