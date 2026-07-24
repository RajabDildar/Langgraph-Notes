import uuid

import streamlit as st
from chatbot_backend_with_db import chatbot, get_all_threads
from langchain_core.messages import AIMessage, HumanMessage


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    # Don't add the thread yet.
    # It will be added only after the first user message.
    st.session_state.thread_id = generate_thread_id()
    st.session_state.message_history = []


def add_thread(thread_id, title):
    st.session_state.chat_threads[thread_id] = title


def load_conversations(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})

    if not state.values:
        return []

    return state.values.get("messages", [])


def load_all_threads():
    chats = {}

    for thread_id in get_all_threads():
        messages = load_conversations(thread_id)

        title = "New Chat"

        for msg in messages:
            if isinstance(msg, HumanMessage):
                title = msg.content[:40]
                if len(msg.content) > 40:
                    title += "..."
                break

        chats[thread_id] = title

    return chats


# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    # {thread_id: title}
    st.session_state.chat_threads = load_all_threads()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Conversations")

for tid, title in reversed(list(st.session_state.chat_threads.items())):
    if st.sidebar.button(title, key=tid):
        st.session_state.thread_id = tid

        messages = load_conversations(tid)

        history = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue

            history.append(
                {
                    "role": role,
                    "content": msg.content,
                }
            )

        st.session_state.message_history = history
        st.rerun()


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------

st.title("Chatbot")

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    thread_id = st.session_state.thread_id

    # Create thread only when first message is sent.
    if thread_id not in st.session_state.chat_threads:
        title = user_input[:40]
        if len(user_input) > 40:
            title += "..."

        add_thread(thread_id, title)

    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    with st.chat_message("assistant"):
        response = st.write_stream(
            message.content
            for message, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input),
                    ]
                },
                config=config,
                stream_mode="messages",
            )
        )

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
