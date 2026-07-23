import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
# loading history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


user_input = st.chat_input("Type here..")
msg_history = st.session_state["message_history"]
config = {"configurable": {"thread_id": "1"}}

if user_input:
    msg_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        ai_msg = st.write_stream(
            message.content
            for message, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
        )
    msg_history.append({"role": "assistant", "content": ai_msg})
