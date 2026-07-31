import tempfile
from pathlib import Path
import asyncio
import queue
import threading
import uuid
from collections.abc import AsyncIterator, Coroutine, Generator
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbot_backend_v6_rag import backend


# ------------------------------------------------------------
# Persistent asyncio runner
# ------------------------------------------------------------


class AsyncRunner:
    """
    Runs all async backend operations on one persistent event loop.

    Streamlit itself reruns the Python script synchronously. Keeping the
    LangGraph backend, aiosqlite connection, and MCP calls on one event loop
    prevents 'event loop is closed' and cross-loop errors.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
        )

        self.thread.start()

        # Initialize MCP tools, SQLite, and LangGraph once.
        self.run(backend.initialize())

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """
        Run an async coroutine on the persistent event loop and return
        its result synchronously.
        """
        future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self.loop,
        )

        return future.result()

    def stream(
        self,
        async_iterator: AsyncIterator,
    ) -> Generator[Any, None, None]:
        """
        Convert an async iterator into a synchronous generator.

        LangGraph still uses graph.astream() internally. This bridge only
        lets Streamlit consume its events safely from the main thread.
        """
        event_queue: queue.Queue = queue.Queue()
        finished = object()

        async def produce_events() -> None:
            try:
                async for item in async_iterator:
                    event_queue.put(("item", item))
            except Exception as exc:
                event_queue.put(("error", exc))
            finally:
                event_queue.put(("finished", finished))

        asyncio.run_coroutine_threadsafe(
            produce_events(),
            self.loop,
        )

        while True:
            event_type, value = event_queue.get()

            if event_type == "item":
                yield value

            elif event_type == "error":
                raise value

            elif event_type == "finished":
                break


@st.cache_resource
def get_async_runner() -> AsyncRunner:
    """
    Create one persistent async runner for the Streamlit application.
    """
    return AsyncRunner()


runner = get_async_runner()


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat() -> None:
    # Do not save the thread until its first user message.
    st.session_state.thread_id = generate_thread_id()
    st.session_state.message_history = []


def add_thread(thread_id: str, title: str) -> None:
    st.session_state.chat_threads[thread_id] = title


def load_conversation(thread_id: str):
    return runner.run(backend.get_messages(thread_id))


def create_thread_title(messages) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content

            # HumanMessage content is normally a string here.
            if not isinstance(content, str):
                content = str(content)

            title = content[:40]

            if len(content) > 40:
                title += "..."

            return title

    return "New Chat"


def load_all_threads() -> dict[str, str]:
    chats: dict[str, str] = {}

    thread_ids = runner.run(backend.get_all_threads())

    for thread_id in thread_ids:
        messages = load_conversation(thread_id)
        chats[thread_id] = create_thread_title(messages)

    return chats


def messages_to_ui_history(messages) -> list[dict[str, str]]:
    """
    Convert LangChain messages into messages that Streamlit can display.

    Tool messages are excluded because they are internal conversation events.
    """
    history = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        content = message.content

        if not content:
            continue

        if not isinstance(content, str):
            content = str(content)

        history.append(
            {
                "role": role,
                "content": content,
            }
        )

    return history


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = load_all_threads()

if "rag_document_name" not in st.session_state:
    st.session_state.rag_document_name = None


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("Chatbot")

if st.sidebar.button(
    "New Chat",
    use_container_width=True,
):
    reset_chat()
    st.rerun()


# ------------------------------------------------------------
# PDF RAG
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Document RAG")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded_pdf is not None and st.sidebar.button(
    "Process PDF",
    use_container_width=True,
):
    temporary_path = None

    try:
        with st.sidebar.status(
            "Processing PDF...",
            expanded=True,
        ) as status:
            # PyPDFLoader expects a filesystem path.
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as temporary_file:
                temporary_file.write(uploaded_pdf.getbuffer())
                temporary_path = temporary_file.name

            result = runner.run(
                backend.index_pdf(
                    file_path=temporary_path,
                    original_filename=uploaded_pdf.name,
                )
            )

            st.session_state.rag_document_name = result["filename"]

            status.update(
                label="PDF processed",
                state="complete",
                expanded=False,
            )

        st.sidebar.success(
            f"Ready: {result['filename']}\n\n"
            f"Pages: {result['pages']} | "
            f"Chunks: {result['chunks']}"
        )

    except Exception as exc:
        st.sidebar.error(f"PDF processing failed: {exc}")

    finally:
        if temporary_path:
            Path(temporary_path).unlink(
                missing_ok=True,
            )

if st.session_state.rag_document_name:
    st.sidebar.caption(f"Current document: {st.session_state.rag_document_name}")


# ------------------------------------------------------------
# Conversations
# ------------------------------------------------------------


st.sidebar.markdown("---")
st.sidebar.subheader("Conversations")

conversation_items = list(st.session_state.chat_threads.items())

for thread_id, title in reversed(conversation_items):
    is_current_thread = thread_id == st.session_state.thread_id

    button_label = title

    if is_current_thread:
        button_label = f"• {title}"

    if st.sidebar.button(
        button_label,
        key=f"thread-{thread_id}",
        use_container_width=True,
    ):
        messages = load_conversation(thread_id)

        st.session_state.thread_id = thread_id
        st.session_state.message_history = messages_to_ui_history(messages)

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

    # Add the conversation to the sidebar only after its first message.
    if thread_id not in st.session_state.chat_threads:
        title = user_input[:40]

        if len(user_input) > 40:
            title += "..."

        add_thread(thread_id, title)

    user_message = {
        "role": "user",
        "content": user_input,
    }

    st.session_state.message_history.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": {
            "thread_id": thread_id,
        },
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        tool_status = None

        def response_stream():
            tool_status = None

            event_stream = backend.astream_events(
                input_state={"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            for event in runner.stream(event_stream):
                event_name = event["event"]
                # ----------------------------------------
                # Tool started
                # ----------------------------------------
                if event_name == "on_tool_start":
                    tool_name = event["name"]

                    if tool_status is None:
                        tool_status = st.status(
                            f"Using tool: {tool_name}",
                            expanded=True,
                        )
                    else:
                        tool_status.update(
                            label=f"Using tool: {tool_name}",
                            expanded=True,
                        )

                # ----------------------------------------
                # Tool finished
                # ----------------------------------------
                elif event_name == "on_tool_end":
                    if tool_status is not None:
                        tool_status.update(
                            label="Tool finished",
                            state="complete",
                            expanded=False,
                        )

                # ----------------------------------------
                # LLM streaming
                # ----------------------------------------
                elif event_name == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]

                    if chunk.content:
                        yield chunk.content

            # --------------------------------------------
            # Mark tool work as completed
            # --------------------------------------------

            if tool_status is not None:
                tool_status.update(
                    label="Tools completed",
                    state="complete",
                    expanded=False,
                )

        try:
            response = st.write_stream(response_stream())

        except Exception as exc:
            if tool_status is not None:
                tool_status.update(
                    label="Tool or model request failed",
                    state="error",
                    expanded=True,
                )

            st.error(f"Request failed: {exc}")
            response = ""

    if response:
        st.session_state.message_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
