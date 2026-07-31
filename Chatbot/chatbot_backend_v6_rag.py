import asyncio
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from typing import Annotated, TypedDict
from collections.abc import AsyncIterator

import aiosqlite
import httpx
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ------------------------------------------------------------
# State
# ------------------------------------------------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------------
# Normal LangChain tools
# ------------------------------------------------------------

search_tool = DuckDuckGoSearchResults(max_results=3)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


@tool
async def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest available stock quote for a symbol such as AAPL or TSLA
    using Alpha Vantage.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {"error": "ALPHA_VANTAGE_API_KEY is missing from the environment."}

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.upper(),
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        # Alpha Vantage can return an HTTP 200 response containing
        # an API limit or error message.
        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data:
            return {"error": data["Note"]}

        if "Information" in data:
            return {"error": data["Information"]}

        quote = data.get("Global Quote", {})

        if not quote:
            return {"error": f"No stock information was returned for {symbol.upper()}."}

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "latest_trading_day": quote.get("07. latest trading day"),
        }

    except httpx.HTTPError as exc:
        return {"error": f"Stock API request failed: {exc}"}
    except Exception as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------
# Async chatbot backend
# ------------------------------------------------------------


class AsyncChatbotBackend:
    def __init__(self) -> None:
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

        self.mcp_client: MultiServerMCPClient | None = None
        self.db_connection: aiosqlite.Connection | None = None
        self.checkpointer: AsyncSqliteSaver | None = None
        self.graph = None
        self.tools = []

        # RAG state
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vector_store: FAISS | None = None
        self.retriever = None
        self.current_document_name: str | None = None

        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize MCP tools, SQLite, and the LangGraph graph.

        This should run once on a persistent asyncio event loop.
        """
        if self._initialized:
            return

        # ----------------------------------------------------
        # MCP client
        # ----------------------------------------------------

        self.mcp_client = MultiServerMCPClient(
            {
                "Calculator": {
                    "transport": "stdio",
                    "command": "python3",
                    "args": [
                        "/home/rajab/Downloads/code/MCPs/local-mcp-fastmcp/main.py"
                    ],
                },
                "expense": {
                    # Current LangChain MCP adapters use "http"
                    # for Streamable HTTP MCP servers.
                    "transport": "http",
                    "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
                },
            }
        )

        # MultiServerMCPClient.get_tools() is asynchronous.
        mcp_tools = await self.mcp_client.get_tools()

        @tool
        def rag_tool(query: str) -> dict:
            """
            Search the PDF uploaded by the user.

            Always use this tool when the user refers to:
            the uploaded PDF, uploaded document, document, file, notes,
            report, paper, chapter, page, or content inside the PDF.

            Use web search only when the question is not about the uploaded PDF.
            """
            if self.retriever is None:
                return {
                    "error": (
                        "No PDF document has been processed yet. "
                        "Ask the user to upload and process a PDF first."
                    )
                }

            result = self.retriever.invoke(query)

            context = [document.page_content for document in result]

            metadata = [document.metadata for document in result]

            return {
                "query": query,
                "document": self.current_document_name,
                "context": context,
                "metadata": metadata,
            }

        # Calculator is no longer defined locally.
        self.tools = [
            search_tool,
            get_stock_price,
            rag_tool,
            *mcp_tools,
        ]

        llm_with_tools = self.llm.bind_tools(self.tools)

        # ----------------------------------------------------
        # Graph nodes
        # ----------------------------------------------------

        async def chat_node(state: ChatState) -> ChatState:
            messages = state["messages"]

            response = await llm_with_tools.ainvoke(messages)

            return {
                "messages": [response],
            }

        tool_node = ToolNode(self.tools)

        # ----------------------------------------------------
        # Async SQLite checkpointer
        # ----------------------------------------------------

        self.db_connection = await aiosqlite.connect("graph_history.db")

        self.checkpointer = AsyncSqliteSaver(self.db_connection)

        # Creates the required checkpoint tables when necessary.
        await self.checkpointer.setup()

        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        graph_builder = StateGraph(ChatState)

        graph_builder.add_node("chat_node", chat_node)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "chat_node")

        graph_builder.add_conditional_edges(
            "chat_node",
            tools_condition,
        )

        graph_builder.add_edge(
            "tools",
            "chat_node",
        )

        self.graph = graph_builder.compile(checkpointer=self.checkpointer)

        self._initialized = True

    def _build_pdf_vector_store(
        self,
        file_path: str,
        original_filename: str,
    ) -> dict:
        """
        Load a PDF, split it into chunks, generate embeddings,
        and create a FAISS vector store.

        This method performs blocking work and should be called
        through asyncio.to_thread().
        """
        pdf_path = Path(file_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {file_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        if not documents:
            raise ValueError("No readable content was found in the PDF.")

        # Replace the temporary file path with the user's original filename.
        for document in documents:
            document.metadata["source"] = original_filename

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("The PDF could not be split into text chunks.")

        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

        self.vector_store = FAISS.from_documents(
            chunks,
            self.embeddings,
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 4,
            }
        )

        self.current_document_name = original_filename

        return {
            "filename": original_filename,
            "pages": len(documents),
            "chunks": len(chunks),
        }

    async def index_pdf(
        self,
        file_path: str,
        original_filename: str,
    ) -> dict:
        """
        Build the PDF vector store without blocking the backend event loop.
        """
        return await asyncio.to_thread(
            self._build_pdf_vector_store,
            file_path,
            original_filename,
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized or self.graph is None:
            raise RuntimeError(
                "Backend has not been initialized. "
                "Call await backend.initialize() first."
            )

    async def astream_events(
        self,
        input_state: ChatState,
        config: dict,
    ) -> AsyncIterator[dict]:
        self._ensure_initialized()

        async for event in self.graph.astream_events(
            input_state,
            config=config,
            version="v2",
        ):
            yield event

    async def get_messages(
        self,
        thread_id: str,
    ) -> list[BaseMessage]:
        """
        Load the latest saved messages for one conversation.
        """
        self._ensure_initialized()

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        state = await self.graph.aget_state(config)

        if not state.values:
            return []

        return state.values.get("messages", [])

    async def get_all_threads(self) -> set[str]:
        """
        Return all stored LangGraph thread IDs.
        """
        self._ensure_initialized()

        thread_ids: set[str] = set()

        async for checkpoint in self.checkpointer.alist(None):
            thread_id = checkpoint.config.get(
                "configurable",
                {},
            ).get("thread_id")

            if thread_id:
                thread_ids.add(thread_id)

        return thread_ids

    async def close(self) -> None:
        """
        Close the SQLite connection.

        Streamlit normally keeps this backend alive until the application
        process exits, so this method is mainly useful for tests or scripts.
        """
        if self.db_connection is not None:
            await self.db_connection.close()

        self.db_connection = None
        self.checkpointer = None
        self.graph = None
        self._initialized = False


backend = AsyncChatbotBackend()
