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

        # Calculator is no longer defined locally.
        self.tools = [
            search_tool,
            get_stock_price,
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

    def _ensure_initialized(self) -> None:
        if not self._initialized or self.graph is None:
            raise RuntimeError(
                "Backend has not been initialized. "
                "Call await backend.initialize() first."
            )

    async def astream(
        self,
        input_state: ChatState,
        config: dict,
    ) -> AsyncIterator[tuple]:
        """
        Stream LangGraph messages asynchronously.
        """
        self._ensure_initialized()

        async for event in self.graph.astream(
            input_state,
            config=config,
            stream_mode="messages",
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
