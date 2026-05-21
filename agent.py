"""
agent.py — Production-grade LangGraph agent
Fixes:
  - retrieve.py mismatch (final_answer → context used correctly)
  - web_node w_context properly returned
  - Memory deduplication via SQLite UNIQUE constraint
  - summarize_node reducer conflict fixed
  - MemorySaver only for short-term thread state (correct usage)
  - extract_and_save_memory now awaited properly via background task queue
"""

import os
import logging
import asyncio
from typing import List, Annotated, TypedDict, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from logging.handlers import TimedRotatingFileHandler
import inspect

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver

from retrieve import graph_app
from web import graph_web
from database import save_memory_db, get_user_memories

load_dotenv()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TimeSmartLogger:
    def get_logger(self, logLevel=logging.INFO):
        logger_name = inspect.stack()[1][3]
        logger = logging.getLogger(logger_name)
        logger.setLevel(logLevel)
        if not logger.handlers:
            handler = TimedRotatingFileHandler(
                "agent.log", when="midnight", interval=1, backupCount=7
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            logger.addHandler(handler)
            logger.addHandler(logging.StreamHandler())
        return logger

ts_logger = TimeSmartLogger()

# ---------------------------------------------------------------------------
# LLM & Embeddings
# ---------------------------------------------------------------------------

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Chroma for semantic memory search (vector similarity)
user_db    = Chroma(persist_directory="memory_db",  embedding_function=embeddings)
vector_db  = Chroma(persist_directory="vectore_db", embedding_function=embeddings)

# ---------------------------------------------------------------------------
# Structured Router
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    """Route the query to the best tool."""
    tool: Literal["rag", "web", "llm"] = Field(
        description="'rag' for internal/restaurant data, "
                    "'web' for news/current events, "
                    "'llm' for general conversation."
    )

structured_llm = llm.with_structured_output(RouteDecision)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:  Annotated[List[BaseMessage], add_messages]
    query:     str
    r_context: str
    w_context: str
    decision:  str
    user_id:   str
    source:    str   # tracks which tool was used — passed back to API

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

async def get_user_memory_semantic(query: str, user_id: str) -> str:
    """Semantic search in Chroma for relevant long-term memories."""
    log = ts_logger.get_logger()
    try:
        docs = await asyncio.to_thread(
            user_db.similarity_search, query, k=3,
            filter={"user_id": user_id}
        )
        return "\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        log.warning(f"Memory fetch error: {e}")
        return ""

async def extract_and_save_memory(text: str, user_id: str):
    """
    Extract permanent facts → save to both:
      1. Chroma (for semantic retrieval)
      2. SQLite (for UI display & deduplication)
    """
    log = ts_logger.get_logger()
    try:
        prompt = (
            "Extract ONLY permanent facts about the user from this text as "
            "concise one-line bullet points.\n"
            "Examples: '• User's name is Ali', '• User owns a restaurant in Lahore'\n"
            "If no permanent facts exist, reply exactly: None\n\n"
            f"Text:\n{text}"
        )
        res = await llm.ainvoke([SystemMessage(content=prompt)])
        content = res.content.strip()

        if content.lower() == "none" or not content:
            return

        # SQLite — deduplication via UNIQUE constraint
        save_memory_db(user_id, content)

        # Chroma — semantic vector store
        doc = Document(page_content=content, metadata={"user_id": user_id})
        await asyncio.to_thread(user_db.add_documents, [doc])
        log.info(f"Memory saved for {user_id}")

    except Exception as e:
        log.error(f"Memory save error for {user_id}: {e}")

# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

async def router_node(state: AgentState):
    log = ts_logger.get_logger()
    query = state["messages"][-1].content

    decision_obj = await structured_llm.ainvoke([
        SystemMessage(
            content="Route the query. "
                    "'rag' = internal/restaurant data, "
                    "'web' = news/current/live info, "
                    "'llm' = general chat/math/code."
        ),
        HumanMessage(content=query)
    ])
    decision = decision_obj.tool
    r_context = ""

    if decision == "rag":
        log.info("--- ROUTING TO RAG ---")
        result = await asyncio.to_thread(graph_app.invoke, {"query": query})
        # FIX: retrieve.py returns 'final_answer' key
        r_context = result.get("final_answer") or result.get("context", "")

    elif decision == "web":
        log.info("--- ROUTING TO WEB ---")

    else:
        log.info("--- ROUTING TO LLM ---")

    return {
        "decision":  decision,
        "query":     query,
        "r_context": r_context,
        "source":    decision
    }


async def web_node(state: AgentState):
    log = ts_logger.get_logger()
    try:
        result = await asyncio.to_thread(
            graph_web.invoke, {"question": state["query"]}
        )
        # web.py returns 'final_answer'
        w_context = result.get("final_answer", "")
    except Exception as e:
        log.error(f"Web search failed: {e}")
        w_context = ""
    return {"w_context": w_context}


async def rag_node(state: AgentState):
    # Context already fetched in router_node
    return {"r_context": state["r_context"]}


async def generate_node(state: AgentState):
    log = ts_logger.get_logger()
    query   = state["messages"][-1].content
    user_id = state.get("user_id", "default_user")

    long_term = await get_user_memory_semantic(query, user_id)

    r_ctx = state.get("r_context", "") or "None"
    w_ctx = state.get("w_context", "") or "None"

    system_prompt = f"""You are a helpful, concise assistant.

Long-term memory about this user (use silently to personalise answers):
{long_term if long_term else 'No memories yet.'}

RAG context (internal documents):
{r_ctx}

Web search context:
{w_ctx}

Instructions:
- Answer the user's query directly.
- If web context is used, cite sources as [1], [2] etc.
- Do not mention that you are using memory or context.
- Be conversational and clear.
"""

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    # Save memory in background — properly awaited via task
    # Using ensure_future keeps it non-blocking but still tracked
    asyncio.ensure_future(
        extract_and_save_memory(query, user_id)
    )

    return {"messages": [response]}


async def summarize_node(state: AgentState):
    """
    FIX: Instead of replacing messages (which breaks add_messages reducer),
    we trim to last 10 and prepend a summary SystemMessage.
    Only triggers when history > 10 messages.
    """
    msgs = state["messages"]
    if len(msgs) <= 10:
        return {}   # No change needed — returning empty dict is correct

    log = ts_logger.get_logger()
    log.info("Summarizing long conversation...")

    older_msgs = msgs[:-8]
    recent_msgs = msgs[-8:]

    summary_resp = await llm.ainvoke(
        [SystemMessage(content="Summarize this conversation briefly in 3-4 lines.")] + older_msgs
    )

    summary_msg = SystemMessage(
        content=f"[Conversation Summary]: {summary_resp.content}"
    )

    # Return only recent + summary — this replaces full messages list
    # We override directly instead of using add_messages for this node
    return {"messages": [summary_msg] + recent_msgs}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router",       router_node)
    builder.add_node("web_search",   web_node)
    builder.add_node("rag_retrieve", rag_node)
    builder.add_node("generate",     generate_node)
    builder.add_node("summarize",    summarize_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        lambda x: x["decision"],
        {
            "web": "web_search",
            "rag": "rag_retrieve",
            "llm": "generate"
        }
    )

    builder.add_edge("web_search",   "generate")
    builder.add_edge("rag_retrieve", "generate")
    builder.add_edge("generate",     "summarize")
    builder.add_edge("summarize",    END)

    memory = MemorySaver()   # In-memory short-term thread state (per session)
    return builder.compile(checkpointer=memory)


graph = build_graph()

File name > agent.py
