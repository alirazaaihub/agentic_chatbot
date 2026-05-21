"""
main.py — Streaming FastAPI backend
/chat/stream  → Server-Sent Events (SSE) streaming endpoint
/chat         → Normal endpoint (history load ke liye rakha)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import uvicorn
import json
import asyncio

from agent import graph, extract_and_save_memory
from langchain_core.messages import HumanMessage
from database import (
    init_db, ensure_user,
    create_conversation, get_user_conversations,
    get_conversation_messages, delete_conversation,
    save_message, get_user_memories, update_conv_title
)

app = FastAPI(title="Agentic Chatbot API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    conv_id: Optional[str] = None
    query:   str

class NewConvRequest(BaseModel):
    user_id: str
    title:   Optional[str] = "New Chat"

# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------

async def stream_response(req: ChatRequest) -> AsyncGenerator[str, None]:
    """
    LangGraph se token by token stream karo.
    SSE format: "data: {...}\n\n"
    """
    try:
        ensure_user(req.user_id)
        conv_id = req.conv_id or create_conversation(req.user_id, req.query[:50])

        # User message save karo
        save_message(conv_id, req.user_id, "user", req.query)

        # Conv ID client ko bhejo pehle — zaruri hai naye conversation ke liye
        yield f"data: {json.dumps({'type': 'conv_id', 'conv_id': conv_id})}\n\n"

        config = {"configurable": {"thread_id": conv_id}}

        full_response = ""
        source        = "llm"

        # astream_events v2 — har token pe event milta hai
        async for event in graph.astream_events(
            {
                "messages": [HumanMessage(content=req.query)],
                "user_id":  req.user_id,
            },
            config=config,
            version="v2"
        ):
            event_name = event.get("event")
            event_data = event.get("data", {})

            # Sirf generate node ke tokens chahiye
            if event_name == "on_chat_model_stream":
                # Node name check — sirf generate node ke tokens
                tags = event.get("tags", [])
                chunk = event_data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    full_response += token
                    # Token client ko bhejo
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Source track karo router se
            elif event_name == "on_chain_end":
                output = event_data.get("output", {})
                if isinstance(output, dict) and "source" in output:
                    source = output.get("source", "llm")

        # Response complete — final metadata bhejo
        update_conv_title(conv_id, req.query[:60])
        save_message(conv_id, req.user_id, "assistant", full_response, source)

        yield f"data: {json.dumps({'type': 'done', 'source': source, 'conv_id': conv_id})}\n\n"

        # Memory background mein save karo
        asyncio.ensure_future(extract_and_save_memory(req.query, req.user_id))

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_response(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",   # Nginx buffering band karo
        }
    )


# ---------------------------------------------------------------------------
# Non-streaming endpoints (history, conversations, memories)
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat_normal(req: ChatRequest):
    """History reload ke liye — streaming nahi chahiye wahan."""
    try:
        ensure_user(req.user_id)
        conv_id = req.conv_id or create_conversation(req.user_id, req.query[:50])
        save_message(conv_id, req.user_id, "user", req.query)

        config = {"configurable": {"thread_id": conv_id}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=req.query)], "user_id": req.user_id},
            config=config
        )

        response_text = result["messages"][-1].content
        source        = result.get("source", "llm")

        save_message(conv_id, req.user_id, "assistant", response_text, source)
        update_conv_title(conv_id, req.query[:60])
        asyncio.ensure_future(extract_and_save_memory(req.query, req.user_id))

        return {"user_id": req.user_id, "conv_id": conv_id,
                "response": response_text, "source": source}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations/new")
def new_conversation(req: NewConvRequest):
    ensure_user(req.user_id)
    conv_id = create_conversation(req.user_id, req.title)
    return {"conv_id": conv_id, "user_id": req.user_id, "title": req.title}

@app.get("/conversations/{user_id}")
def list_conversations(user_id: str):
    return {"conversations": get_user_conversations(user_id)}

@app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str):
    return {"messages": get_conversation_messages(conv_id)}

@app.delete("/conversations/{conv_id}")
def del_conversation(conv_id: str):
    delete_conversation(conv_id)
    return {"deleted": conv_id}

@app.get("/memories/{user_id}")
def memories(user_id: str):
    return {"memories": get_user_memories(user_id)}

@app.get("/")
def health():
    return {"status": "running", "version": "3.0", "streaming": True}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
