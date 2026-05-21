# 🤖 Agentic Chatbot — Smart Routing · RAG · Web Search · Long-Term Memory

A **production-grade agentic chatbot** built with LangGraph, FastAPI, and Streamlit. The agent intelligently routes every query to the most appropriate tool — internal document retrieval (RAG), live web search, or direct LLM response — while maintaining persistent long-term memory across all sessions.

---

## 📸 Architecture Overview

```
User Query
    │
    ▼
┌─────────────┐
│  Router Node │  ← Structured LLM decides: RAG | WEB | LLM
└──────┬──────┘
       │
  ┌────┴─────┬──────────┐
  ▼          ▼          ▼
RAG       Web Search   Direct
Retrieve  (DuckDuckGo) LLM
  │          │          │
  └────┬─────┘          │
       ▼                │
  ┌────────────┐        │
  │  Generate  │◄───────┘
  │   Node     │  (injects long-term memory + context)
  └─────┬──────┘
        ▼
  ┌───────────────┐
  │ Summarize Node│  (trims history > 10 msgs)
  └───────────────┘
        ▼
     Response (SSE streaming)
```

---

## 📸 Preview




<img width="1113" height="656" alt="Screenshot 2026-05-20 105006" src="https://github.com/user-attachments/assets/02a554bd-bd72-4885-bbb4-a8121dddb7da" />

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Smart Routing** | LLM-powered structured router selects RAG, Web, or LLM per query |
| **Multi-Query RAG** | Generates 3 query variations, retrieves with similarity threshold, deduplicates |
| **Live Web Search** | DuckDuckGo search with source citations [1], [2] in responses |
| **Long-Term Memory** | Extracts user facts → stores in Chroma (semantic) + SQLite (persistent) |
| **SSE Streaming** | Token-by-token streaming via Server-Sent Events, like ChatGPT |
| **Conversation History** | Full multi-turn history with SQLite persistence per user |
| **Conversation Summarization** | Auto-summarizes older messages when history exceeds 10 turns |
| **Source Badges** | UI shows whether response came from RAG, Web, or LLM |
| **ChatGPT-style UI** | Dark mode Streamlit frontend with sidebar conversation management |

---

## 🗂️ Project Structure

```
agentic-chatbot/
│
├── main.py              # FastAPI backend — SSE streaming + REST endpoints
├── agent.py             # LangGraph agent — routing, memory, generation nodes
├── retrieve.py          # RAG sub-graph — multi-query + Chroma retrieval
├── web.py               # Web search sub-graph — DuckDuckGo + citation answer
├── database.py          # SQLite layer — users, conversations, messages, memories
├── streamlit_app.py     # Streamlit frontend — ChatGPT-style streaming UI
│
├── memory_db/           # Chroma vector store — long-term user memories
├── vectore_db/          # Chroma vector store — your document embeddings
│
├── agent.log            # Auto-rotating daily log (7-day retention)
├── chatbot.db           # SQLite database (auto-created on first run)
│
├── .env                 # API keys (not committed)
└── requirements.txt     # Dependencies
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | Google Gemini — `gemini-embedding-001` |
| **Agent Framework** | LangGraph (StateGraph) |
| **Vector Store** | ChromaDB |
| **Web Search** | DuckDuckGo via LangChain |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Database** | SQLite (WAL mode) |
| **Short-Term Memory** | LangGraph MemorySaver (per-thread) |
| **Long-Term Memory** | Chroma (semantic) + SQLite (persistent) |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/alirazaaihub/agentic_chatbot.git
cd agentic-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Get your keys:
- **Groq API Key** → [console.groq.com](https://console.groq.com)
- **Google API Key** → [aistudio.google.com](https://aistudio.google.com)

### 5. Ingest Your Documents (RAG Setup)

Before running, populate the `vectore_db` Chroma store with your documents. Example:

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
Chroma.from_documents(chunks, embeddings, persist_directory="vectore_db")
```

### 6. Run the Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### 7. Run the Frontend

Open a new terminal:

```bash
streamlit run streamlit_app.py
```

Frontend available at: `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat/stream` | SSE streaming chat response |
| `POST` | `/chat` | Non-streaming chat (for history reload) |
| `POST` | `/conversations/new` | Create a new conversation |
| `GET` | `/conversations/{user_id}` | List all conversations for a user |
| `GET` | `/conversations/{conv_id}/messages` | Get all messages in a conversation |
| `DELETE` | `/conversations/{conv_id}` | Delete a conversation and its messages |
| `GET` | `/memories/{user_id}` | Get all long-term memories for a user |
| `GET` | `/` | Health check |

### Example: Streaming Chat Request

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id": "ali_raza", "query": "What is on our menu?"}'
```

### SSE Event Types

```json
{ "type": "conv_id",  "conv_id": "uuid" }
{ "type": "token",    "token": "Hello" }
{ "type": "done",     "source": "rag", "conv_id": "uuid" }
{ "type": "error",    "message": "..." }
```

---

## 🧠 How the Agent Works

### Routing Logic
The router uses a **structured LLM output** (`RouteDecision` Pydantic model) to classify every query into one of three paths:

- **`rag`** — Questions about internal/domain-specific documents
- **`web`** — News, current events, live information
- **`llm`** — General knowledge, math, coding, conversation

### RAG Pipeline (`retrieve.py`)
1. Generates **3 query variations** from the original query
2. Retrieves top-3 docs per variation from Chroma (similarity threshold: 0.3)
3. Deduplicates results by content
4. LLM checks if context is sufficient → retries up to 3 iterations if not
5. Falls back gracefully with a clear "not found" message

### Long-Term Memory
- After every response, a background task extracts **permanent user facts** (name, preferences, etc.)
- Facts are stored in:
  - **Chroma** (`memory_db/`) — for semantic similarity search at query time
  - **SQLite** (`memories` table) — for UI display, with `UNIQUE` constraint preventing duplicates
- Memory is injected silently into the system prompt — the user never sees it referenced directly

### Conversation Summarization
When message history exceeds 10 turns, the `summarize_node` compresses older messages into a 3-4 line summary and retains only the last 8 messages, preventing context window overflow.

---

## 🗄️ Database Schema

```sql
users          (user_id PK, created_at)
conversations  (conv_id PK, user_id FK, title, created_at, updated_at)
messages       (msg_id PK, conv_id FK, user_id FK, role, content, source, created_at)
memories       (mem_id PK, user_id FK, content UNIQUE, created_at)
```

SQLite is configured with **WAL (Write-Ahead Logging)** for concurrent read performance.

---

## 📦 Requirements

```txt
fastapi
uvicorn
streamlit
langchain
langchain-groq
langchain-google-genai
langchain-chroma
langchain-community
langgraph
chromadb
python-dotenv
pydantic
requests
duckduckgo-search
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq LLM API key |
| `GOOGLE_API_KEY` | ✅ | Google Gemini embeddings key |

---

## 📝 Logging

The agent uses a `TimedRotatingFileHandler` that:
- Writes to `agent.log`
- Rotates daily at midnight
- Retains last **7 days** of logs
- Format: `timestamp | level | function | message`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 👤 Author

**Ali** — AI/ML Engineer  
Building production-grade agentic AI systems.



---
📌 [LinkedIn](www.linkedin.com/in/alirazaaihub)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
