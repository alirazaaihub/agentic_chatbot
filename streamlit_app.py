"""
streamlit_app.py — ChatGPT-style streaming frontend
Token by token display — bilkul ChatGPT jaisa
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Söhne', 'ui-sans-serif', system-ui, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background-color: #202123;
    border-right: 1px solid #3e3f4b;
}
section[data-testid="stSidebar"] * { color: #ececec !important; }

.main { background-color: #343541; }

.user-msg {
    background: #40414f;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    color: #ececec;
    font-size: 15px;
    line-height: 1.6;
}
.bot-msg {
    background: #444654;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    color: #ececec;
    font-size: 15px;
    line-height: 1.6;
}
.source-badge {
    display: inline-block;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 999px;
    margin-top: 6px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-rag  { background:#10a37f22; color:#10a37f; border:1px solid #10a37f55; }
.badge-web  { background:#1a8fd122; color:#1a8fd1; border:1px solid #1a8fd155; }
.badge-llm  { background:#7c3aed22; color:#a78bfa; border:1px solid #7c3aed55; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in {
    "user_id":       None,
    "conv_id":       None,
    "messages":      [],
    "conversations": [],
    "show_memories": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        return r.json() if r.ok else {}
    except:
        return {}

def api_delete(path):
    try:
        requests.delete(f"{API_URL}{path}", timeout=10)
    except:
        pass

def load_conversations():
    data = api_get(f"/conversations/{st.session_state.user_id}")
    st.session_state.conversations = data.get("conversations", [])

def load_messages(conv_id):
    data = api_get(f"/conversations/{conv_id}/messages")
    st.session_state.messages = data.get("messages", [])
    st.session_state.conv_id  = conv_id

def source_badge(source):
    s   = (source or "llm").lower()
    cls = f"badge-{s}" if s in ("rag","web","llm") else "badge-llm"
    return f'<span class="source-badge {cls}">{s.upper()}</span>'

# ---------------------------------------------------------------------------
# ⭐ Streaming call — yeh sabse important function hai
# ---------------------------------------------------------------------------

def stream_chat(query: str, user_id: str, conv_id: str | None):
    """
    SSE stream read karo aur Streamlit write_stream compatible
    generator return karo.
    Yielded values: plain text tokens
    Side effects: sets st.session_state.conv_id & source
    """
    payload = {"user_id": user_id, "conv_id": conv_id, "query": query}

    with requests.post(
        f"{API_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=120
    ) as resp:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

            if not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()
            if not data_str:
                continue

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "conv_id":
                # Naya conversation ID set karo
                st.session_state.conv_id = event["conv_id"]

            elif etype == "token":
                # ⭐ Yeh token user ko dikhao
                yield event["token"]

            elif etype == "done":
                # Source save karo — badge ke liye
                st.session_state._last_source = event.get("source", "llm")
                st.session_state.conv_id = event.get(
                    "conv_id", st.session_state.conv_id
                )

            elif etype == "error":
                yield f"\n\n❌ Error: {event.get('message', 'Unknown error')}"

# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------

if not st.session_state.user_id:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## 🤖 AI Assistant")
        st.markdown("Apna User ID enter karein")
        uid = st.text_input("User ID", placeholder="e.g. ali_raza",
                            label_visibility="collapsed")
        if st.button("Continue →", use_container_width=True, type="primary"):
            if uid.strip():
                st.session_state.user_id = uid.strip()
                load_conversations()
                st.rerun()
            else:
                st.error("Valid User ID daalen")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.user_id}")
    st.markdown("---")

    if st.button("✏️  New Chat", use_container_width=True):
        st.session_state.conv_id  = None
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Conversations**")

    load_conversations()
    for conv in st.session_state.conversations:
        col_a, col_b = st.columns([5, 1])
        with col_a:
            is_active = conv["conv_id"] == st.session_state.conv_id
            label = conv["title"][:28] + ("…" if len(conv["title"]) > 28 else "")
            if st.button(label, key=f"c_{conv['conv_id']}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                load_messages(conv["conv_id"])
                st.rerun()
        with col_b:
            if st.button("🗑", key=f"d_{conv['conv_id']}"):
                api_delete(f"/conversations/{conv['conv_id']}")
                if st.session_state.conv_id == conv["conv_id"]:
                    st.session_state.conv_id  = None
                    st.session_state.messages = []
                st.rerun()

    st.markdown("---")
    if st.button("🧠  Memories", use_container_width=True):
        st.session_state.show_memories = not st.session_state.show_memories

    if st.session_state.show_memories:
        mems = api_get(f"/memories/{st.session_state.user_id}").get("memories", [])
        if mems:
            for m in mems:
                st.markdown(
                    f"<small style='color:#aaa'>• {m['content']}</small>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<small style='color:#666'>Koi memory nahi abhi.</small>",
                        unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪  Logout", use_container_width=True):
        for k in ["user_id","conv_id","messages","conversations","show_memories"]:
            st.session_state[k] = None if k in ("user_id","conv_id") else (False if k == "show_memories" else [])
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.markdown(
    "<h4 style='color:#ececec;margin-bottom:0'>🤖 AI Assistant</h4>"
    "<small style='color:#888'>Smart routing · RAG · Web · LLM</small>",
    unsafe_allow_html=True
)
st.markdown("---")

# Purane messages render karo
if not st.session_state.messages:
    st.markdown(
        "<div style='text-align:center;color:#555;margin-top:80px'>"
        "<h3 style='color:#666'>Kya help kar sakta hun?</h3>"
        "<p>Documents search · Web browse · General knowledge</p>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            badge = source_badge(msg.get("source", "llm"))
            st.markdown(
                f'<div class="bot-msg">🤖 {msg["content"]}<br>{badge}</div>',
                unsafe_allow_html=True
            )

# ---------------------------------------------------------------------------
# Chat input — streaming response
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Message AI Assistant..."):

    # User message dikhao
    st.markdown(
        f'<div class="user-msg">👤 {prompt}</div>',
        unsafe_allow_html=True
    )

    # ⭐ Streamlit write_stream — token by token render karta hai
    with st.chat_message("assistant"):
        full_text = st.write_stream(
            stream_chat(
                query   = prompt,
                user_id = st.session_state.user_id,
                conv_id = st.session_state.conv_id
            )
        )

    # Response ko session mein save karo
    source = getattr(st.session_state, "_last_source", "llm")
    st.session_state.messages.append({"role": "user",      "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": full_text,
                                       "source": source})

    st.rerun()
