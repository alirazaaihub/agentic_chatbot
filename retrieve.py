from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

vector_db = Chroma(
    persist_directory="vectore_db",
    embedding_function=embeddings
)


class State(TypedDict):
    query: str
    m_query: List[str]
    context: str
    answer: str
    final_answer: str
    iterations: int


def multiquery_node(state: State):

    prompt = f"""
You are a helpful assistant.

User Query:
{state['query']}

Generate 3 different variations of this query
for better retrieval from a vector database.

Return ONLY the queries as a numbered list.
"""

    response = llm.invoke(prompt)

    lines = [
        l.strip()
        for l in response.content.strip().split("\n")
        if l.strip()
    ]

    queries = [
        l.lstrip("1234567890.) ").strip()
        for l in lines[:3]
    ]

    return {
        "m_query": queries,
        "iterations": state.get("iterations", 0) + 1
    }


def retriever_node(state: State):

    queries = state["m_query"]

    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.3
        }
    )

    all_docs = []

    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # Remove duplicates
    seen = set()
    unique_docs = []

    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    context = "\n".join(
        doc.page_content for doc in unique_docs
    )

    return {
        "context": context
    }


def check_node(state: State):

    # Empty context protection
    if not state["context"].strip():
        return {"answer": "no"}

    prompt = f"""
You are a helpful assistant.

Context:
{state['context']}

Query:
{state['query']}

Does the context contain enough information
to answer the query?

Reply ONLY with:
yes
or
no
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content.strip().lower()
    }


def no_answer_node(state: State):

    return {
        "final_answer":
        "The relevant documents are not available in the vector database."
    }


def router_node(state: State):

    if state["answer"] == "yes":
        return END

    elif state.get("iterations", 0) >= 3:
        return "NoAnswerNode"

    else:
        return "MultiQueryNode"


def graph_builder():

    graph = StateGraph(State)

    # Nodes
    graph.add_node("MultiQueryNode", multiquery_node)
    graph.add_node("RetrieverNode", retriever_node)
    graph.add_node("CheckNode", check_node)
    graph.add_node("NoAnswerNode", no_answer_node)

    # Main flow
    graph.add_edge(START, "MultiQueryNode")
    graph.add_edge("MultiQueryNode", "RetrieverNode")
    graph.add_edge("RetrieverNode", "CheckNode")

    # Conditional routing
    graph.add_conditional_edges(
        "CheckNode",
        router_node,
        {
            "MultiQueryNode": "MultiQueryNode",
            "NoAnswerNode": "NoAnswerNode",
            END: END
        }
    )

    graph.add_edge("NoAnswerNode", END)

    return graph.compile()


graph_app = graph_builder()
