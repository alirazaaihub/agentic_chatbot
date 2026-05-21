import os
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

search =DuckDuckGoSearchAPIWrapper(max_results=5)

class AgentState(TypedDict):
    question: str
    search_results: List[dict]
    final_answer: str

def search_node(state: AgentState):
    print("\n🔎 Searching web with structured results...\n")
    query = state["question"]
    results = search.results(query, max_results=5)
    return {"search_results": results}

def answer_node(state: AgentState):
    print("\n🤖 Generating answer with citations...\n")
    formatted_sources = ""
    source_links = []
    for i, result in enumerate(state["search_results"], start=1):
        formatted_sources += f"""
        Source {i}:
        Title: {result['title']}
        Snippet: {result['snippet']}
        URL: {result['link']}
        ------------------------
        """
        source_links.append(f"{i}. {result['link']}")
    prompt = f"""
    You are a professional research assistant.\n
    User Question:\n
    {state['question']}

    Below are web search sources:\n

    {formatted_sources}\n

    Instructions:
    - Analyze all sources carefully
    - Combine the information
    - Provide a clear structured answer
    - Mention source numbers like [1], [2] inside the answer where relevant
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    final_output = response.content + "\n\nSources:\n" + "\n".join(source_links)
    return {"final_answer": final_output}

def web_invoke():
    builder = StateGraph(AgentState)
    builder.add_node("search", search_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "search")
    builder.add_edge("search", "answer")
    builder.add_edge("answer", END)

    return builder.compile()
graph_web = web_invoke()



if __name__ == "__main__":
    question = input("Enter your question: ")

    result = graph_web.invoke({"question": question})

    print("\n Final Answer:\n")
    print(result["final_answer"])
