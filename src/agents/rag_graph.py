"""LangGraph state machine for the RAG customer support system."""

import warnings, logging, os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
for _l in ["chromadb","sentence_transformers","langchain","httpx","httpcore","openai","groq","transformers"]:
    logging.getLogger(_l).setLevel(logging.CRITICAL)

from typing import TypedDict, List, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()


def get_llm(cfg: dict = {}):
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(model=cfg.get("llm_model", "llama3-8b-8192"), temperature=0, groq_api_key=groq_key)
    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cfg.get("llm_model", "gpt-3.5-turbo"), temperature=0, openai_api_key=openai_key)
    else:
        raise ValueError("No API key! Add GROQ_API_KEY or OPENAI_API_KEY to .env")

def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  model_kwargs={"device": "cpu"},
                                  encode_kwargs={"normalize_embeddings": True})

def get_vectorstore(chroma_dir, collection_name):
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings(),
                  collection_name=collection_name)


class SupportState(TypedDict):
    query: str
    retrieved_docs: List[str]
    relevant_docs: List[str]
    answer: str
    confidence: float
    escalated: bool
    escalation_reason: str
    session_id: str
    messages: Annotated[List, operator.add]


def intent_classifier_node(state: SupportState, config: dict) -> SupportState:
    cfg = config.get("configurable", {})
    keywords = cfg.get("escalation_keywords", ["legal","lawsuit","sue","fraud","hacked","data breach"])
    q = state["query"].lower()
    for kw in keywords:
        if kw in q:
            return {**state, "escalated": True, "escalation_reason": f"Sensitive keyword: '{kw}'", "answer": ""}
    return {**state, "escalated": False, "escalation_reason": ""}


def retriever_node(state: SupportState, config: dict) -> SupportState:
    cfg = config.get("configurable", {})
    vs = get_vectorstore(cfg.get("chroma_dir", "./chroma_db"), cfg.get("collection_name", "customer_support_kb"))
    docs = vs.similarity_search(state["query"], k=cfg.get("top_k", 4))
    return {**state, "retrieved_docs": [d.page_content for d in docs]}


def grader_node(state: SupportState, config: dict) -> SupportState:
    from langchain_core.prompts import ChatPromptTemplate
    cfg = config.get("configurable", {})
    llm = get_llm(cfg)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a relevance grader. Reply ONLY 'YES' if the document answers the question, else 'NO'."),
        ("human", "Question: {question}\n\nDocument: {document}")
    ])
    relevant = []
    for doc in state["retrieved_docs"]:
        try:
            r = llm.invoke(prompt.format_messages(question=state["query"], document=doc))
            if r.content.strip().upper().startswith("YES"):
                relevant.append(doc)
        except Exception:
            continue
    return {**state, "relevant_docs": relevant}


def generator_node(state: SupportState, config: dict) -> SupportState:
    from langchain_core.prompts import ChatPromptTemplate
    cfg = config.get("configurable", {})
    llm = get_llm(cfg)
    context = "\n\n---\n\n".join(state["relevant_docs"])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a warm, helpful customer support assistant for ShopEasy.\n"
         "Answer using ONLY the context below. Do not use outside knowledge.\n"
         "If the answer isn't in the context, say: "
         "'I don't have specific information about that. Let me connect you with a specialist.'\n"
         "Be concise and friendly.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    try:
        r = llm.invoke(prompt.format_messages(context=context, question=state["query"]))
        answer = r.content.strip()
    except Exception:
        answer = "I don't have specific information about that. Let me connect you with a specialist."
    low = ["i don't have","let me connect","specialist","not mentioned","i'm not sure"]
    confidence = 0.35 if any(p in answer.lower() for p in low) else 0.88
    return {**state, "answer": answer, "confidence": confidence}


def hitl_node(state: SupportState, config: dict) -> SupportState:
    msg = ("Thank you for reaching out! Your question needs attention from one of our "
           "specialist agents. A team member will follow up within 24 hours. "
           "Is there anything else I can help you with?")
    return {**state, "answer": msg, "escalated": True, "messages": [AIMessage(content=msg)]}


def route_after_intent(state): return "hitl" if state.get("escalated") else "retriever"
def route_after_grader(state): return "hitl" if not state.get("relevant_docs") else "generator"
def route_after_generator(state, config):
    t = config.get("configurable", {}).get("confidence_threshold", 0.60)
    return "hitl" if state.get("confidence", 1.0) < t else "end"


def build_graph(config=None):
    g = StateGraph(SupportState)
    g.add_node("intent_classifier", intent_classifier_node)
    g.add_node("retriever", retriever_node)
    g.add_node("grader", grader_node)
    g.add_node("generator", generator_node)
    g.add_node("hitl", hitl_node)
    g.set_entry_point("intent_classifier")
    g.add_conditional_edges("intent_classifier", route_after_intent, {"retriever":"retriever","hitl":"hitl"})
    g.add_edge("retriever", "grader")
    g.add_conditional_edges("grader", route_after_grader, {"generator":"generator","hitl":"hitl"})
    g.add_conditional_edges("generator", route_after_generator, {"end":END,"hitl":"hitl"})
    g.add_edge("hitl", END)
    return g.compile()


def run_query(graph, query: str, session_id: str = "default") -> dict:
    from src.utils.config import Config
    cfg = Config()
    state: SupportState = {
        "query": query, "retrieved_docs": [], "relevant_docs": [],
        "answer": "", "confidence": 1.0, "escalated": False,
        "escalation_reason": "", "session_id": session_id,
        "messages": [HumanMessage(content=query)]
    }
    run_cfg = {"configurable": {
        "chroma_dir": cfg.CHROMA_DIR, "collection_name": cfg.COLLECTION_NAME,
        "top_k": cfg.TOP_K, "llm_model": cfg.LLM_MODEL,
        "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
        "escalation_keywords": cfg.ESCALATION_KEYWORDS,
    }}
    return graph.invoke(state, config=run_cfg)