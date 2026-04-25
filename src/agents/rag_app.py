"""RAG Customer Support Assistant application."""

import os
from dotenv import load_dotenv
from src.agents.rag_graph import build_graph, run_query
from src.utils.config import Config

load_dotenv()


def print_banner():
    """Display application banner."""
    print("\n" + "=" * 55)
    print("   RAG Customer Support Assistant")
    print("   Built with LangGraph + ChromaDB + HITL")
    print("=" * 55)
    print("Type your question, or 'quit' to exit.\n")


def main():
    """Main application loop."""
    print_banner()

    config = Config()
    graph = build_graph(config)

    session_id = "session_001"

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! Have a great day.")
            break

        print("\nAssistant: ", end="", flush=True)
        result = run_query(graph, user_input, session_id)
        print(result["answer"])

        if result.get("escalated"):
            print("\n[Note: This query has been escalated to a human agent.]")

        print()


if __name__ == "__main__":
    main()
