"""Ingest PDFs into ChromaDB vector store."""

import argparse
import os
from src.utils.document_loader import load_and_chunk_pdf
from src.utils.config import Config
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def ingest(pdf_path: str):
    """Ingest a PDF into the vector store."""
    config = Config()

    print(f"\n[Ingestion] Loading PDF: {pdf_path}")
    chunks = load_and_chunk_pdf(
        pdf_path,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    print(f"[Ingestion] Created {len(chunks)} chunks")

    print("[Ingestion] Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    print("[Ingestion] Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME
    )
    vectorstore.persist()

    print(f"[Ingestion] Done! {len(chunks)} chunks indexed in ChromaDB at '{config.CHROMA_DIR}'")
    print("[Ingestion] You can now run: python src/agents/rag_app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF into ChromaDB")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: File not found: {args.pdf}")
        exit(1)

    ingest(args.pdf)