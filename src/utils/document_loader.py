"""Document loading and chunking utilities for PDF processing."""

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Load a PDF and split it into overlapping chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Target size (in characters) for each chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        List of Document objects, each with page_content and metadata.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
        length_function=len
    )

    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = pdf_path

    return chunks
