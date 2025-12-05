import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import (
    DATA_RAW_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENAI_API_KEY,
)

def ingest_pdf(file_path):
    print("Loading PDF...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    print("Building FAISS vector database...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    faiss_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"FAISS index saved at: {VECTORSTORE_DIR}")
    print("Ingestion complete!")

if __name__ == "__main__":
    # Example usage
    sample_pdf = os.path.join(DATA_RAW_DIR, "sample.pdf")
    if os.path.exists(sample_pdf):
        ingest_pdf(sample_pdf)
    else:
        print("Place a PDF file at data/raw/sample.pdf first.")
