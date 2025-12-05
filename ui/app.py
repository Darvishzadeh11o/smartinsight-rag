import os
import tempfile
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables (OPENAI_API_KEY from .env)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")


# --------- Helper functions --------- #

def build_vectorstore_from_pdf(uploaded_file) -> FAISS:
    """Save uploaded PDF to a temp file, load it, chunk it, and create a FAISS vectorstore."""
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def answer_question_with_vectorstore(
    vectorstore: FAISS,
    query: str,
    k: int = 4
) -> Dict:
    """Retrieve relevant chunks from the given vectorstore and ask the LLM."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "I couldn't find relevant information in this document.",
            "sources": [],
        }

    context_parts = []
    sources = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[Source {i}] (page {page}):\n{doc.page_content}")
        sources.append(
            {
                "source_id": i,
                "page": page,
                "metadata": doc.metadata,
            }
        )

    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the provided context. "
        "If the answer is not in the context, say you don't know based on this document."
    )

    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer in 3–5 sentences, and mention which sources you used, like [Source 1], [Source 2]."
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    answer_text = response.content

    return {
        "answer": answer_text,
        "sources": sources,
    }


# --------- Streamlit UI --------- #

st.set_page_config(page_title="SmartInsight PDF Chat", layout="wide")
st.title("📄💬 SmartInsight – Chat with your PDF")

st.write(
    "Upload a PDF, then ask questions about it. "
    "Each time you upload a new file, a new mini knowledge base is created from that document."
)

# Session state to store vectorstore and file name
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF and building vector store..."):
        st.session_state.vectorstore = build_vectorstore_from_pdf(uploaded_file)
        st.session_state.pdf_name = uploaded_file.name
    st.success(f"PDF processed: **{uploaded_file.name}**. You can now ask questions.")

# Question input
st.subheader("Ask a question about the uploaded PDF")

question = st.text_input("Your question", placeholder="e.g., What is the weighted mean?")

if st.button("Ask"):

    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = answer_question_with_vectorstore(st.session_state.vectorstore, question)

        st.markdown("### 🧠 Answer")
        st.write(result["answer"])

        st.markdown("### 📚 Sources")
        if not result["sources"]:
            st.write("No sources found.")
        else:
            for s in result["sources"]:
                st.write(f"- **Source {s['source_id']}** – page {s['page']}")
