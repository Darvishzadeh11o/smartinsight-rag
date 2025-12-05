import os
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # not used here but handy later
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import VECTORSTORE_DIR, OPENAI_API_KEY


def load_vectorstore() -> FAISS:
    """Load the saved FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # needed for local FAISS load
    )
    return vectorstore


def answer_question(query: str, k: int = 4) -> Dict:
    """
    Retrieve relevant chunks for the query and ask the LLM to answer
    based ONLY on those chunks.
    Returns a dict with 'answer' and 'sources'.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 1) Retrieve relevant document chunks
    docs = retriever.invoke(query)


    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the document.",
            "sources": [],
        }

    # 2) Build context string with sources
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

    # 3) Create prompt for the LLM
    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the provided context. "
        "If the answer is not in the context, say you don't know based on the document."
    )

    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer in 3–5 sentences and, when possible, mention which sources you used, "
        "like [Source 1], [Source 2]."
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


if __name__ == "__main__":
    # Simple CLI to test the pipeline
    print("Ask a question about your PDF (sample.pdf). Type 'exit' to quit.")
    while True:
        user_q = input("\nYour question: ")
        if user_q.strip().lower() in {"exit", "quit"}:
            print("Bye!")
            break

        result = answer_question(user_q)
        print("\n=== Answer ===")
        print(result["answer"])

        print("\n=== Sources ===")
        if not result["sources"]:
            print("No sources found.")
        else:
            for s in result["sources"]:
                print(f"Source {s['source_id']} - page {s['page']} - metadata: {s['metadata']}")
