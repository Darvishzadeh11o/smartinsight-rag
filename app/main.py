from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_pipeline import answer_question


app = FastAPI(
    title="SmartInsight RAG API",
    description="Ask questions about your ingested PDFs using Retrieval-Augmented Generation.",
    version="0.1.0",
)


class QuestionRequest(BaseModel):
    question: str
    k: int | None = 4  # how many chunks to retrieve


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: QuestionRequest):
    """
    Ask a question about the ingested documents.

    Body:
    {
      "question": "Your question here",
      "k": 4
    }
    """
    result = answer_question(req.question, k=req.k or 4)
    return result
