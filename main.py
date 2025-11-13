"""
main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chatbot import ask
import uvicorn

# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(
    title="RAG Company Policy Chatbot API",
    version="1.0",
    description="A lightweight FastAPI backend exposing the hybrid RAG chatbot."
)


# ✅ ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "🤖 RAG Chatbot API is running.",
        "usage": "POST your question to /ask",
        "example": {"question": "What is the company leave policy?"}
    }


@app.post("/ask", response_model=AskResponse)
def ask_api(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ✅ Use your existing ask() function
    answer = ask(question)
    return AskResponse(answer=answer)


# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
