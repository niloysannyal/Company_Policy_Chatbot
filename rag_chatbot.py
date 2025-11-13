"""
rag_chatbot.py
------------------
Streamlined RAG chatbot that uses Chroma + SentenceTransformer with MMR reranking.
"""

import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# Optional Gemini model
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Paths / constants
VECTOR_DB_DIR = "./vector_db"
COLLECTION_NAME = "company_policies"

# Models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_MEMORY_TURNS = 5

embed_model = SentenceTransformer(EMBED_MODEL)

# Load vector store
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

conversation_history: List[Tuple[str, str]] = []


def clean_text(text: str) -> str:
    """Normalize whitespace and strip stray brackets."""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.replace(" .", ".").replace(" ,", ",").strip()


def dense_retrieve(query: str, top_k: int = 10) -> Tuple[np.ndarray, List[Dict]]:
    """Retrieve top_k candidates from Chroma and return the query embedding."""
    query_embedding = embed_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    candidates = []
    for idx, doc in enumerate(docs):
        candidates.append(
            {
                "id": ids[idx],
                "text": doc,
                "metadata": metas[idx] or {},
                "distance": distances[idx],
            }
        )
    return query_embedding, candidates


def mmr_rerank(
    query_embedding: np.ndarray,
    candidates: List[Dict],
    top_n: int = 5,
    diversity: float = 0.3,
) -> List[Dict]:
    """Apply Max Marginal Relevance reranking using only embedding similarities."""
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    doc_embeddings = embed_model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarities = doc_embeddings @ query_embedding
    selected_indices: List[int] = []
    reranked: List[Dict] = []

    for _ in range(min(top_n, len(candidates))):
        if not selected_indices:
            next_index = int(np.argmax(similarities))
        else:
            redundancy = doc_embeddings @ doc_embeddings[selected_indices].T
            if redundancy.ndim == 1:
                redundancy = redundancy[:, None]
            penalty = redundancy.max(axis=1)
            mmr_scores = similarities - diversity * penalty
            mmr_scores[selected_indices] = -np.inf
            next_index = int(np.argmax(mmr_scores))

        if next_index in selected_indices:
            break

        selected_indices.append(next_index)
        doc = dict(candidates[next_index])
        doc["score"] = float(similarities[next_index])
        reranked.append(doc)

    return reranked


def format_context_snippets(docs: List[Dict]) -> str:
    """Format retrieved docs for prompting."""
    chunks = []
    for doc in docs:
        title = doc["metadata"].get("doc_title") or doc["metadata"].get("question") or "Source"
        snippet = clean_text(doc["text"])
        chunks.append(f"[{title}] {snippet}")
    return "\n\n".join(chunks)


def synthesize_local_answer(question: str, docs: List[Dict], max_sentences: int = 3) -> str:
    """Heuristic summarizer when Gemini is unavailable."""
    if not docs:
        return "I don't have information about that."

    pattern = re.compile(r"\w+")
    query_terms = {w.lower() for w in pattern.findall(question) if len(w) > 2}
    candidates: List[Tuple[float, str, str]] = []

    for rank, doc in enumerate(docs):
        title = doc["metadata"].get("doc_title") or doc["metadata"].get("question") or "Source"
        sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
        for sentence in sentences:
            normalized = clean_text(sentence)
            if not normalized:
                continue
            tokens = {w.lower() for w in pattern.findall(normalized)}
            if not tokens:
                continue
            overlap = len(tokens & query_terms)
            if overlap == 0 and rank > 0:
                continue
            score = overlap / max(len(tokens), 1) + 0.05 * (1.0 - rank / max(len(docs), 1))
            candidates.append((score, normalized, title))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected_sentences: List[Tuple[str, str]] = []
    used_sentences = set()
    for score, sentence, title in candidates:
        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)
        selected_sentences.append((sentence, title))
        if len(selected_sentences) >= max_sentences:
            break

    if not selected_sentences and docs:
        selected_sentences.append(
            (clean_text(docs[0]["text"]), docs[0]["metadata"].get("doc_title", "Source"))
        )

    body = " ".join(sentence for sentence, _ in selected_sentences).strip()
    if not body:
        return "I don't have information about that."

    titles: List[str] = []
    for _, title in selected_sentences:
        if title and title not in titles:
            titles.append(title)
    if titles:
        body += "\n\nSources: " + "; ".join(f"[{title}]" for title in titles)
    return body


def create_prompt(query: str, context: str, memory: str = "") -> str:
    return f"""You are a helpful assistant answering questions about company policies.
Use only the provided context. Do not hallucinate or invent details.

Conversation History:
{memory}

User Question:
{query}

Context:
{context}

If the answer is not found, reply exactly with: I don't have information about that.
Cite document titles inline like [doc_title].
"""


def generate_with_gemini(prompt: str) -> str:
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        raise RuntimeError("Gemini not configured")
    response = gemini_model.generate_content(prompt)
    return getattr(response, "text", str(response))


def get_conversation_history() -> List[Tuple[str, str]]:
    """Return a shallow copy of the stored conversation history."""
    return list(conversation_history)


def set_conversation_history(history: Sequence[Tuple[str, str]]) -> None:
    """Replace the in-memory conversation with the provided sequence."""
    conversation_history.clear()
    for pair in history:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        question, answer = pair
        conversation_history.append((str(question), str(answer)))


def ask(question: str) -> str:
    question = question.strip()
    if not question:
        return "I don't have information about that."

    query_embedding, candidates = dense_retrieve(question, top_k=12)
    if not candidates:
        print("No candidates found.")
        return "I don't have information about that."

    top_docs = mmr_rerank(query_embedding, candidates, top_n=5)
    context_snippets = format_context_snippets(top_docs)

    memory_text = ""
    for idx, (prev_q, prev_a) in enumerate(conversation_history[-MAX_MEMORY_TURNS:], start=1):
        memory_text += f"PrevQ{idx}: {prev_q}\nPrevA{idx}: {prev_a}\n"

    prompt = create_prompt(question, context_snippets, memory_text)

    try:
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            answer = clean_text(generate_with_gemini(prompt))
        else:
            answer = synthesize_local_answer(question, top_docs)
    except Exception as exc:
        answer = f"I don't have information about that. (Error: {exc})"

    answer = clean_text(answer)
    conversation_history.append((question, answer))

    print("\n🔹 Question:", question)
    print("\n💬 Answer:\n", answer)

    return answer


if __name__ == "__main__":
    print("🤖 Company Policy Chatbot (RAG + Chroma)")
    print("Type 'exit' to quit.")
    while True:
        q = input("\nAsk: ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue
        ask(q)
