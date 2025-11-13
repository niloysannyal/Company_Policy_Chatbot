"""
build_index.py
------------------
Build a fresh Chroma vector database directly from `dataset/dataset.jsonl`.
- Extract question/answer pairs without intermediate artifacts
- Chunk answers with overlap for better recall
- Persist embeddings and metadata in a clean collection
"""

import json
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# Paths
DATASET_PATH = Path("./dataset/dataset.jsonl")
VECTOR_DB_DIR = Path("./vector_db")
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters
MAX_WORDS_PER_CHUNK = 140
CHUNK_OVERLAP = 30


def iter_message_pairs() -> Iterable[Tuple[str, str]]:
    """Yield (question, answer) tuples from the dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Provide the dataset or adjust DATASET_PATH."
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            for idx, entry in enumerate(messages):
                if entry.get("role") != "user":
                    continue
                question = entry.get("content", "").strip()
                if not question:
                    continue
                answer = ""
                for follow in messages[idx + 1 :]:
                    if follow.get("role") == "assistant":
                        answer = follow.get("content", "").strip()
                        break
                if answer:
                    yield question, answer


def chunk_text(
    text: str,
    max_words: int = MAX_WORDS_PER_CHUNK,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []

    stride = max(max_words - overlap, 1)
    chunks: List[str] = []
    for start in range(0, len(words), stride):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
    return chunks


def reset_collection(client: chromadb.PersistentClient, name: str) -> chromadb.Collection:
    """Drop any existing collection with the given name and return an empty one."""
    if any(col.name == name for col in client.list_collections()):
        client.delete_collection(name)
    return client.get_or_create_collection(name=name)


def main():
    print("📦 Loading question/answer pairs from dataset...")
    qa_pairs = list(iter_message_pairs())
    if not qa_pairs:
        raise RuntimeError("No question/answer pairs found in dataset.")

    embed_model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = reset_collection(client, "company_policies")

    all_texts: List[str] = []
    all_metadatas: List[dict] = []
    all_ids: List[str] = []

    for question, answer in tqdm(qa_pairs, desc="🔹 Chunking answers"):
        base_id = str(uuid.uuid4())
        chunks = chunk_text(answer)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{base_id}-{idx}"
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "doc_title": question,
                    "question": question,
                    "chunk_index": idx,
                }
            )
            all_ids.append(chunk_id)

    if not all_texts:
        raise RuntimeError("No chunks produced from dataset answers.")

    print(f"Total chunks to embed: {len(all_texts)}")
    print("⚙️ Generating embeddings...")
    embeddings = embed_model.encode(
        all_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    print("💾 Writing chunks to ChromaDB...")
    collection.add(
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )
    print(
        f"✅ Added {len(all_texts)} chunks to collection 'company_policies' at {VECTOR_DB_DIR}"
    )
    print("🎉 Index build complete!")


if __name__ == "__main__":
    main()
