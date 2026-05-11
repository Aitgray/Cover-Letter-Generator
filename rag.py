import os
import hashlib

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import config


def _collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
    return client.get_or_create_collection("projects", embedding_function=ef)


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + config.CHUNK_SIZE])
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return chunks


def add_document(filepath: str) -> int:
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    filename = os.path.basename(filepath)
    prefix = hashlib.md5(filename.encode()).hexdigest()[:8]
    chunks = _chunk(text)

    col = _collection()

    existing = col.get(where={"source": filename})
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    col.add(
        ids=[f"{prefix}_c{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def query(text: str, k: int | None = None) -> list[str]:
    col = _collection()
    count = col.count()
    if count == 0:
        return []
    results = col.query(query_texts=[text], n_results=min(k or config.TOP_K, count))
    return results["documents"][0]


def list_documents() -> list[str]:
    col = _collection()
    data = col.get()
    if not data["metadatas"]:
        return []
    return sorted({m["source"] for m in data["metadatas"]})
