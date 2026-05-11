# ── User info ────────────────────────────────────────────────────────────────
USER_NAME    = "Your Name"
USER_TITLE   = "Software Engineer"
USER_CONTACT = ""           # e.g. "email@example.com | linkedin.com/in/you"

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5:27b"
OLLAMA_HOST  = "http://localhost:11434"

# ── RAG ───────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"  # sentence-transformers model (auto-downloaded)
CHROMA_PATH  = ".chroma"
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 150
TOP_K        = 5
