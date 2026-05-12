# Cover Letter Generator

Generates tailored cover letters using a locally-running LLM via Ollama. Uses RAG to pull relevant context from your own project reports, so the output references your actual work rather than generic filler.

## Prerequisites

- [Ollama](https://ollama.com) running locally
- Your preferred model pulled, e.g. `ollama pull qwen3.5:27b`
- Python 3.11+

## Installation

```bash
pip install -r requirements.txt
```

The sentence-transformers embedding model (~90 MB) downloads automatically on first use.

## Configuration

Edit `config.py` before first use:

```python
USER_NAME    = "Jane Smith"
USER_TITLE   = "Software Engineer"
USER_CONTACT = "jane@example.com | linkedin.com/in/janesmith"

OLLAMA_MODEL = "qwen2.5:27b"   # must match a model you've pulled in Ollama
```

## Usage

### Add project reports to the knowledge base

```bash
python main.py add documents/my-project.md
python main.py add documents/final-report.pdf   # PDFs supported
```

Reports only need to be added once — they persist in the local `.chroma/` vector DB.

```bash
python main.py list   # see what's in the knowledge base
```

### Generate a cover letter

```bash
# Job description as a string
python main.py generate --job "We're looking for a backend engineer..."

# Or from a file
python main.py generate --file job.txt

# Save the output to a file
python main.py generate --file job.txt --output cover_letter.txt
```

While generating, a live progress line shows token count, speed, and elapsed time so you can tell at a glance that it hasn't frozen:

```
  127 tokens | 11.8 tok/s | 10s elapsed
```

## How RAG works

When you run `generate`, the job description is embedded and compared against all chunks in the vector DB. The most relevant excerpts from your project reports are injected into the prompt as context, allowing the model to reference specific work you've done that's relevant to the role.

> **Note:** Models with built-in reasoning (e.g. Qwen 3) generate internal thinking tokens before producing the letter. This is expected and results in higher token counts and generation times.
