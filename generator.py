import time

import ollama
import config

_SYSTEM = """\
You are an expert cover letter writer. Produce professional, specific, and compelling cover letters.

Rules:
- Write in first person from the applicant's perspective
- Be concrete — reference specific projects, technologies, or outcomes when the context supplies them
- Match the tone and requirements of the job description
- Three to four focused paragraphs; no filler phrases
- Lead with what the applicant brings to the role, not what they want from it
- End with a confident, brief closing"""


def generate(job_description: str, context_chunks: list[str], capture: bool = False) -> str | None:
    user_info = f"Applicant: {config.USER_NAME}"
    if config.USER_TITLE:
        user_info += f"\nTitle: {config.USER_TITLE}"
    if config.USER_CONTACT:
        user_info += f"\nContact: {config.USER_CONTACT}"

    context_block = ""
    if context_chunks:
        context_block = "\n\n## Relevant Project Experience\n\n" + "\n\n---\n\n".join(context_chunks)

    prompt = (
        f"{user_info}\n\n"
        f"## Job Description\n\n{job_description}"
        f"{context_block}\n\n"
        "Write a professional cover letter for this position."
    )

    stream = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.7},
        stream=True,
    )

    parts: list[str] = []
    start = time.time()
    last_chunk = None

    for chunk in stream:
        parts.append(chunk["message"]["content"])
        last_chunk = chunk
        elapsed = time.time() - start
        rate = len(parts) / elapsed if elapsed > 0 else 0
        print(
            f"\r  {len(parts)} tokens | {rate:.1f} tok/s | {elapsed:.0f}s elapsed   ",
            end="",
            flush=True,
        )

    # Use Ollama's reported counts if available (more accurate than chunk count)
    if last_chunk:
        eval_count = last_chunk.get("eval_count", len(parts))
        eval_duration_ns = last_chunk.get("eval_duration", 0)
        if eval_duration_ns > 0:
            final_rate = eval_count / (eval_duration_ns / 1e9)
        else:
            elapsed = time.time() - start
            final_rate = eval_count / elapsed if elapsed > 0 else 0
        total_elapsed = (last_chunk.get("total_duration", 0) or 0) / 1e9 or (time.time() - start)
        print(f"\r  Done: {eval_count} tokens in {total_elapsed:.1f}s ({final_rate:.1f} tok/s)        ")
    else:
        print()

    letter = "".join(parts)
    print(letter)

    return letter if capture else None
