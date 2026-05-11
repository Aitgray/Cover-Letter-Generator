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
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        if capture:
            parts.append(token)

    print()
    return "".join(parts) if capture else None
