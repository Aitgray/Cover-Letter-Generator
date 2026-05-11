import argparse
import os
import sys

import rag
import generator
import config


def cmd_generate(args: argparse.Namespace) -> None:
    if args.file:
        if not os.path.exists(args.file):
            sys.exit(f"File not found: {args.file}")
        with open(args.file, encoding="utf-8") as f:
            job_description = f.read()
    else:
        job_description = args.job

    if not job_description.strip():
        sys.exit("Error: job description is empty.")

    print("Retrieving relevant project experience...")
    context = rag.query(job_description)

    if context:
        print(f"Found {len(context)} relevant chunk(s) from your project documents.\n")
    else:
        print("No project documents in knowledge base — generating without RAG context.\n")

    print(f"Generating with {config.OLLAMA_MODEL}...\n")
    print("=" * 60)
    letter = generator.generate(job_description, context, capture=args.output is not None)
    print("=" * 60)

    if args.output and letter:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(letter)
        print(f"\nSaved to {args.output}")


def cmd_add(args: argparse.Namespace) -> None:
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        n = rag.add_document(filepath)
        print(f"Added '{os.path.basename(filepath)}' ({n} chunk{'s' if n != 1 else ''})")


def cmd_list(_args: argparse.Namespace) -> None:
    docs = rag.list_documents()
    if not docs:
        print("Knowledge base is empty. Use 'add' to add project reports.")
    else:
        print(f"{len(docs)} document(s) in the knowledge base:")
        for doc in docs:
            print(f"  - {doc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cover letters from a job description using a local LLM + RAG."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", aliases=["gen"], help="Generate a cover letter")
    src = gen.add_mutually_exclusive_group(required=True)
    src.add_argument("--job", "-j", metavar="TEXT", help="Job description as a string")
    src.add_argument("--file", "-f", metavar="PATH", help="Path to a file containing the job description")
    gen.add_argument("--output", "-o", metavar="PATH", help="Also save the letter to this file")

    add = sub.add_parser("add", help="Add project report(s) to the knowledge base")
    add.add_argument("files", nargs="+", metavar="FILE", help="Path(s) to project report files")

    sub.add_parser("list", help="List documents currently in the knowledge base")

    args = parser.parse_args()

    if args.command in ("generate", "gen"):
        cmd_generate(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
