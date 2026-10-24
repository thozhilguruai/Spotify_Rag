import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "chunks"
    / "spotify_chunks.json"
)


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower())


def main():
    if not CHUNKS_PATH.exists():
        print(f"ERROR: Chunks file not found: {CHUNKS_PATH}")
        return

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        print("ERROR: No chunks found.")
        return

    documents = [chunk["text"] for chunk in chunks]
    tokenized_documents = [tokenize(text) for text in documents]

    bm25 = BM25Okapi(tokenized_documents)

    print(f"BM25 indexed {len(chunks)} chunks.")

    while True:
        question = input(
            "\nEnter a keyword search query "
            "(type 'exit' to stop): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        query_tokens = tokenize(question)

        if not query_tokens:
            print("Please enter a query containing searchable terms.")
            continue

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        print("\n--- BM25 Search Results ---")

        result_count = min(3, len(chunks))

        for index in ranked_indices[:result_count]:
            chunk = chunks[index]

            print(f"\nBM25 score: {scores[index]:.4f}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Passage:\n{chunk['text'][:700]}")


if __name__ == "__main__":
    main()