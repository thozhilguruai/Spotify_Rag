import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "spotify_architecture.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "chunks"
    / "spotify_chunks.json"
)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300


def create_chunks(pages):
    chunks = []
    chunk_number = 1

    for page_data in pages:
        text = page_data["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_number:04d}",
                    "source": "Spotify_web_app_architecture.pdf",
                    "page": page_data["page"],
                    "text": chunk_text
                })

                chunk_number += 1

            if end == len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        return

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    chunks = create_chunks(pages)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Chunking completed successfully!")
    print(f"Pages processed: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()