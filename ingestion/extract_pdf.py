import json
from pathlib import Path

import pymupdf

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# PDF input
PDF_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "Spotify_web_app_architecture.pdf"
)

# Extracted text output
OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "spotify_architecture.json"
)


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    pages = []

    with pymupdf.open(PDF_PATH) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append({
                "page": page_number,
                "text": page.get_text("text").strip()
            })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            pages,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("PDF extraction successful!")
    print(f"Total pages: {len(pages)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()