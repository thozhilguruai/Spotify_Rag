import json
from pathlib import Path
from uuid import uuid4

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "chunks"
    / "spotify_chunks.json"
)

COLLECTION_NAME = "spotify_architecture"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not CHUNKS_PATH.exists():
        print(f"ERROR: Chunks file not found: {CHUNKS_PATH}")
        return

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        print("ERROR: No chunks found.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    # Temporary in-memory vector database.
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(embeddings[0]),
            distance=models.Distance.COSINE
        )
    )

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Successfully indexed {len(points)} chunks!")

    try:
        while True:
            question = input(
                "\nAsk a question about the architecture "
                "(type 'exit' to stop): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            query_vector = model.encode(
                question,
                normalize_embeddings=True
            ).tolist()

            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=min(3, len(chunks)),
                with_payload=True
            ).points

            print("\n--- Search Results ---")

            for result in results:
                payload = result.payload

                print(f"\nSimilarity score: {result.score:.4f}")
                print(f"Source: {payload['source']}")
                print(f"Page: {payload['page']}")
                print(f"Passage:\n{payload['text'][:700]}")

    finally:
        client.close()


if __name__ == "__main__":
    main()