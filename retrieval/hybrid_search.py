import json
import re
from pathlib import Path
from uuid import uuid4

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "chunks" / "spotify_chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "spotify_hybrid"


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower())


def reciprocal_rank_fusion(rankings, k=60):
    """Combine ranked result lists using Reciprocal Rank Fusion."""
    fused_scores = {}

    for ranking in rankings:
        for rank, chunk_index in enumerate(ranking, start=1):
            fused_scores[chunk_index] = (
                fused_scores.get(chunk_index, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )


def main():
    if not CHUNKS_PATH.exists():
        print(f"ERROR: Chunks file not found: {CHUNKS_PATH}")
        return

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        print("ERROR: No chunks found.")
        return

    texts = [chunk["text"] for chunk in chunks]

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    # Set up the temporary vector database.
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(embeddings[0]),
            distance=models.Distance.COSINE
        )
    )

    points = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={"chunk_index": index}
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    # Build the BM25 index.
    tokenized_documents = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_documents)

    print(f"\nHybrid search ready: {len(chunks)} chunks indexed.")

    try:
        while True:
            question = input(
                "\nAsk a question (type 'exit' to stop): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            # 1. Semantic/vector search.
            query_vector = model.encode(
                question,
                normalize_embeddings=True
            ).tolist()

            vector_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=min(5, len(chunks)),
                with_payload=True
            ).points

            vector_ranking = [
                int(result.payload["chunk_index"])
                for result in vector_results
            ]

            # 2. BM25 keyword search.
            query_tokens = tokenize(question)
            bm25_scores = bm25.get_scores(query_tokens)

            bm25_ranking = sorted(
                range(len(chunks)),
                key=lambda index: bm25_scores[index],
                reverse=True
            )[:min(5, len(chunks))]

            # 3. Combine the two rankings.
            fused_ranking = reciprocal_rank_fusion(
                [vector_ranking, bm25_ranking]
            )

            print("\n===== HYBRID SEARCH RESULTS =====")

            for rank, index in enumerate(fused_ranking[:3], start=1):
                chunk = chunks[index]

                print(f"\nRank: {rank}")
                print(f"Chunk ID: {chunk['chunk_id']}")
                print(f"Source: {chunk['source']}")
                print(f"Page: {chunk['page']}")
                print(f"RRF score: {sum(1 / (60 + r + 1) for r, ranking in enumerate([])):.4f}"
                      if False else "")
                print(f"Passage:\n{chunk['text'][:700]}")

    finally:
        client.close()


if __name__ == "__main__":
    main()