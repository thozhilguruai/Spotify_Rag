import json
import re
from pathlib import Path

import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "chunks" / "spotify_chunks.json"
GRAPH_PATH = BASE_DIR / "data" / "processed" / "spotify_knowledge_graph.graphml"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CONCEPT_KEYWORDS = {
    "Frontend": ["frontend", "next.js", "react", "tailwind", "user interface"],
    "Backend": ["backend", "node.js", "nestjs", "api", "server"],
    "Database": ["database", "postgresql", "redis"],
    "Search": ["search", "elasticsearch", "opensearch", "index"],
    "Authentication": ["authentication", "oauth2", "oidc", "jwt", "login"],
    "Storage": ["storage", "s3", "cdn", "files"],
    "Infrastructure": ["infrastructure", "docker", "kubernetes", "microservices"],
    "Messaging": ["messaging", "kafka", "rabbitmq", "nats", "message bus"],
}


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9.+#_-]+\b", text.lower())


def load_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("The chunks file is empty.")

    return chunks


def get_chunk_text(chunk):
    return str(chunk.get("text", chunk.get("content", "")))


def get_chunk_id(chunk, index):
    return chunk.get("chunk_id", chunk.get("id", f"chunk_{index + 1:04d}"))


def get_source(chunk):
    return chunk.get("source", "Spotify_web_app_architecture.pdf")


def get_page(chunk):
    return chunk.get("page", chunk.get("page_number", "Unknown"))


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {GRAPH_PATH}")

    return nx.read_graphml(GRAPH_PATH)


def combined_search(query, top_k=5):
    chunks = load_chunks()
    graph = load_graph()

    texts = [get_chunk_text(chunk) for chunk in chunks]
    tokenized_corpus = [tokenize(text) for text in texts]
    query_tokens = tokenize(query)

    # 1. BM25 keyword search
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query_tokens)

    # 2. Vector similarity search
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)
    vector_scores = cosine_similarity(query_embedding, embeddings)[0]

    # 3. Knowledge graph concept matching
    query_lower = query.lower()
    matched_concepts = []

    for concept in graph.nodes:
        keywords = CONCEPT_KEYWORDS.get(concept, [concept.lower()])
        if any(keyword in query_lower for keyword in keywords):
            matched_concepts.append(concept)

    graph_related_concepts = set(matched_concepts)

    for concept in matched_concepts:
        graph_related_concepts.update(graph.neighbors(concept))

    # Combine normalized BM25 and vector scores.
    def normalize(scores):
        maximum = max(scores) if len(scores) else 0
        if maximum <= 0:
            return [0.0] * len(scores)
        return [float(score / maximum) for score in scores]

    normalized_bm25 = normalize(bm25_scores)
    normalized_vector = normalize(vector_scores)

    results = []

    for index, chunk in enumerate(chunks):
        text_lower = texts[index].lower()

        graph_score = 0.0
        for concept in graph_related_concepts:
            keywords = CONCEPT_KEYWORDS.get(concept, [concept.lower()])
            if any(keyword in text_lower for keyword in keywords):
                graph_score = 1.0
                break

        final_score = (
            0.4 * normalized_vector[index]
            + 0.4 * normalized_bm25[index]
            + 0.2 * graph_score
        )

        results.append({
            "chunk_id": get_chunk_id(chunk, index),
            "source": get_source(chunk),
            "page": get_page(chunk),
            "score": round(final_score, 4),
            "vector_score": round(float(vector_scores[index]), 4),
            "bm25_score": round(float(bm25_scores[index]), 4),
            "graph_score": round(graph_score, 4),
            "text": texts[index],
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k], matched_concepts


def main():
    print("Spotify Architecture — Combined Retrieval")
    print("Combines vector, BM25, and knowledge graph search.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.\n")
            continue

        try:
            results, concepts = combined_search(query)

            print("\nMatched graph concepts:", concepts or "None")

            for rank, result in enumerate(results, start=1):
                print(f"\n--- Result {rank} ---")
                print(f"Chunk: {result['chunk_id']}")
                print(f"Source: {result['source']}")
                print(f"Page: {result['page']}")
                print(f"Combined score: {result['score']}")
                print(f"Vector score: {result['vector_score']}")
                print(f"BM25 score: {result['bm25_score']}")
                print(f"Graph score: {result['graph_score']}")
                print(f"Passage:\n{result['text']}")

            print()

        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"\nError: {error}\n")
            break


if __name__ == "__main__":
    main()