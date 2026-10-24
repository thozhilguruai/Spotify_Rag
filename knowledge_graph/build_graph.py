import json
from pathlib import Path
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "chunks" / "spotify_chunks.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "spotify_knowledge_graph.graphml"


def main():
    if not CHUNKS_PATH.exists():
        print(f"ERROR: Chunks file not found: {CHUNKS_PATH}")
        return

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    graph = nx.Graph()

    # Architecture concepts to identify in the document
    concepts = {
        "Frontend": ["frontend", "next.js", "react", "tailwind css"],
        "Backend": ["backend", "node.js", "nestjs", "api gateway"],
        "Database": ["postgresql", "database", "redis"],
        "Search": ["elasticsearch", "opensearch", "search service"],
        "Authentication": ["oauth2", "oidc", "authentication", "jwt"],
        "Storage": ["s3", "object storage", "cdn"],
        "Infrastructure": ["docker", "kubernetes", "microservices"],
        "Messaging": ["kafka", "rabbitmq", "nats", "message bus"],
    }

    # Create concept nodes and connect concepts mentioned in the same chunk.
    for chunk in chunks:
        text = str(chunk.get("text", chunk.get("content", ""))).lower()

        if not text.strip():
            continue

        found = []

        for concept, keywords in concepts.items():
            if any(keyword in text for keyword in keywords):
                graph.add_node(concept, type="ArchitectureConcept")
                found.append(concept)

        for i, source in enumerate(found):
            for target in found[i + 1:]:
                graph.add_edge(
                    source,
                    target,
                    relationship="MentionedTogether"
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, OUTPUT_PATH)

    print("Knowledge graph created successfully!")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Relationships: {graph.number_of_edges()}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nDiscovered concepts:")
    for node in sorted(graph.nodes):
        print(f"- {node}")


if __name__ == "__main__":
    main()