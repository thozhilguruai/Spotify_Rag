import re
from pathlib import Path

import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent
GRAPH_PATH = BASE_DIR / "data" / "processed" / "spotify_knowledge_graph.graphml"

# Searchable concepts and their related keywords
CONCEPT_KEYWORDS = {
    "Frontend": ["frontend", "next.js", "react", "tailwind", "user interface"],
    "Backend": ["backend", "node.js", "nestjs", "api", "server"],
    "Database": ["database", "postgresql", "redis", "storage data"],
    "Search": ["search", "elasticsearch", "opensearch", "index"],
    "Authentication": ["authentication", "oauth2", "oidc", "jwt", "login"],
    "Storage": ["storage", "s3", "cdn", "files"],
    "Infrastructure": ["infrastructure", "docker", "kubernetes", "microservices"],
    "Messaging": ["messaging", "kafka", "rabbitmq", "nats", "message bus"],
}


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found: {GRAPH_PATH}\n"
            "Run: python knowledge_graph/build_graph.py"
        )

    return nx.read_graphml(GRAPH_PATH)


def find_matching_concepts(query, graph):
    query_lower = query.lower()
    matches = []

    for concept in graph.nodes:
        keywords = CONCEPT_KEYWORDS.get(concept, [concept.lower()])

        if any(keyword in query_lower for keyword in keywords):
            matches.append(concept)

    # Also match a concept when the query mentions one of its keywords.
    for concept in graph.nodes:
        keywords = CONCEPT_KEYWORDS.get(concept, [concept.lower()])

        if concept not in matches and any(
            keyword in query_lower for keyword in keywords
        ):
            matches.append(concept)

    # If no exact keyword matches, try matching concept names.
    if not matches:
        for concept in graph.nodes:
            if concept.lower() in query_lower:
                matches.append(concept)

    return matches


def search_graph(query, max_results=5):
    graph = load_graph()
    matches = find_matching_concepts(query, graph)

    if not matches:
        return []

    results = []

    for concept in matches:
        results.append({
            "concept": concept,
            "relationship": "MatchedQuery",
            "related_concept": None,
        })

        # Return directly connected concepts.
        for neighbour in graph.neighbors(concept):
            results.append({
                "concept": concept,
                "relationship": graph[concept][neighbour].get(
                    "relationship", "RelatedTo"
                ),
                "related_concept": neighbour,
            })

    # Remove duplicate results while preserving order.
    unique_results = []
    seen = set()

    for result in results:
        key = (
            result["concept"],
            result["relationship"],
            result["related_concept"],
        )

        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    return unique_results[:max_results]


def main():
    print("Spotify Architecture — Knowledge Graph Search")
    print("Enter a question or type 'exit' to stop.\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() == "exit":
            print("Graph search stopped.")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            results = search_graph(query)

            if not results:
                print("\nNo matching concepts found in the graph.")
                print("Try terms such as frontend, backend, database, or authentication.\n")
                continue

            print("\nKnowledge Graph Results:")

            for index, result in enumerate(results, start=1):
                print(f"\n{index}. Concept: {result['concept']}")
                print(f"   Relationship: {result['relationship']}")

                if result["related_concept"]:
                    print(f"   Related concept: {result['related_concept']}")

            print()

        except (FileNotFoundError, nx.NetworkXError) as error:
            print(f"\nError: {error}\n")
            break


if __name__ == "__main__":
    main()