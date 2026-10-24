import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from combined_search import combined_search

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "your_api_key_here":
    raise RuntimeError(
        "OpenAI API key is missing. Check the .env file in D:\\Spotify."
    )

client = OpenAI(api_key=api_key)


def generate_answer(question):
    # Retrieve relevant document passages and graph concepts.
    results, concepts = combined_search(question, top_k=5)

    if not results:
        return {
            "answer": "I could not find relevant information in the knowledge base.",
            "sources": [],
        }

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Chunk: {result['chunk_id']}\n"
            f"Text:\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a technical assistant answering questions about the supplied
Spotify-style web application architecture document.

Rules:
1. Answer using only the supplied document passages.
2. Do not invent technologies, services, or relationships.
3. If the passages do not contain enough information, say so clearly.
4. Distinguish facts stated in the document from reasonable inferences.
5. Keep answers clear and practical.
6. Include source page references such as [Page 2].
7. Do not claim the supplied architecture is Spotify's verified production
   architecture; treat it as the architecture described by the document.
"""

    user_prompt = f"""
Question:
{question}

Knowledge graph concepts matched:
{", ".join(concepts) if concepts else "No matching concepts"}

Retrieved document passages:
{context}

Provide a concise answer with supporting page references.
"""

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=system_prompt,
        input=user_prompt,
    )

    answer = response.output_text

    sources = [
        {
            "source": result["source"],
            "page": result["page"],
            "chunk_id": result["chunk_id"],
            "retrieval_score": result["score"],
        }
        for result in results
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


def main():
    print("Spotify Architecture — RAG Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            result = generate_answer(question)

            print("\nANSWER")
            print(result["answer"])

            print("\nSOURCES")
            for source in result["sources"]:
                print(
                    f"- {source['source']}, "
                    f"page {source['page']}, "
                    f"chunk {source['chunk_id']}"
                )

            print()

        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()