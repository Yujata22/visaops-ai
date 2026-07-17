from services.knowledge_retriever import (
    KnowledgeBaseError,
    VisaKnowledgeRetriever,
)


def main() -> None:
    print("VisaOPS Knowledge Search")
    print("Type 'exit' to stop.")
    print()

    try:
        retriever = VisaKnowledgeRetriever()
    except KnowledgeBaseError as error:
        print(f"Knowledge-base error: {error}")
        return

    while True:
        question = input("Ask a visa question: ").strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Search closed.")
            break

        if not question:
            print("Please enter a question.")
            continue

        matches = retriever.search(
            query=question,
            number_of_results=3,
        )

        if not matches:
            print("No relevant documents were found.")
            continue

        print()
        print("Top matches")
        print("=" * 70)

        for result_number, match in enumerate(
            matches,
            start=1,
        ):
            print(
                f"\nResult {result_number}"
            )

            print(
                f"Source: {match['source']}"
            )

            print(
                f"Category: {match['visa_category']}"
            )

            print(
                f"Distance: {match['distance']:.4f}"
            )

            print()
            print(match["content"])
            print("-" * 70)

        print()


if __name__ == "__main__":
    main()
