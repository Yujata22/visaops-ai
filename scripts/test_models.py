import os
import time

from services.knowledge_retriever import VisaKnowledgeRetriever
from services.model_providers import (
    generate_gemini_answer,
    generate_local_answer,
    generate_rule_based_answer,
)
from services.evaluation_engine import evaluate_answer


question = "What happens after H-1B employment ends?"

retriever = VisaKnowledgeRetriever()
matches = retriever.search(
    query=question,
    number_of_results=3,
)

providers = [
    (
        "Rule-Based",
        lambda: generate_rule_based_answer(question, matches),
    ),
    (
        "Local",
        lambda: generate_local_answer(question, matches),
    ),
]

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    providers.append(
        (
            "Gemini",
            lambda: generate_gemini_answer(
                api_key=api_key,
                question=question,
                matches=matches,
            ),
        )
    )


for provider_name, provider_function in providers:
    start = time.perf_counter()
    result = provider_function()
    latency = time.perf_counter() - start

    print("\n" + "=" * 80)
    print(provider_name)
    print("=" * 80)

    if result["error"]:
        print("ERROR:", result["error"])
        continue

    print(result["answer"])

    metrics = evaluate_answer(
        answer=result["answer"],
        matches=matches,
        latency_seconds=latency,
    )

    print("\nMetrics:")
    print(metrics)
