from typing import Any

from services.answer_generator import generate_educational_answer


def generate_rule_based_answer(
    question: str,
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate an answer using the existing deterministic VisaOPS engine.
    """

    result = generate_educational_answer(
        question=question,
        matches=matches,
    )

    if isinstance(result, dict):
        answer = result.get("answer") or result.get("summary") or str(result)
    else:
        answer = str(result)

    return {
        "model_name": "Rule-Based RAG",
        "answer": answer,
        "error": None,
    }
