from functools import lru_cache
from typing import Any

from transformers import pipeline


@lru_cache(maxsize=1)
def load_local_pipeline():
    """
    Load the local text-generation model once.
    """
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-small",
    )


def _build_context(matches: list[dict[str, Any]]) -> str:
    passages = []

    for match in matches:
        text = (
            match.get("document")
            or match.get("text")
            or match.get("content")
            or ""
        )

        if text:
            passages.append(text)

    return "\n\n".join(passages)


def generate_local_answer(
    question: str,
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate a response locally using FLAN-T5 Small.
    """

    context = _build_context(matches)

    prompt = f"""
Answer the immigration question using only the context below.

Question:
{question}

Context:
{context}

Give:
1. A short educational summary
2. Important considerations
3. What should be verified with an immigration attorney

Answer:
"""

    try:
        generator = load_local_pipeline()

        response = generator(
            prompt,
            max_new_tokens=300,
            do_sample=False,
        )

        answer = response[0]["generated_text"]

        return {
            "model_name": "FLAN-T5 Small",
            "answer": answer.strip(),
            "error": None,
        }

    except Exception as exc:
        return {
            "model_name": "FLAN-T5 Small",
            "answer": "",
            "error": str(exc),
        }
