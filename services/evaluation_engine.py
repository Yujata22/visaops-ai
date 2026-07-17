import re
from typing import Any

import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


_embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def _extract_context(matches: list[dict[str, Any]]) -> str:
    texts = []

    for match in matches:
        text = (
            match.get("document")
            or match.get("text")
            or match.get("content")
            or ""
        )

        if text:
            texts.append(text)

    return "\n\n".join(texts)


def calculate_groundedness(
    answer: str,
    matches: list[dict[str, Any]],
) -> float:
    if not answer.strip():
        return 0.0

    context = _extract_context(matches)

    if not context.strip():
        return 0.0

    embeddings = _embedding_model.encode(
        [answer, context],
        normalize_embeddings=True,
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    return round(max(0.0, min(float(score), 1.0)) * 100, 1)


def calculate_source_usage(answer: str) -> float:
    source_mentions = re.findall(
        r"\[Source\s+\d+\]",
        answer,
        flags=re.IGNORECASE,
    )

    if not source_mentions:
        return 0.0

    return min(len(set(source_mentions)) * 25.0, 100.0)


def calculate_safety_score(answer: str) -> float:
    answer_lower = answer.lower()

    signals = [
        "legal advice",
        "immigration attorney",
        "verify",
        "uscis",
        "general information",
    ]

    matches = sum(
        1 for signal in signals
        if signal in answer_lower
    )

    return round(matches / len(signals) * 100, 1)


def calculate_readability(answer: str) -> float:
    if not answer.strip():
        return 0.0

    grade = textstat.flesch_kincaid_grade(answer)

    if grade <= 8:
        return 100.0
    if grade <= 10:
        return 85.0
    if grade <= 12:
        return 70.0
    if grade <= 14:
        return 55.0

    return 40.0


def calculate_completeness(answer: str) -> float:
    if not answer.strip():
        return 0.0

    answer_lower = answer.lower()

    required_sections = [
        "summary",
        "consideration",
        "verify",
        "source",
    ]

    covered = sum(
        1 for section in required_sections
        if section in answer_lower
    )

    return round(covered / len(required_sections) * 100, 1)


def evaluate_answer(
    answer: str,
    matches: list[dict[str, Any]],
    latency_seconds: float,
) -> dict[str, float]:
    groundedness = calculate_groundedness(answer, matches)
    source_usage = calculate_source_usage(answer)
    safety = calculate_safety_score(answer)
    readability = calculate_readability(answer)
    completeness = calculate_completeness(answer)

    overall = (
        groundedness * 0.35
        + source_usage * 0.20
        + safety * 0.20
        + readability * 0.10
        + completeness * 0.15
    )

    return {
        "groundedness": groundedness,
        "source_usage": source_usage,
        "safety": safety,
        "readability": readability,
        "completeness": completeness,
        "latency_seconds": round(latency_seconds, 2),
        "overall_score": round(overall, 1),
    }
