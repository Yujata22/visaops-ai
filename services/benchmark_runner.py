from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from services.evaluation_engine import evaluate_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "benchmark_questions.json"
)


def load_benchmark_questions() -> list[dict[str, Any]]:
    """Load benchmark questions from the local JSON file."""
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}"
        )

    with BENCHMARK_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "Benchmark questions must be stored as a JSON list."
        )

    return questions


def run_benchmark(
    questions: list[dict[str, Any]],
    retriever: Any,
    provider_factories: list[
        tuple[str, Callable[[str, list[dict]], dict]]
    ],
    number_of_results: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Run each benchmark question against every configured model provider.
    """

    score_rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []

    for question_record in questions:
        question_id = str(
            question_record.get("id", "")
        )

        category = str(
            question_record.get("category", "General")
        )

        question = str(
            question_record.get("question", "")
        ).strip()

        if not question:
            continue

        matches = retriever.search(
            query=question,
            number_of_results=number_of_results,
        )

        for model_name, provider_factory in provider_factories:
            start_time = time.perf_counter()

            try:
                response = provider_factory(
                    question,
                    matches,
                )
            except Exception as exc:
                response = {
                    "answer": "",
                    "error": str(exc),
                }

            latency_seconds = (
                time.perf_counter() - start_time
            )

            error_message = response.get("error")

            if error_message:
                detailed_results.append(
                    {
                        "question_id": question_id,
                        "category": category,
                        "question": question,
                        "model": model_name,
                        "answer": "",
                        "error": error_message,
                    }
                )
                continue

            answer = str(
                response.get("answer", "")
            ).strip()

            metrics = evaluate_answer(
                answer=answer,
                matches=matches,
                latency_seconds=latency_seconds,
            )

            score_row = {
                "Question ID": question_id,
                "Category": category,
                "Question": question,
                "Model": model_name,
                "Overall": metrics["overall_score"],
                "Groundedness": metrics["groundedness"],
                "Source Usage": metrics["source_usage"],
                "Safety": metrics["safety"],
                "Completeness": metrics["completeness"],
                "Readability": metrics["readability"],
                "Latency (sec)": metrics["latency_seconds"],
            }

            score_rows.append(score_row)

            detailed_results.append(
                {
                    "question_id": question_id,
                    "category": category,
                    "question": question,
                    "model": model_name,
                    "answer": answer,
                    "metrics": metrics,
                    "error": None,
                }
            )

    score_dataframe = pd.DataFrame(score_rows)

    return score_dataframe, detailed_results


def summarize_benchmark(
    score_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate average benchmark scores by model."""
    if score_dataframe.empty:
        return pd.DataFrame()

    summary = (
        score_dataframe.groupby(
            "Model",
            as_index=False,
        )
        .agg(
            {
                "Overall": "mean",
                "Groundedness": "mean",
                "Source Usage": "mean",
                "Safety": "mean",
                "Completeness": "mean",
                "Readability": "mean",
                "Latency (sec)": "mean",
            }
        )
        .round(2)
        .sort_values(
            by="Overall",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return summary
